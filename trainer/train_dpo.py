"""
DPO 偏好对齐（改自 minimind/trainer/train_dpo.py）

  策略模型：CPT 权重 + SFT LoRA 合并得到 SFT 模型，再注入一组新的 LoRA 作为可训练部分（--full_param 1 则全参训练）
  参考模型：--ref_mode copy     深拷贝一份 SFT 模型并冻结（默认）
            --ref_mode adapter  不额外拷贝，计算参考 log-prob 时临时关闭策略模型的 LoRA（大模型省一份显存）
  损失：    L = -log σ(β·[(logπ(y_w)-logπ_ref(y_w)) - (logπ(y_l)-logπ_ref(y_l))]) + α·NLL(y_w)
            α>0 时即 RPO 式的 chosen NLL 正则，防止 chosen 概率整体下降

python trainer/train_dpo.py --from_weight cpt_fin --sft_lora lora_sft --data_path data/dpo/dpo_fin.jsonl --lora_name lora_dpo
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import copy
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from dataset.lm_dataset import DPODataset, pad_collate
from model.model_lora import apply_lora, load_lora, lora_disabled, lora_parameters, merge_lora_inplace, save_lora
from trainer.trainer_utils import (
    ROOT,
    Logger,
    add_model_args,
    autocast_context,
    build_loader,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lora_path,
    save_full_weights,
    setup_seed,
)

warnings.filterwarnings('ignore')


def sequence_log_probs(model, x, y, mask):
    """逐 token log-prob 后按 mask 求和，得到每条序列在回答部分的 log π(y|x)"""
    logits = model(input_ids=x).logits
    logp = -F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction='none').view_as(y)
    return (logp * mask).sum(dim=1), mask.sum(dim=1).clamp(min=1)


def dpo_loss(policy_logps, ref_logps, beta):
    half = policy_logps.shape[0] // 2
    pi_chosen, pi_rejected = policy_logps[:half], policy_logps[half:]
    ref_chosen, ref_rejected = ref_logps[:half], ref_logps[half:]
    logits = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    loss = -F.logsigmoid(beta * logits).mean()
    chosen_rewards = beta * (pi_chosen - ref_chosen).detach()
    rejected_rewards = beta * (pi_rejected - ref_rejected).detach()
    return loss, chosen_rewards, rejected_rewards


def train_epoch(epoch, loader, iters, trainable, wandb=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=1):
        x = torch.cat([batch['x_chosen'], batch['x_rejected']], dim=0).to(args.device)
        y = torch.cat([batch['y_chosen'], batch['y_rejected']], dim=0).to(args.device)
        mask = torch.cat([batch['mask_chosen'], batch['mask_rejected']], dim=0).to(args.device).float()
        global_step = epoch * iters + step
        lr = get_lr(global_step, args.epochs * iters, args.learning_rate, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                if args.ref_mode == 'adapter':
                    with lora_disabled(model):
                        ref_logps, _ = sequence_log_probs(model, x, y, mask)
                else:
                    ref_logps, _ = sequence_log_probs(ref_model, x, y, mask)
            policy_logps, n_tokens = sequence_log_probs(model, x, y, mask)
            loss, chosen_r, rejected_r = dpo_loss(policy_logps, ref_logps, args.beta)
            if args.sft_alpha > 0:
                half = x.shape[0] // 2
                loss = loss + args.sft_alpha * (-(policy_logps[:half] / n_tokens[:half]).mean())
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            reward_acc = (chosen_r > rejected_r).float().mean().item()
            margin = (chosen_r - rejected_r).mean().item()
            # chosen 的逐 token NLL：持续上升说明 chosen 概率在被一起压低（likelihood displacement）
            half = x.shape[0] // 2
            chosen_nll = -(policy_logps[:half] / n_tokens[:half]).mean().item()
            eta_min = spend_time / step * (iters - step) / 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {loss.item() * args.accumulation_steps:.4f}, '
                   f'reward_acc: {reward_acc:.3f}, margin: {margin:.4f}, chosen_r: {chosen_r.mean().item():.4f}, '
                   f'rejected_r: {rejected_r.mean().item():.4f}, chosen_nll: {chosen_nll:.4f}, lr: {lr:.8f}, epoch_time_left: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": loss.item() * args.accumulation_steps, "reward_acc": reward_acc, "margin": margin, "learning_rate": lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            if args.full_param:
                save_full_weights(model, args, args.lora_name, tokenizer)
            else:
                save_lora(model, lora_path(args, args.lora_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinAlign DPO")
    parser.add_argument("--save_dir", type=str, default=os.path.join(ROOT, 'out'), help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_dpo", help="DPO 适配器名称（full_param=1 时为全量权重名）")
    parser.add_argument('--from_weight', default='cpt_fin', type=str, help="SFT 所基于的权重（CPT 后）")
    parser.add_argument('--sft_lora', default='lora_sft', type=str, help="SFT LoRA 名称；none 表示 from_weight 已是合并后的 SFT 权重")
    parser.add_argument("--sft_lora_rank", default=16, type=int, help="SFT LoRA 的 rank（需与训练时一致）")
    parser.add_argument("--sft_lora_alpha", default=32, type=float)
    parser.add_argument("--lora_rank", default=16, type=int, help="DPO LoRA rank")
    parser.add_argument("--lora_alpha", default=32, type=float)
    parser.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str)
    parser.add_argument("--full_param", default=0, type=int, choices=[0, 1], help="是否全参数 DPO")
    parser.add_argument("--ref_mode", default="copy", choices=["copy", "adapter"], help="参考模型实现方式")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO 中的 beta")
    parser.add_argument('--sft_alpha', default=0.0, type=float, help="chosen NLL 正则系数（RPO），0 为标准 DPO")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size（每个偏好对含 chosen/rejected 两条序列）")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="峰值学习率")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument("--data_path", type=str, default=os.path.join(ROOT, 'data/dpo/dpo_fin.jsonl'))
    parser.add_argument('--max_samples', default=0, type=int, help="最多使用的偏好对数（0 为全部）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="FinAlign-DPO")
    add_model_args(parser)
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    autocast_ctx = autocast_context(args.device, args.dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"DPO-beta{args.beta}-LR-{args.learning_rate}")

    # ========== 2. 以 SFT 权重初始化策略模型 ==========
    model, tokenizer = init_model(args, args.from_weight, device=args.device)
    if args.sft_lora != 'none':
        apply_lora(model, rank=args.sft_lora_rank, alpha=args.sft_lora_alpha, target_modules=tuple(args.lora_targets.split(',')))
        load_lora(model, lora_path(args, args.sft_lora))
        merge_lora_inplace(model)
        Logger(f'已合并 SFT LoRA: {lora_path(args, args.sft_lora)}')
    for p in model.parameters():
        p.requires_grad = bool(args.full_param)

    # ========== 3. 冻结一份 SFT 副本作为参考模型 ==========
    ref_model = None
    if args.ref_mode == 'copy' or args.full_param:
        args.ref_mode = 'copy'
        ref_model = copy.deepcopy(model).eval()
        ref_model.requires_grad_(False)
        Logger('参考模型：冻结的 SFT 副本')
    else:
        Logger('参考模型：关闭 LoRA 的策略模型（共享权重）')

    if args.full_param:
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, target_modules=tuple(args.lora_targets.split(',')))
        trainable = lora_parameters(model)
        for p in trainable:
            p.requires_grad = True
    Logger(f'策略模型可训练参数: {sum(p.numel() for p in trainable) / 1e6:.3f} M')

    # ========== 4. 数据和优化器 ==========
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if args.max_samples:
        train_ds.samples = train_ds.samples[:args.max_samples]
    loader = build_loader(train_ds, args.batch_size, pad_collate(tokenizer.pad_token_id), args.num_workers, seed=args.seed)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16' and 'cuda' in args.device))
    optimizer = optim.AdamW(trainable, lr=args.learning_rate)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    Logger(f'偏好对数: {len(train_ds)}，每轮步数: {len(loader)}')

    # ========== 5. 开始训练 ==========
    model.train()
    for epoch in range(args.epochs):
        if dist.is_initialized():
            loader.sampler.set_epoch(epoch)
        train_epoch(epoch, loader, len(loader), trainable, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
