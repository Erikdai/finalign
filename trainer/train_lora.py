"""
LoRA 监督微调（改自 minimind/trainer/train_lora.py）
在 CPT 权重上注入 LoRA（默认 q/k/v/o/gate/up/down 全部线性层），只训练低秩矩阵；loss 仅计算 assistant 回答部分。

python trainer/train_lora.py --from_weight cpt_fin --data_path data/sft/sft_fin.jsonl --lora_name lora_sft
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from dataset.lm_dataset import SFTDataset, pad_collate, render_chat
from model.model_lora import apply_lora, load_lora, lora_parameters, save_lora
from trainer.trainer_utils import (
    ROOT,
    Logger,
    add_model_args,
    autocast_context,
    build_loader,
    causal_lm_forward,
    evaluate_loss,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lora_path,
    setup_seed,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids, labels = input_ids.to(args.device), labels.to(args.device)
        global_step = epoch * iters + step
        lr = get_lr(global_step, args.epochs * iters, args.learning_rate, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            _, loss, aux_loss = causal_lm_forward(model, input_ids, labels)
            loss = (loss + aux_loss) / args.accumulation_steps
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            eta_min = spend_time / step * (iters - step) / 60
            msg = f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {lr:.8f}, epoch_time_left: {eta_min:.1f}min'
            log = {"loss": current_loss, "learning_rate": lr}
            if dev_loader is not None and (step % args.eval_interval == 0 or step == iters):
                log['dev_loss'] = evaluate_loss(model, dev_loader, args.device, autocast_ctx, max_batches=args.eval_batches)
                msg += f", dev_loss: {log['dev_loss']:.4f}"
            Logger(msg)
            if wandb:
                wandb.log(log)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_lora(model, lora_path(args, args.lora_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinAlign LoRA SFT")
    parser.add_argument("--save_dir", type=str, default=os.path.join(ROOT, 'out'), help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_sft", help="LoRA 权重名称")
    parser.add_argument("--lora_rank", default=16, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=32, type=float, help="LoRA alpha（缩放 = alpha / rank）")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="LoRA dropout")
    parser.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str, help="注入 LoRA 的层名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="峰值学习率")
    parser.add_argument("--warmup_steps", type=int, default=20, help="warmup 步数")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=20, help="日志打印间隔")
    parser.add_argument("--eval_interval", type=int, default=200, help="验证集 loss 计算间隔")
    parser.add_argument("--eval_batches", type=int, default=10, help="每次验证使用的 batch 数")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（tokens）")
    parser.add_argument("--data_path", type=str, default=os.path.join(ROOT, 'data/sft/sft_fin.jsonl'), help="SFT 数据")
    parser.add_argument("--dev_path", type=str, default=os.path.join(ROOT, 'data/sft/dev_fin.jsonl'), help="验证集（可为空）")
    parser.add_argument('--from_weight', default='cpt_fin', type=str, help="基于哪个权重训练（CPT 后的权重）")
    parser.add_argument('--max_samples', default=0, type=int, help="最多使用的训练样本数（0 为全部，CPU 调试用）")
    parser.add_argument('--skip_samples', default=0, type=int, help="跳过数据前 N 条（分段续训时跳过已训练的部分）")
    parser.add_argument('--init_lora', default='', type=str, help="从已有 LoRA 继续训练（如 lora_sft），为空则随机初始化")
    parser.add_argument('--group_by_length', default=0, type=int, choices=[0, 1], help="按长度分组采样，减少 padding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 swanlab/wandb 记录")
    parser.add_argument("--wandb_project", type=str, default="FinAlign-SFT")
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
        wandb.init(project=args.wandb_project, name=f"SFT-{args.lora_name}-r{args.lora_rank}-LR-{args.learning_rate}")

    # ========== 2. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    model, tokenizer = init_model(args, args.from_weight, device=args.device)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout, target_modules=tuple(args.lora_targets.split(',')))
    if args.init_lora:
        load_lora(model, lora_path(args, args.init_lora))
        Logger(f'从 {lora_path(args, args.init_lora)} 继续训练')
    lora_params = lora_parameters(model)
    for p in lora_params:
        p.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters())
    lora_count = sum(p.numel() for p in lora_params)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M，LoRA 参数量: {lora_count / 1e6:.3f} M（{lora_count / total_params * 100:.2f}%）")

    # ========== 3. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_ds.samples = train_ds.samples[args.skip_samples:]
    if args.max_samples:
        train_ds.samples = train_ds.samples[:args.max_samples]
    collate = pad_collate(tokenizer.pad_token_id)
    lengths = [len(render_chat(s['conversations'])[0]) for s in train_ds.samples] if args.group_by_length else None
    loader = build_loader(train_ds, args.batch_size, collate, args.num_workers, seed=args.seed, lengths=lengths)
    dev_loader = None
    if args.dev_path and os.path.exists(args.dev_path):
        dev_loader = build_loader(SFTDataset(args.dev_path, tokenizer, max_length=args.max_seq_len), args.batch_size, collate, shuffle=False)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16' and 'cuda' in args.device))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=0.0)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    Logger(f'SFT 样本数: {len(train_ds)}，每轮步数: {len(loader)}')

    # ========== 4. 开始训练 ==========
    model.train()
    for epoch in range(args.epochs):
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        train_epoch(epoch, loader, len(loader), lora_params, wandb)
    Logger(f'LoRA 权重已保存: {lora_path(args, args.lora_name)}')

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
