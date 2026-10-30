"""
领域继续预训练（CPT，改自 minimind/trainer/train_pretrain.py）
从已有预训练权重（minimind-3 的 pretrain_768.pth，或 hf 后端的 Qwen3-8B-Base）出发，在清洗后的公告语料上继续做 next-token 预训练。

python trainer/train_pretrain.py --from_weight pretrain --data_path data/cpt/pretrain_fin.jsonl --save_weight cpt_fin
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

from dataset.lm_dataset import PretrainDataset, pad_collate
from trainer.trainer_utils import (
    ROOT,
    Logger,
    add_model_args,
    autocast_context,
    build_loader,
    causal_lm_forward,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    save_full_weights,
    setup_seed,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, wandb=None):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            eta_min = spend_time / step * (iters - step) / 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {lr:.8f}, epoch_time_left: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "learning_rate": lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            path = save_full_weights(model, args, args.save_weight, tokenizer)
            Logger(f'保存权重: {path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinAlign 领域继续预训练")
    parser.add_argument("--save_dir", type=str, default=os.path.join(ROOT, 'out'), help="模型保存目录")
    parser.add_argument('--save_weight', default='cpt_fin', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="峰值学习率（继续预训练宜小于从头预训练）")
    parser.add_argument("--warmup_steps", type=int, default=20, help="warmup 步数")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=20, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度（tokens）")
    parser.add_argument("--data_path", type=str, default=os.path.join(ROOT, 'data/cpt/pretrain_fin.jsonl'), help="CPT 语料")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="初始化权重：pretrain=minimind-3 预训练权重；hf 后端填 base")
    parser.add_argument('--max_steps', default=0, type=int, help="每轮最多训练步数（0 为不限制，CPU 调试用）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 swanlab/wandb 记录")
    parser.add_argument("--wandb_project", type=str, default="FinAlign-CPT")
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
        wandb.init(project=args.wandb_project, name=f"CPT-{args.backend}-LR-{args.learning_rate}")

    # ========== 2. 模型、数据、优化器 ==========
    model, tokenizer = init_model(args, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = build_loader(train_ds, args.batch_size, pad_collate(tokenizer.pad_token_id), args.num_workers, seed=args.seed)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16' and 'cuda' in args.device))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    iters = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    Logger(f'CPT 样本数: {len(train_ds)}，每轮步数: {iters}')

    # ========== 3. 开始训练 ==========
    model.train()
    for epoch in range(args.epochs):
        if dist.is_initialized():
            loader.sampler.set_epoch(epoch)
        limited = (b for i, b in enumerate(loader) if i < iters)
        train_epoch(epoch, limited, iters, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
