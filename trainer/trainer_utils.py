"""
训练工具函数集合（改自 minimind/trainer/trainer_utils.py）
新增双后端：
  --backend minimind  原生 MiniMind（Qwen3 同构的 64M 小模型），权重为 out/{name}_{hidden_size}.pth
  --backend hf        任意 transformers CausalLM（如 Qwen3-8B-Base），权重目录为 HF 格式
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content, flush=True)


def get_lr(current_step, total_steps, lr, warmup_steps=0):
    if current_step < warmup_steps:
        return lr * (current_step + 1) / warmup_steps
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / max(1, total_steps))))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device, dtype):
    if 'cuda' not in device:
        return nullcontext()  # CPU 上 bf16 autocast 在无 AMX 的处理器上反而更慢
    return torch.autocast('cuda', dtype=torch.bfloat16 if dtype == 'bfloat16' else torch.float16)


def add_model_args(parser):
    parser.add_argument('--backend', default='minimind', choices=['minimind', 'hf'], help="模型后端")
    parser.add_argument('--hf_path', default='', type=str, help="hf 后端：基座模型目录，如 Qwen/Qwen3-8B-Base")
    parser.add_argument('--tokenizer_path', default=os.path.join(ROOT, 'model'), type=str, help="minimind 后端的 tokenizer 目录")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--gradient_checkpointing', default=0, type=int, choices=[0, 1], help="hf 后端大模型建议开启")


def weight_path(args, name, save_dir=None):
    """minimind: out/{name}_{hidden}.pth；hf: out/{name}_hf/（HF 目录）"""
    save_dir = save_dir or args.save_dir
    if args.backend == 'hf':
        return os.path.join(save_dir, f'{name}_hf')
    moe = '_moe' if args.use_moe else ''
    return os.path.join(save_dir, f'{name}_{args.hidden_size}{moe}.pth')


def lora_path(args, name, save_dir=None):
    """LoRA 适配器只保存低秩矩阵：minimind: out/{name}_{hidden}.pth；hf: out/{name}_hf_lora.pth"""
    save_dir = save_dir or args.save_dir
    if args.backend == 'hf':
        return os.path.join(save_dir, f'{name}_hf_lora.pth')
    return os.path.join(save_dir, f"{name}_{args.hidden_size}{'_moe' if args.use_moe else ''}.pth")


def init_model(args, from_weight='pretrain', device='cpu'):
    """from_weight: 'none' 随机初始化；权重名（在 save_dir 下按约定查找）；或直接给出 .pth 文件/HF 目录路径"""
    if args.backend == 'hf':
        path = args.hf_path if from_weight in ('none', 'base', '') else (
            from_weight if os.path.isdir(from_weight) else weight_path(args, from_weight))
        tokenizer = AutoTokenizer.from_pretrained(args.hf_path or path)
        dtype = torch.bfloat16 if 'cuda' in device else torch.float32
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            model.enable_input_require_grads()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                                   use_moe=bool(args.use_moe)))
        if from_weight != 'none':
            path = from_weight if from_weight.endswith('.pth') else weight_path(args, from_weight)
            model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
            Logger(f'加载权重: {path}')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    Logger(f'模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M')
    return model.to(device), tokenizer


def unwrap(model):
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(model, '_orig_mod', model)


def save_full_weights(model, args, name, tokenizer=None):
    raw = unwrap(model)
    path = weight_path(args, name)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if args.backend == 'hf':
        raw.save_pretrained(path, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
    else:
        tmp = path + '.tmp'
        torch.save({k: v.detach().half().cpu() for k, v in raw.state_dict().items()}, tmp)
        os.replace(tmp, path)
    return path


def causal_lm_forward(model, input_ids, labels=None, attention_mask=None):
    """统一两种后端的输出：返回 (logits, loss, aux_loss)"""
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    aux = getattr(out, 'aux_loss', None)
    if aux is None:
        aux = out.logits.new_zeros(())
    return out.logits, out.loss, aux


@torch.no_grad()
def evaluate_loss(model, loader, device, autocast_ctx, max_batches=50):
    model.eval()
    total, n = 0.0, 0
    for i, (input_ids, labels) in enumerate(loader):
        if i >= max_batches:
            break
        with autocast_ctx:
            _, loss, _ = causal_lm_forward(model, input_ids.to(device), labels.to(device))
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


class LengthGroupedSampler(torch.utils.data.Sampler):
    """同 HF group_by_length：先打乱，再在每 mega_batch（batch_size×50 条）内按长度降序排，batch 内长度接近，动态 padding 浪费少"""

    def __init__(self, lengths, batch_size, seed=42, mega=50):
        self.lengths, self.batch_size, self.seed, self.mega, self.epoch = lengths, batch_size, seed, mega, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.randperm(len(self.lengths), generator=g).tolist()
        m = self.batch_size * self.mega
        return iter([j for i in range(0, len(idx), m) for j in sorted(idx[i:i + m], key=lambda j: -self.lengths[j])])


def build_loader(dataset, batch_size, collate_fn, num_workers=0, shuffle=True, seed=42, lengths=None):
    """lengths 非空时按长度分组采样（仅单机）"""
    from torch.utils.data import DataLoader, DistributedSampler
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed) if dist.is_initialized() else None
    if sampler is None and shuffle and lengths is not None:
        sampler = LengthGroupedSampler(lengths, batch_size, seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler,
                      collate_fn=collate_fn, num_workers=num_workers, generator=g, drop_last=False)
