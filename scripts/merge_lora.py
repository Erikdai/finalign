"""
适配器合并：基座(CPT) + SFT LoRA [+ DPO LoRA] → 单一权重，用于推理部署 / RAG 集成

python scripts/merge_lora.py --from_weight cpt_fin --loras lora_sft,lora_dpo --save_weight finalign
（hf 后端会保存为 HF 目录 out/finalign_hf/，可直接被 vLLM / transformers 加载）
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

import torch

from model.model_lora import apply_lora, load_lora, merge_lora_inplace
from trainer.trainer_utils import ROOT, add_model_args, init_model, lora_path, save_full_weights


def load_merged(args, device='cpu'):
    """按顺序合并多个 LoRA：每个 LoRA 都是在"前面所有权重合并后"的模型上训练的"""
    model, tokenizer = init_model(args, args.from_weight, device=device)
    for name in [x for x in args.loras.split(',') if x and x != 'none']:
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, target_modules=tuple(args.lora_targets.split(',')))
        load_lora(model, lora_path(args, name))
        merge_lora_inplace(model)
        print(f'merged {lora_path(args, name)}')
    return model.eval(), tokenizer


def add_merge_args(parser):
    parser.add_argument('--save_dir', type=str, default=os.path.join(ROOT, 'out'))
    parser.add_argument('--from_weight', type=str, default='cpt_fin', help="基座权重名（或 .pth 路径 / HF 目录）")
    parser.add_argument('--loras', type=str, default='lora_sft,lora_dpo', help="依次合并的 LoRA 名称，逗号分隔")
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=32)
    parser.add_argument('--lora_targets', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    add_model_args(parser)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器")
    add_merge_args(parser)
    parser.add_argument('--save_weight', type=str, default='finalign')
    args = parser.parse_args()
    with torch.no_grad():
        model, tokenizer = load_merged(args)
        print(f'saved -> {save_full_weights(model, args, args.save_weight, tokenizer)}')
