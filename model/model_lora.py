"""
LoRA 实现（改自 minimind/model/model_lora.py）
相比原版：
  1. 支持 target_modules 指定注入层（原版只注入方阵 Linear），兼容 MiniMind 与 HF Qwen3 的层命名
  2. 支持 alpha/rank 缩放与 dropout
  3. 提供 lora_disabled() 上下文：DPO 中可用"关闭 adapter 的策略模型"充当参考模型，省一份模型显存
  4. merge_lora 同时支持写回 state_dict 与原地合并（用于推理集成）
"""
import contextlib

import torch
from torch import nn

DEFAULT_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=None, dropout=0.0):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.scaling = (alpha or rank) / rank
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化，保证训练开始时输出与原模型一致
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling

    def delta_weight(self):
        return (self.B.weight.data.float() @ self.A.weight.data.float()) * self.scaling


def _is_target(name, module, target_modules):
    if not isinstance(module, nn.Linear) or 'lora' in name or name.endswith('lm_head'):
        return False
    return target_modules is None or name.split('.')[-1] in target_modules


def apply_lora(model, rank=16, alpha=None, dropout=0.0, target_modules=DEFAULT_TARGETS):
    device = next(model.parameters()).device
    for name, module in list(model.named_modules()):
        if _is_target(name, module, target_modules):
            lora = LoRA(module.in_features, module.out_features, rank=rank, alpha=alpha, dropout=dropout).to(device)
            module.lora = lora
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                out = layer1(x)
                return out + layer2(x).to(out.dtype) if layer2.enabled else out

            module.forward = forward_with_lora


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if '.lora.' in n]


@contextlib.contextmanager
def lora_disabled(model):
    """临时关闭全部 LoRA 分支，模型退化为注入前的权重（DPO 参考模型的省显存实现）"""
    loras = [m for m in model.modules() if isinstance(m, LoRA)]
    for m in loras:
        m.enabled = False
    try:
        yield model
    finally:
        for m in loras:
            m.enabled = True


def load_lora(model, path):
    state_dict = torch.load(path, map_location=next(model.parameters()).device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if k.startswith(f'{name}.lora.')}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


@torch.no_grad()
def merge_lora_inplace(model):
    """把 LoRA 增量写回 Linear.weight 并移除 LoRA 分支，得到普通模型"""
    for module in list(model.modules()):
        lora = getattr(module, 'lora', None)
        if isinstance(module, nn.Linear) and isinstance(lora, LoRA):
            module.weight.data += lora.delta_weight().to(module.weight.dtype)
            del module.lora
            module.forward = nn.Linear.forward.__get__(module, nn.Linear)
    return model


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    merge_lora_inplace(model)
    raw_model = getattr(model, '_orig_mod', model)
    torch.save({k: v.cpu().half() for k, v in raw_model.state_dict().items()}, save_path)
