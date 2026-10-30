"""
数据集（改自 minimind/dataset/lm_dataset.py）
与原版的区别：
  1. 对话统一手工渲染为 ChatML（<|im_start|>role\\n...<|im_end|>\\n），不依赖 tokenizer 的 chat_template，
     MiniMind tokenizer 与 Qwen3 tokenizer 走同一套逻辑；不插入 <think> 标签
  2. 用 offset_mapping 定位 assistant 内容生成 labels / loss_mask，避免分段 tokenize 带来的边界不一致
  3. 不在 __getitem__ 里 pad 到 max_length，由 collate 按 batch 内最长样本动态 padding（CPU 上训练提速明显）
"""
import json

import torch
from torch.utils.data import Dataset

IM_START, IM_END = '<|im_start|>', '<|im_end|>'


def render_chat(messages, add_generation_prompt=False):
    """返回 (文本, assistant 可训练区间列表[(start, end)])；区间覆盖 assistant 内容及其 <|im_end|>"""
    text, spans = '', []
    for m in messages:
        text += f"{IM_START}{m['role']}\n"
        start = len(text)
        text += f"{m['content']}{IM_END}"
        if m['role'] == 'assistant':
            spans.append((start, len(text)))
        text += '\n'
    if add_generation_prompt:
        text += f'{IM_START}assistant\n'
    return text, spans


def tokenize_with_mask(tokenizer, messages, max_length, train_last_only=False):
    text, spans = render_chat(messages)
    if train_last_only:
        spans = spans[-1:]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_length)
    mask = [int(any(s <= a and b <= e and b > a for s, e in spans)) for a, b in enc['offset_mapping']]
    return enc['input_ids'], mask


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_jsonl(data_path)
        self.bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tokens = self.tokenizer(str(self.samples[index]['text']), add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        input_ids = torch.tensor(self.bos + tokens + [self.tokenizer.eos_token_id], dtype=torch.long)
        return input_ids, input_ids.clone()


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=768):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_jsonl(jsonl_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ids, mask = tokenize_with_mask(self.tokenizer, self.samples[index]['conversations'], self.max_length)
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[torch.tensor(mask) == 0] = -100
        return input_ids, labels


class DPODataset(Dataset):
    """chosen / rejected 为完整消息列表（minimind 格式），只对最后一轮 assistant 计算 log-prob"""

    def __init__(self, file_path, tokenizer, max_length=768):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_jsonl(file_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        out = {}
        for name in ('chosen', 'rejected'):
            ids, mask = tokenize_with_mask(self.tokenizer, sample[name], self.max_length, train_last_only=True)
            # 预测下一个 token：x 去掉最后一个，y/mask 去掉第一个
            out[f'x_{name}'] = torch.tensor(ids[:-1], dtype=torch.long)
            out[f'y_{name}'] = torch.tensor(ids[1:], dtype=torch.long)
            out[f'mask_{name}'] = torch.tensor(mask[1:], dtype=torch.long)
        return out


def pad_collate(pad_id):
    """动态 padding：LM 样本 (input_ids, labels) 或 DPO 字典样本"""
    def _pad(seqs, value):
        n = max(len(s) for s in seqs)
        return torch.stack([torch.cat([s, s.new_full((n - len(s),), value)]) for s in seqs])

    def collate(batch):
        if isinstance(batch[0], dict):
            out = {}
            for k in batch[0]:
                value = pad_id if k.startswith(('x_', 'y_')) else 0
                out[k] = _pad([b[k] for b in batch], value)
            # chosen 与 rejected 拼成一个 batch 前需等长
            n = max(out['x_chosen'].shape[1], out['x_rejected'].shape[1])
            for k, v in out.items():
                if v.shape[1] < n:
                    fill = pad_id if k.startswith(('x_', 'y_')) else 0
                    out[k] = torch.cat([v, v.new_full((v.shape[0], n - v.shape[1]), fill)], dim=1)
            return out
        input_ids = _pad([b[0] for b in batch], pad_id)
        labels = _pad([b[1] for b in batch], -100)
        return input_ids, labels
    return collate
