"""
on-policy 负例挖掘：用 SFT 模型对 DPO prompt 采样，判分器判错的生成作为 rejected，金标作为 chosen。
规则扰动的 rejected 覆盖"可能犯的错"，on-policy 负例覆盖"模型实际会犯的错"。

python scripts/mine_policy_negatives.py --from_weight cpt_fin --loras lora_sft --dpo_path data/dpo/dpo_fin.jsonl \
    --out data/dpo/dpo_fin_onpolicy.jsonl --num_prompts 1000 --samples 2 --temperature 0.8 --mix_out data/dpo/dpo_fin_mix.jsonl
--mix_out：on-policy 负例 + 等量规则扰动偏好对（取自未用于挖掘的 prompt）打乱后写出，作为 DPO 训练集；
--permute_evidence 1 时每对的证据顺序随机打乱、引用编号同步改写，避免 DPO 学到"引用哪个位置"的捷径
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import random

import torch

from data_pipeline.build_dpo import permute_evidence
from dataset.lm_dataset import IM_END, load_jsonl, render_chat
from evaluation.metrics import score_answer
from scripts.merge_lora import add_merge_args, load_merged


@torch.inference_mode()
def sample(model, tokenizer, prompt, n, temperature, max_new_tokens, device):
    enc = tokenizer([prompt] * n, return_tensors='pt', add_special_tokens=False).to(device)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    out = model.generate(inputs=enc['input_ids'], attention_mask=enc['attention_mask'], max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, top_p=0.95, eos_token_id=im_end, pad_token_id=tokenizer.pad_token_id)
    texts = []
    for row in out[:, enc['input_ids'].shape[1]:].tolist():
        row = row[:row.index(im_end)] if im_end in row else row
        texts.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return texts


def main():
    parser = argparse.ArgumentParser(description="SFT 模型 on-policy 负例挖掘")
    add_merge_args(parser)
    parser.add_argument('--dpo_path', type=str, default='data/dpo/dpo_fin.jsonl', help="提供 prompt 与金标的偏好数据")
    parser.add_argument('--out', type=str, default='data/dpo/dpo_fin_onpolicy.jsonl')
    parser.add_argument('--mix_out', type=str, default='', help="非空则另存 on-policy + 等量规则扰动的混合偏好数据")
    parser.add_argument('--permute_evidence', type=int, default=1, choices=[0, 1], help="混合数据中打乱证据顺序并改写引用编号")
    parser.add_argument('--num_prompts', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=2, help="每个 prompt 采样数")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_new_tokens', type=int, default=120)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_merged(args, device=args.device)
    rows = load_jsonl(args.dpo_path)
    random.shuffle(rows)
    kept = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(rows[:args.num_prompts]):
            prompt = render_chat(r['chosen'][:-1], add_generation_prompt=True)[0]
            for text in sample(model, tokenizer, prompt, args.samples, args.temperature, args.max_new_tokens, args.device):
                s = score_answer(text, r['target'])
                wrong = not s['refusal_ok'] if r['target']['kind'] == 'refusal' else not s['correct']
                if wrong and text:
                    f.write(json.dumps({**r, 'rejected': r['chosen'][:-1] + [{'role': 'assistant', 'content': text}],
                                        'error_type': 'on_policy'}, ensure_ascii=False) + '\n')
                    kept += 1
                    break
            print(f'\r{i + 1}/{args.num_prompts} mined={kept}', end='', flush=True)
    print(f'\non-policy negatives: {kept} -> {args.out}')
    if args.mix_out:
        mixed = load_jsonl(args.out) + rows[args.num_prompts:args.num_prompts + kept]
        random.shuffle(mixed)
        if args.permute_evidence:
            perm_rng = random.Random(args.seed)
            mixed = [permute_evidence(r, perm_rng) for r in mixed]
        with open(args.mix_out, 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in mixed)
        print(f'mixed {len(mixed)} pairs -> {args.mix_out}')


if __name__ == '__main__':
    main()
