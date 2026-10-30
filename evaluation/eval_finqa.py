"""
金融公告问答评测：贪心解码生成答案 → 数值 / 单位 / 报告期 / 引用 / 拒答 判分

python evaluation/eval_finqa.py --from_weight cpt_fin --loras lora_sft --tag sft
python evaluation/eval_finqa.py --from_weight cpt_fin --loras lora_sft,lora_dpo --tag sft_dpo
结果写入 results/{tag}/predictions.jsonl 与 results/{tag}/metrics.json
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import time
from collections import defaultdict

import torch

from data_pipeline.fin_text import extract_values, values_equal
from dataset.lm_dataset import IM_END, load_jsonl, render_chat
from evaluation.metrics import aggregate, score_answer
from scripts.merge_lora import add_merge_args, load_merged
from trainer.trainer_utils import ROOT


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens, device):
    tokenizer.padding_side = 'left'
    enc = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    out = model.generate(inputs=enc['input_ids'], attention_mask=enc['attention_mask'], max_new_tokens=max_new_tokens,
                         do_sample=False, eos_token_id=im_end, pad_token_id=tokenizer.pad_token_id)
    texts = []
    for row in out[:, enc['input_ids'].shape[1]:]:
        ids = row.tolist()
        if im_end in ids:
            ids = ids[:ids.index(im_end)]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    return texts


def distractor_hit(pred, target):
    """错误归因：回答中的数值是否恰好等于证据里其他报告期/口径的同名指标（报告期混淆）"""
    vals = [m for m in extract_values(pred) if m['cls'] == target['cls']]
    return bool(vals) and any(values_equal(vals[0]['value'], v) for v in target.get('distractor_values', []))


def main():
    parser = argparse.ArgumentParser(description="FinAlign 金融问答评测")
    add_merge_args(parser)
    parser.add_argument('--eval_path', type=str, default=os.path.join(ROOT, 'data/eval/finqa_test.jsonl'))
    parser.add_argument('--tag', type=str, required=True, help="结果目录名")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=120)
    parser.add_argument('--limit', type=int, default=0, help="只评测前 N 条（调试用）")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model, tokenizer = load_merged(args, device=args.device)
    if 'cuda' in args.device:
        model = model.to(torch.bfloat16)
    samples = load_jsonl(args.eval_path)
    if args.limit:
        samples = samples[:args.limit]
    prompts = [render_chat(s['conversations'][:-1], add_generation_prompt=True)[0] for s in samples]
    order = sorted(range(len(samples)), key=lambda i: len(prompts[i]))  # 按长度分桶，减少 padding
    preds = [None] * len(samples)
    t0 = time.time()
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        for i, text in zip(idx, generate_batch(model, tokenizer, [prompts[i] for i in idx], args.max_new_tokens, args.device)):
            preds[i] = text
        print(f'\r{min(b + args.batch_size, len(order))}/{len(order)}  {time.time() - t0:.0f}s', end='', flush=True)
    print()

    out_dir = os.path.join(ROOT, 'results', args.tag)
    os.makedirs(out_dir, exist_ok=True)
    rows, by_type = [], defaultdict(list)
    for s, pred in zip(samples, preds):
        target = s['turns'][-1]
        sc = score_answer(pred, target)
        if target['kind'] != 'refusal' and not sc['num_ok']:
            sc['distractor_confusion'] = distractor_hit(pred, target)
        rows.append({'id': s['id'], 'type': s['type'], 'question': s['conversations'][-2]['content'].split('问题：')[-1],
                     'reference': s['conversations'][-1]['content'], 'prediction': pred, 'score': sc})
        by_type[s['type']].append(sc)
    answerable = [r['score'] for r in rows if r['type'] != 'refusal']
    refusal = [r['score'] for r in rows if r['type'] == 'refusal']
    wrong = [r for r in answerable if not r['num_ok']]
    metrics = {
        'tag': args.tag, 'model': {'from_weight': args.from_weight, 'loras': args.loras, 'backend': args.backend},
        'answerable': aggregate(answerable),
        'refusal': aggregate(refusal),
        'num_errors_from_distractor': sum(r.get('distractor_confusion', False) for r in wrong),
        'num_errors': len(wrong),
        'by_type': {t: aggregate(v) for t, v in sorted(by_type.items())},
        'seconds': round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, 'predictions.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    a = metrics['answerable']
    print(f"[{args.tag}] 数值准确率 {a.get('num_acc')}% | 引用正确率 {a.get('cite_acc')}% | 报告期准确率 {a.get('period_acc')}% | "
          f"单位正确率 {a.get('unit_acc')}% | 完全正确 {a.get('correct_acc')}% | 拒答准确率 {metrics['refusal'].get('refusal_acc')}%")


if __name__ == '__main__':
    main()
