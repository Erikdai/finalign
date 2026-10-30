"""
领域继续预训练（CPT）语料：train 公司 × hist 报告期的清洗后公告，按行切成不超过 max_chars 的片段
可选混入通用语料（如 minimind 的 pretrain_t2t_mini.jsonl）缓解灾难性遗忘。

用法：python data_pipeline/build_cpt.py --out data/cpt/pretrain_fin.jsonl [--general_path xxx.jsonl --general_ratio 0.2]
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import random


def segments(text, max_chars):
    buf = ''
    for line in text.split('\n'):
        if buf and len(buf) + len(line) + 1 > max_chars:
            yield buf
            buf = line
        else:
            buf = f'{buf}\n{line}' if buf else line
    if buf:
        yield buf


def main():
    parser = argparse.ArgumentParser(description="构造领域继续预训练语料")
    parser.add_argument('--docs', type=str, default='data/processed/docs.jsonl')
    parser.add_argument('--out', type=str, default='data/cpt/pretrain_fin.jsonl')
    parser.add_argument('--max_chars', type=int, default=450, help="片段最大字符数（金融文本数字多，约 1 字符/token）")
    parser.add_argument('--max_docs', type=int, default=0, help="最多使用的公告数（0 为全部，CPU 调试时可调小）")
    parser.add_argument('--general_path', type=str, default='', help="可选：通用预训练语料 jsonl（text 字段）")
    parser.add_argument('--general_ratio', type=float, default=0.0, help="通用语料占比")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    with open(args.docs, encoding='utf-8') as f:
        docs = [json.loads(line) for line in f]
    docs = [d for d in docs if d['company_split'] == 'train' and d['time_split'] == 'hist']
    rng.shuffle(docs)
    if args.max_docs:
        docs = docs[:args.max_docs]
    rows = [{'text': seg} for d in docs for seg in segments(d['text'], args.max_chars)]
    n_fin = len(rows)
    if args.general_path and args.general_ratio > 0:
        n_gen = int(n_fin * args.general_ratio / (1 - args.general_ratio))
        with open(args.general_path, encoding='utf-8') as f:
            general = [json.loads(line)['text'] for _, line in zip(range(n_gen * 5), f)]
        rows += [{'text': t} for t in rng.sample(general, min(n_gen, len(general)))]
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'docs={len(docs)} fin_segments={n_fin} total={len(rows)} chars={sum(len(r["text"]) for r in rows)} -> {args.out}')


if __name__ == '__main__':
    main()
