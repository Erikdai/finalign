"""
构造 SFT 多轮会话 / 验证集 / 评测集

  --mode sft   train 公司 × hist 报告期，默认 13,600 条多轮会话（问题 + 检索证据 + 引用答案）
  --mode dev   dev 公司 × recent 报告期，用于验证集 loss
  --mode eval  test 公司 × recent 报告期，单轮问答 500 条 + 拒答 50 条（公司、时间双重隔离）

用法：
  python data_pipeline/build_qa.py --mode sft --num 13600 --out data/sft/sft_fin.jsonl
  python data_pipeline/build_qa.py --mode eval --num 500 --num_refusal 50 --out data/eval/finqa_test.jsonl
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import random
from collections import Counter

from data_pipeline.fin_text import UNIT_CLASS
from data_pipeline.qa_builder import TABLE_METRICS, FactStore, QABuilder

SPLITS = {
    'sft': (lambda r: r['company_split'] == 'train' and r['time_split'] == 'hist', lambda _: True),
    'dpo': (lambda r: r['company_split'] == 'train' and r['time_split'] == 'hist', lambda _: True),
    'dev': (lambda r: r['company_split'] == 'dev', lambda f: f['time_split'] == 'recent'),
    'eval': (lambda r: r['company_split'] == 'test', lambda f: f['time_split'] == 'recent'),
}
# 会话类型分布：多轮定期报告问答为主，覆盖同比、业绩预告、分红、回购与拒答
CONV_WEIGHTS = {'multi': 0.55, 'value_any': 0.12, 'yoy': 0.10, 'forecast': 0.06, 'dividend': 0.07, 'buyback': 0.05, 'refusal': 0.05}
EVAL_WEIGHTS = {'value_any': 0.50, 'yoy': 0.20, 'forecast': 0.08, 'dividend': 0.12, 'buyback': 0.10}


def fact_pools(store, fact_filter):
    pools = {'table': [], 'value_any': [], 'yoy': [], 'forecast': [], 'dividend': [], 'buyback': []}
    for f in store.facts:
        if not fact_filter(f):
            continue
        k, m = f['kind'], f['metric']
        if k == 'value' and m in TABLE_METRICS and not f.get('narrative'):
            pools['table'].append(f)
        if k in ('value', 'prev'):
            pools['value_any'].append(f)
        elif k == 'yoy':
            pools['yoy'].append(f)
        elif k == 'forecast':
            pools['forecast'].append(f)
        elif k == 'dividend' and m != 'shares':
            pools['dividend'].append(f)
        elif k == 'buyback':
            pools['buyback'].append(f)
    return pools


def make_sample(builder, pools, ctype, rng, max_turns=3):
    if ctype == 'multi':
        f = rng.choice(pools['table'])
        turns, cur = [builder.value_turn(f)], f
        n = rng.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0] if max_turns > 1 else 1
        for _ in range(n - 1):
            t, nxt = builder.follow_up(cur, asked_metrics={x.slots['metric'] for x in turns})
            if t is None:
                break
            turns.append(t)
            cur = nxt
    elif ctype == 'refusal':
        f = rng.choice(pools['value_any'])
        turns = [builder.refusal_turn(f)]
    else:
        f = rng.choice(pools[ctype])
        fn = {'value_any': builder.value_turn, 'yoy': builder.yoy_turn, 'forecast': builder.forecast_turn,
              'dividend': builder.dividend_turn, 'buyback': builder.buyback_turn}[ctype]
        turns = [fn(f)]
    query = ' '.join(t.question.replace(f['short_name'], '') + ' ' + t.slots['period_text'] for t in turns)
    sample = builder.assemble(turns, f['code'], query)
    sample['type'] = ctype
    return sample


def main():
    parser = argparse.ArgumentParser(description="构造 SFT / dev / eval 问答数据")
    parser.add_argument('--mode', choices=['sft', 'dev', 'eval'], default='sft')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--num', type=int, default=13600)
    parser.add_argument('--num_refusal', type=int, default=50, help="eval 模式下额外的拒答样本数")
    parser.add_argument('--k_evidence', type=int, default=3, help="每条样本的证据块数（含难负例）")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='data/sft/sft_fin.jsonl')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keep, fact_filter = SPLITS[args.mode]
    store = FactStore(args.data_dir, keep=keep)
    builder = QABuilder(store, rng, k_evidence=args.k_evidence)
    pools = fact_pools(store, fact_filter)
    print({k: len(v) for k, v in pools.items()})

    samples, seen = [], set()
    if args.mode in ('sft', 'dev'):
        types, weights = zip(*CONV_WEIGHTS.items())
        while len(samples) < args.num:
            s = make_sample(builder, pools, rng.choices(types, weights)[0], rng)
            sig = s['conversations'][1]['content'][-60:] + s['conversations'][2]['content']
            if sig in seen:
                continue
            seen.add(sig)
            samples.append(s)
    else:
        # 评测：单轮，按事实语义键去重，类型按配额抽样（池子不足时由 value_any 补齐）
        quotas = {t: int(args.num * w) for t, w in EVAL_WEIGHTS.items()}
        for ctype in ('forecast', 'dividend', 'buyback', 'yoy', 'value_any'):
            if ctype == 'value_any':
                quotas[ctype] = args.num - len(samples)
            pool = pools[ctype][:]
            rng.shuffle(pool)
            got = 0
            for f in pool:
                if got >= quotas[ctype]:
                    break
                if (f['code'], f['key']) in seen:
                    continue
                seen.add((f['code'], f['key']))
                pools_one = {**pools, ctype: [f]}
                samples.append(make_sample(builder, pools_one, ctype, rng, max_turns=1))
                got += 1
        refusal_pool = pools['value_any'][:]
        rng.shuffle(refusal_pool)
        for f in refusal_pool[:args.num_refusal]:
            samples.append(make_sample(builder, {**pools, 'value_any': [f]}, 'refusal', rng))
        for i, s in enumerate(samples):
            s['id'] = f'finqa-{i:04d}'
            # 记录证据中"同名指标、其他报告期/口径"的数值，评测时用于把错误归因到报告期混淆
            t = s['turns'][-1]
            metric = 'np' if t['metric'] == 'np_forecast' else t['metric']
            t['distractor_values'] = sorted({f['value'] for pos, cid in enumerate(s['evidence_ids'], start=1) if pos not in t['support']
                                             for f in store.by_chunk.get(cid, []) if f['metric'] == metric
                                             and UNIT_CLASS.get(f['unit']) == t['cls'] and not isinstance(f['value'], list)})

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    n_turns = Counter(len(s['turns']) for s in samples)
    print(f'mode={args.mode} samples={len(samples)} types={dict(Counter(s["type"] for s in samples))} turns={dict(n_turns)} -> {args.out}')


if __name__ == '__main__':
    main()
