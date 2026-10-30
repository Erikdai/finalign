"""
构造 DPO 偏好对：围绕数字、单位、报告期、引用错误生成 chosen / rejected

prompt 与 SFT 同分布（train 公司 × hist 报告期，换随机种子），chosen 为金标答案，rejected 对最后一轮答案做定点扰动：
  period_value  报告期混淆：换成证据中"同一指标、其他报告期"的数值并引用该干扰块（RAG 最常见的错误）
  period_text   报告期表述错误：数值不变，报告期写错（年份错位 / 季度与半年度混淆）
  unit          单位错误：万元↔亿元↔元、%↔个百分点、元/股↔元、股↔万股
  number        数字错误：改错一位、小数点错位、相邻数字颠倒
  citation      引用错误：引用干扰块或漏掉引用
  hallucination 证据不足时编造数值（chosen 为拒答）
每个 rejected 都用评测判分器校验"确实错误"，保证偏好信号干净。
（on-policy 负例：scripts/mine_policy_negatives.py 用 SFT 模型采样，判分错误的生成可追加为 rejected）

用法：python data_pipeline/build_dpo.py --num 4000 --out data/dpo/dpo_fin.jsonl
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import random
import re
from collections import Counter

from data_pipeline.build_qa import CONV_WEIGHTS, SPLITS, fact_pools, make_sample
from data_pipeline.fin_text import PERIOD_TEXT, UNIT_CLASS
from data_pipeline.qa_builder import FactStore, QABuilder, label_of
from evaluation.metrics import score_answer

ERROR_WEIGHTS = {'period_value': 0.30, 'period_text': 0.15, 'unit': 0.20, 'number': 0.20, 'citation': 0.15}
UNIT_CONFUSION = {'元': ['万元', '亿元'], '万元': ['亿元', '元'], '亿元': ['万元', '元'], '%': ['个百分点'],
                  '个百分点': ['%'], '元/股': ['元'], '股': ['万股']}


def perturb_digits(disp, rng):
    """保持格式地改错数字：改一位 / 小数点错位 / 相邻两位颠倒"""
    neg, body = disp.startswith('-'), disp.lstrip('-')
    intp, _, frac = body.replace(',', '').partition('.')
    op = rng.choice(['digit', 'shift', 'swap'])
    if op == 'shift' and len(intp) > 1:
        v = float(body.replace(',', '')) * rng.choice([10, 0.1])
        new = f"{v:,.{len(frac)}f}" if ',' in body else f"{v:.{len(frac)}f}"
    else:
        digits = list(body)
        pos = [i for i, ch in enumerate(digits) if ch.isdigit()]
        if op == 'swap' and len(pos) >= 2:
            i = rng.randrange(len(pos) - 1)
            a, b = pos[i], pos[i + 1]
            if digits[a] != digits[b]:
                digits[a], digits[b] = digits[b], digits[a]
                return ('-' if neg else '') + ''.join(digits)
        i = rng.choice(pos[1:] if len(pos) > 1 else pos)
        digits[i] = rng.choice([d for d in '0123456789' if d != digits[i]])
        new = ''.join(digits)
    return ('-' if neg else '') + new


def wrong_period_text(ptxt, slots, rng):
    m = re.match(r'(20\d{2})年', ptxt)
    if not m:
        return None
    year = int(m.group(1))
    if slots.get('ptype') and ('季度' in ptxt or '上半年' in ptxt or '年度' in ptxt) and rng.random() < 0.5:
        others = [p for p in PERIOD_TEXT if p != slots['ptype']]
        p = rng.choice(others)
        return f'{year}年度' if p == 'FY' else f'{year}年{PERIOD_TEXT[p]}'
    return ptxt.replace(f'{year}年', f'{year + rng.choice([-1, 1])}年', 1)


def distractor_fact(store, sample, turn):
    """证据中与目标同指标、但不支持目标的数值事实（即其他报告期/其他口径的同名指标）"""
    want_cls = turn['cls']
    for pos, cid in enumerate(sample['evidence_ids'], start=1):
        if pos in turn['support']:
            continue
        for f in store.by_chunk.get(cid, []):
            metric_ok = f['metric'] == turn['metric'] or (turn['metric'] == 'np_forecast' and f['metric'] == 'np')
            if (metric_ok and UNIT_CLASS.get(f['unit']) == want_cls and isinstance(f['display'], str)
                    and (f['kind'] != 'yoy' or turn['kind'] == 'yoy')):
                return pos, f
    return None, None


def make_rejected(store, sample, rng, etype):
    turn = sample['turns'][-1]
    chosen = sample['conversations'][-1]['content']
    sl = turn['slots']
    disp, unit, cite = sl['disp'], sl['unit'], turn['cite']
    span = f'{disp}{unit}'
    if turn['kind'] == 'refusal':
        pos, f = distractor_fact(store, sample, turn)
        if f is None:
            return None, None
        return f"{sl['short']}{sl['period_text']}{label_of(f['metric'], f)}为{f['display']}{f['unit']}[{pos}]。", 'hallucination'
    if etype == 'period_value':
        pos, f = distractor_fact(store, sample, turn)
        if f is not None and span in chosen:
            new_disp = f['display'].lstrip('-') if turn['kind'] == 'yoy' else f['display']
            return chosen.replace(span, f"{new_disp}{f['unit']}", 1).replace(f'[{cite}]', f'[{pos}]'), etype
        etype = 'number'
    if etype == 'period_text':
        wp = wrong_period_text(sl['period_text'], sl, rng)
        if wp and sl['period_text'] in chosen:
            return chosen.replace(sl['period_text'], wp, 1), etype
        etype = 'unit'
    if etype == 'unit' and span in chosen and unit in UNIT_CONFUSION:
        return chosen.replace(span, f'{disp}{rng.choice(UNIT_CONFUSION[unit])}', 1), etype
    if etype in ('unit', 'number') and span in chosen:
        return chosen.replace(span, f'{perturb_digits(disp, rng)}{unit}', 1), 'number'
    # citation
    wrong = [i for i in range(1, len(sample['evidence_ids']) + 1) if i not in turn['support']]
    if wrong and rng.random() < 0.7:
        return chosen.replace(f'[{cite}]', f'[{rng.choice(wrong)}]'), 'citation'
    return chosen.replace(f'[{cite}]', ''), 'citation'


def is_wrong(text, turn):
    s = score_answer(text, turn)
    return not s['refusal_ok'] if turn['kind'] == 'refusal' else not (s['correct'] and s['unit_ok'] is not False)


EVIDENCE_HEAD, QUESTION_HEAD = '参考证据：\n', '\n\n问题：'
EVIDENCE_START = re.compile(r'^\[\d+\](?=《)', re.M)
CITATION = re.compile(r'\[(\d+)\]')


def permute_evidence(pair, rng):
    """打乱证据块顺序并同步改写 chosen / rejected 各轮回答中的引用编号，使引用位置本身不带偏好信号"""
    user = next(m['content'] for m in pair['chosen'] if m['role'] == 'user' and m['content'].startswith(EVIDENCE_HEAD))
    body, _, question = user[len(EVIDENCE_HEAD):].partition(QUESTION_HEAD)
    starts = [m.start() for m in EVIDENCE_START.finditer(body)] + [len(body)]
    blocks = [EVIDENCE_START.sub('', body[a:b].rstrip('\n'), count=1) for a, b in zip(starts, starts[1:])]
    order = list(range(len(blocks)))
    rng.shuffle(order)
    new_pos = {old + 1: new + 1 for new, old in enumerate(order)}
    new_user = EVIDENCE_HEAD + '\n'.join(f'[{i + 1}]{blocks[old]}' for i, old in enumerate(order)) + QUESTION_HEAD + question

    def renumber(text):
        return CITATION.sub(lambda m: f'[{new_pos.get(int(m.group(1)), int(m.group(1)))}]', text)

    def rewrite(messages):
        out = []
        for m in messages:
            if m['content'] == user:
                m = {**m, 'content': new_user}
            elif m['role'] == 'assistant':
                m = {**m, 'content': renumber(m['content'])}
            out.append(m)
        return out

    target = dict(pair['target'])
    target['support'] = [new_pos.get(i, i) for i in target.get('support') or []]
    if target.get('cite'):
        target['cite'] = new_pos.get(target['cite'], target['cite'])
    return {**pair, 'chosen': rewrite(pair['chosen']), 'rejected': rewrite(pair['rejected']), 'target': target}


def main():
    parser = argparse.ArgumentParser(description="构造金融问答 DPO 偏好对")
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--num', type=int, default=4000)
    parser.add_argument('--k_evidence', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1234, help="与 SFT 不同的种子，避免 prompt 完全重合")
    parser.add_argument('--out', type=str, default='data/dpo/dpo_fin.jsonl')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keep, fact_filter = SPLITS['dpo']
    store = FactStore(args.data_dir, keep=keep)
    builder = QABuilder(store, rng, k_evidence=args.k_evidence)
    pools = fact_pools(store, fact_filter)
    types, weights = zip(*CONV_WEIGHTS.items())
    etypes, ew = zip(*ERROR_WEIGHTS.items())
    out, stat, n_invalid = [], Counter(), 0
    while len(out) < args.num:
        sample = make_sample(builder, pools, rng.choices(types, weights)[0], rng)
        rejected, etype = make_rejected(store, sample, rng, rng.choices(etypes, ew)[0])
        turn = sample['turns'][-1]
        if rejected is None or rejected == sample['conversations'][-1]['content'] or not is_wrong(rejected, turn):
            n_invalid += 1
            continue
        prompt = sample['conversations'][:-1]
        out.append({'chosen': prompt + [sample['conversations'][-1]], 'rejected': prompt + [{'role': 'assistant', 'content': rejected}],
                    'error_type': etype, 'type': sample['type'], 'target': turn})
        stat[etype] += 1
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'pairs={len(out)} error_types={dict(stat)} discarded={n_invalid} -> {args.out}')


if __name__ == '__main__':
    main()
