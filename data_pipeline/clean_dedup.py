"""
公告清洗、去重与数据划分

1. 清洗：全角数字/符号归一、数字内千分位修复、删除页眉页脚与页码行、去除多余空白、过滤过短文档
2. 去重：
   - 精确去重：清洗后文本 MD5 一致（多源转载）
   - 近重复去重：字符 5-gram MinHash + LSH，Jaccard ≥ 阈值视为同一公告（更正公告/补充公告），保留最新发布版本
3. 划分（防止信息泄漏）：
   - 按公司：公司代码哈希到 train / dev / test，三者公司不重叠
   - 按时间：报告期年份 < cutoff 为 hist，≥ cutoff 为 recent；训练只用 train 公司的 hist 部分，
     评测只问 test 公司 recent 报告期的问题（其 hist 公告仅作为检索干扰项）

用法：python data_pipeline/clean_dedup.py --in data/raw/announcements.jsonl --out data/processed/docs.jsonl
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import hashlib
import json
import re
from collections import defaultdict

import numpy as np

_FW = str.maketrans({**{chr(0xFF10 + i): str(i) for i in range(10)}, '％': '%', '．': '.', '＋': '+', '－': '-'})
PAGE_LINE_RE = re.compile(r'第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$|^\s*-?\s*\d+\s*-?\s*$|^\s*\d+\s*/\s*\d+\s*$')
_DIGIT_COMMA_RE = re.compile(r'(?<=\d)，(?=\d{3}(?!\d))')
_SPACE_RE = re.compile(r'[ \t ]+')


def clean_line(line):
    line = line.translate(_FW)
    line = _SPACE_RE.sub('', line)  # 中文公告中半角空格均为 PDF 抽取噪声
    return _DIGIT_COMMA_RE.sub(',', line)


def clean_text(text):
    out = []
    for line in text.split('\n'):
        if PAGE_LINE_RE.search(line):
            continue
        line = clean_line(line)
        if line and line not in ('□适用√不适用', '√适用□不适用'):
            out.append(line)
    return '\n'.join(out)


# ---------------- MinHash LSH ----------------
_MERSENNE = (1 << 61) - 1


class MinHashLSH:
    def __init__(self, num_perm=128, bands=32, ngram=5, seed=42):
        assert num_perm % bands == 0
        self.num_perm, self.bands, self.rows, self.ngram = num_perm, bands, num_perm // bands, ngram
        rs = np.random.RandomState(seed)
        self.a = rs.randint(1, 1 << 31, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.b = rs.randint(0, 1 << 31, size=num_perm, dtype=np.int64).astype(np.uint64)

    def signature(self, text):
        grams = {text[i:i + self.ngram] for i in range(max(1, len(text) - self.ngram + 1))}
        h = np.array([int.from_bytes(hashlib.md5(g.encode('utf-8')).digest()[:4], 'little') for g in grams], dtype=np.uint64)
        # (a*h+b) mod p，h<2^32、a<2^31，乘积不会溢出 uint64
        return (((h[:, None] * self.a[None, :]) + self.b[None, :]) % np.uint64(_MERSENNE)).min(axis=0)

    def candidate_pairs(self, sigs):
        pairs = set()
        for band in range(self.bands):
            buckets = defaultdict(list)
            for idx, sig in enumerate(sigs):
                buckets[sig[band * self.rows:(band + 1) * self.rows].tobytes()].append(idx)
            for ids in buckets.values():
                if 1 < len(ids) <= 50:
                    pairs.update((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))
        return pairs


class UnionFind:
    def __init__(self, n): self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b): self.p[self.find(a)] = self.find(b)


_NUM_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')


def numeric_jaccard(a, b):
    """公告模板高度相似，文本 Jaccard 会把同一公司不同报告期的定期报告误判为重复；
    再要求两篇公告披露的数值集合基本一致，才认定为同一公告的不同版本"""
    na, nb = set(_NUM_RE.findall(a)), set(_NUM_RE.findall(b))
    return len(na & nb) / max(1, len(na | nb))


def stable_bucket(key, seed):
    return int(hashlib.md5(f'{seed}-{key}'.encode()).hexdigest(), 16) % 10000 / 10000


def main():
    parser = argparse.ArgumentParser(description="公告清洗、去重与按公司/时间划分")
    parser.add_argument('--in', dest='inp', type=str, default='data/raw/announcements.jsonl')
    parser.add_argument('--out', type=str, default='data/processed/docs.jsonl')
    parser.add_argument('--min_chars', type=int, default=80, help="清洗后过短文档直接丢弃")
    parser.add_argument('--jaccard', type=float, default=0.85, help="近重复判定阈值（字符 5-gram）")
    parser.add_argument('--num_jaccard', type=float, default=0.9, help="近重复判定阈值（数值集合）")
    parser.add_argument('--dev_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--time_cutoff', type=int, default=2024, help="报告期年份≥该值记为 recent")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    docs, n_raw = [], 0
    with open(args.inp, encoding='utf-8') as f:
        for line in f:
            n_raw += 1
            d = json.loads(line)
            d['text'] = clean_text(d['text'])
            if len(d['text']) >= args.min_chars:
                docs.append(d)

    # 精确去重
    by_md5 = {}
    for d in sorted(docs, key=lambda x: x['publish_date'], reverse=True):
        by_md5.setdefault(hashlib.md5(d['text'].encode('utf-8')).hexdigest(), d)
    exact = list(by_md5.values())

    # 近重复去重：只在同一公司内比较，避免跨公司模板相似造成误删
    lsh = MinHashLSH()
    by_company = defaultdict(list)
    for d in exact:
        by_company[d['code']].append(d)
    kept, n_near = [], 0
    for group in by_company.values():
        sigs = [lsh.signature(d['text']) for d in group]
        uf = UnionFind(len(group))
        for i, j in lsh.candidate_pairs(sigs):
            if (sigs[i] == sigs[j]).mean() >= args.jaccard and numeric_jaccard(group[i]['text'], group[j]['text']) >= args.num_jaccard:
                uf.union(i, j)
        clusters = defaultdict(list)
        for i in range(len(group)):
            clusters[uf.find(i)].append(group[i])
        for members in clusters.values():
            members.sort(key=lambda x: x['publish_date'], reverse=True)  # 保留最新（更正后）版本
            kept.append(members[0])
            n_near += len(members) - 1

    # 划分
    for d in kept:
        r = stable_bucket(d['code'], args.seed)
        d['company_split'] = 'test' if r < args.test_ratio else 'dev' if r < args.test_ratio + args.dev_ratio else 'train'
        d['time_split'] = 'recent' if (d.get('year') or 0) >= args.time_cutoff else 'hist'
    kept.sort(key=lambda x: (x['code'], x['publish_date'], x['ann_id']))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    stat = defaultdict(int)
    for d in kept:
        stat[(d['company_split'], d['time_split'])] += 1
    report = {'raw': n_raw, 'after_clean': len(docs), 'exact_dup_removed': len(docs) - len(exact),
              'near_dup_removed': n_near, 'kept': len(kept),
              'companies': {s: len({d['code'] for d in kept if d['company_split'] == s}) for s in ('train', 'dev', 'test')},
              'docs_by_split': {f'{a}/{b}': v for (a, b), v in sorted(stat.items())}}
    with open(os.path.join(os.path.dirname(args.out), 'dedup_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
