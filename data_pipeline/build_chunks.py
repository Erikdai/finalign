"""
公告切块 + 事实定位

输出：
  data/processed/chunks.jsonl  每行一个证据块（附公司、报告期、划分等元数据）
  data/processed/facts.jsonl   每行一个事实，带所在 chunk_id 与"语义键" key
key 相同的事实描述的是同一个数（例如 2023 年报表格中的"2023年营业收入"、正文中的"营业收入xx亿元"、
2024 年报表格中的"2023年"列），它们所在的块都是该问题的合法引用来源（support set）。

用法：python data_pipeline/build_chunks.py --docs data/processed/docs.jsonl --max_chars 100
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json

from data_pipeline.clean_dedup import clean_line
from rag.retriever import chunk_document

META_KEYS = ('ann_id', 'code', 'short_name', 'title', 'doc_type', 'year', 'ptype', 'publish_date', 'company_split', 'time_split')


def fact_key(f):
    if f['kind'] in ('value', 'prev'):
        return f"{f['metric']}|{f['year']}|{f['ptype']}|val"
    if f['kind'] == 'yoy':
        return f"{f['metric']}|{f['year']}|{f['ptype']}|yoy"
    if f['kind'] == 'buyback':
        return f"{f['metric']}|{f['date']}|bb"
    return f"{f['metric']}|{f['year']}|{f['ptype']}|{f['kind']}"


def main():
    parser = argparse.ArgumentParser(description="公告切块与事实定位")
    parser.add_argument('--docs', type=str, default='data/processed/docs.jsonl')
    parser.add_argument('--out_dir', type=str, default='data/processed')
    parser.add_argument('--max_chars', type=int, default=100, help="块的目标最大字符数（行不切断）")
    args = parser.parse_args()

    chunks, facts, n_unlocated = [], [], 0
    with open(args.docs, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            doc_chunks = []
            for j, text in enumerate(chunk_document(d['text'], max_chars=args.max_chars)):
                c = {'chunk_id': f"{d['ann_id']}-{j:02d}", **{k: d.get(k) for k in META_KEYS}, 'text': text}
                doc_chunks.append(c)
            chunks.extend(doc_chunks)
            for i, fct in enumerate(d.get('facts', [])):
                anchor = clean_line(fct['anchor'])
                host = next((c for c in doc_chunks if anchor in c['text']), None)
                if host is None:
                    n_unlocated += 1
                    continue
                rec = {k: v for k, v in fct.items() if k != 'anchor'}
                rec.update(fact_id=f"{d['ann_id']}-f{i:02d}", chunk_id=host['chunk_id'], ann_id=d['ann_id'], code=d['code'],
                           short_name=d['short_name'], doc_type=d['doc_type'], company_split=d['company_split'],
                           time_split=d['time_split'], key=fact_key(fct))
                facts.append(rec)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (('chunks', chunks), ('facts', facts)):
        with open(os.path.join(args.out_dir, f'{name}.jsonl'), 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    lens = sorted(len(c['text']) for c in chunks)
    print(f'chunks={len(chunks)} facts={len(facts)} unlocated_facts={n_unlocated} '
          f'chunk_chars p50={lens[len(lens) // 2]} p95={lens[int(len(lens) * 0.95)]} max={lens[-1]}')


if __name__ == '__main__':
    main()
