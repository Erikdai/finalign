"""
巨潮资讯（cninfo）公告抓取 → PDF 文本抽取 → 关键财务表线性化 + 事实抽取

输出与 synth_announcements.py 相同的 raw 格式（每行一篇公告，含 text 与 facts），可直接接入
clean_dedup.py → build_chunks.py → build_cpt.py / build_qa.py。
真实公告版式差异大，事实抽取只覆盖"主要会计数据和财务指标"表中的常见行（营业收入、归母净利润、扣非净利润、
经营现金流、基本每股收益、加权ROE、总资产），抽取不到的公告仍可用于 CPT 语料。

用法（请控制抓取频率，仅用于研究）：
  python data_pipeline/crawl_cninfo.py --codes 000001,600519 --categories ndbg,yjdbg --start 2022-01-01 --end 2024-12-31 \
      --out data/raw/cninfo.jsonl --pdf_dir data/raw/pdf --max_pages 12
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import re
import time

import requests

from data_pipeline.fin_text import UNIT_SCALE
from data_pipeline.synth_announcements import fact

QUERY_URL = 'http://www.cninfo.com.cn/new/hisAnnouncement/query'
STOCK_LIST_URL = 'http://www.cninfo.com.cn/new/data/szse_stock.json'
PDF_BASE = 'http://static.cninfo.com.cn/'
HEADERS = {'User-Agent': 'Mozilla/5.0 (research crawler; FinAlign)', 'Accept': 'application/json, text/plain, */*'}
CATEGORIES = {  # 简称 -> (cninfo 类目, doc_type, ptype)
    'ndbg': ('category_ndbg_szsh', 'periodic', 'FY'),
    'bndbg': ('category_bndbg_szsh', 'periodic', 'H1'),
    'yjdbg': ('category_yjdbg_szsh', 'periodic', 'Q1'),
    'sjdbg': ('category_sjdbg_szsh', 'periodic', 'Q3'),
    'yjyg': ('category_yjygjxz_szsh', 'forecast', None),
    'qyfp': ('category_qyfpxzcs_szsh', 'dividend', None),
}
KEY_ROWS = [  # 表格行标签正则 -> (metric, 规范标签)
    (r'营业(总)?收入', 'revenue', '营业收入'),
    (r'归属于上市公司(股东|普通股股东)的净利润', 'np', '归属于上市公司股东的净利润'),
    (r'归属于上市公司股东的扣除非经常性损益的净利润', 'dnp', '归属于上市公司股东的扣除非经常性损益的净利润'),
    (r'经营活动产生的现金流量净额', 'ocf', '经营活动产生的现金流量净额'),
    (r'基本每股收益', 'eps', '基本每股收益'),
    (r'加权平均净资产收益率', 'roe', '加权平均净资产收益率'),
    (r'(资产总额|总资产)', 'assets', '总资产'),
]
_NUM = r'(-?\d[\d,]*\.?\d*)'


def load_org_ids(session):
    data = session.get(STOCK_LIST_URL, headers=HEADERS, timeout=30).json()
    return {s['code']: s['orgId'] for s in data['stockList']}


def query_announcements(session, code, org_id, category, start, end, page_size=30, sleep=1.0):
    column = 'sse' if code.startswith('6') else 'szse'
    page, out = 1, []
    while True:
        form = {'pageNum': page, 'pageSize': page_size, 'column': column, 'tabName': 'fulltext', 'plate': '',
                'stock': f'{code},{org_id}', 'searchkey': '', 'secid': '', 'category': f'{category};', 'trade': '',
                'seDate': f'{start}~{end}', 'sortName': '', 'sortType': '', 'isHLtitle': 'true'}
        r = session.post(QUERY_URL, data=form, headers=HEADERS, timeout=30).json()
        out.extend(r.get('announcements') or [])
        if not r.get('hasMore'):
            return out
        page += 1
        time.sleep(sleep)


def pdf_text(path, max_pages):
    import pdfplumber
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:max_pages]:
            lines.append(page.extract_text() or '')
    return '\n'.join(lines)


def parse_title(title):
    m = re.search(r'(20\d{2})\s*年', title)
    year = int(m.group(1)) if m else None
    if '第一季度' in title or '一季度' in title:
        ptype = 'Q1'
    elif '半年' in title:
        ptype = 'H1'
    elif '第三季度' in title or '三季度' in title:
        ptype = 'Q3'
    elif '年度报告' in title or '年报' in title:
        ptype = 'FY'
    else:
        ptype = None
    return year, ptype


_UNIT_DECL_RE = re.compile(r'单位\s*[:：]\s*(百万元|千元|万元|元)')
_PP_RE = re.compile(r'(增加|减少)\s*(\d[\d.]*)\s*个百分点')
# 各类定期报告"主要会计数据"表的列布局（沪深两市模板基本一致）
LAYOUTS = {'Q1': ('cur', 'yoy'), 'Q3': ('q', 'q_yoy', 'cur', 'yoy'), 'H1': ('cur', 'prev', 'yoy'), 'FY': ('cur', 'prev', 'yoy')}


def merge_wrapped_rows(lines):
    """PDF 抽取会把长行标签折成"标签前半 / 数字 / 标签后半"三行，这里拼回一行"""
    out, i = [], 0
    while i < len(lines):
        a = lines[i]
        if i + 2 < len(lines) and not re.search(r'\d', a) and re.match(r'-?\d', lines[i + 1]) and not re.search(r'\d', lines[i + 2]) \
                and len(lines[i + 2]) <= 12:
            out.append(f'{a}{lines[i + 2]} {lines[i + 1]}')
            i += 3
        else:
            out.append(a)
            i += 1
    return out


def linearize_key_table(text, year, ptype):
    """把"营业收入 45,775,517,043.29 18.11"这类表格行改写为与合成数据一致的行格式，并抽取事实"""
    if year is None or ptype is None:
        return text, []
    new_lines, facts, unit_decl, done = [], [], '元', set()
    cur_lbl, prev_lbl = (f'{year}年', f'{year - 1}年') if ptype == 'FY' else ('本报告期', '上年同期')
    for line in merge_wrapped_rows([x.strip() for x in text.split('\n')]):
        m = _UNIT_DECL_RE.search(line)
        if m:
            unit_decl = m.group(1)
        hit = next(((mt, lbl) for pat, mt, lbl in KEY_ROWS if re.match(pat, line)), None)
        if not hit or hit[0] in done:  # 每个指标只取第一次出现（主要会计数据表）
            new_lines.append(line)
            continue
        metric, label = hit
        body = line[len(re.match(r'[^\d-]*', line).group(0)):]
        pp = _PP_RE.search(body)
        nums = re.findall(_NUM, _PP_RE.sub('', body))
        layout = ('cur', 'prev', 'yoy') if metric == 'assets' else LAYOUTS[ptype]
        if len(nums) < (1 if pp else 2):
            new_lines.append(line)
            continue
        col = dict(zip(layout, nums))
        if 'cur' not in col:
            new_lines.append(line)
            continue
        unit = '元/股' if metric == 'eps' else '%' if metric == 'roe' else unit_decl
        if unit not in UNIT_SCALE:
            unit = '元'
        cur, prev = col['cur'], col.get('prev')
        if metric == 'assets':
            parts = [f'本报告期末为{cur}', f'上年度末为{prev}' if prev else None]
        else:
            parts = [f'{"年初至报告期末" if ptype == "Q3" else cur_lbl}为{cur}', f'{prev_lbl}为{prev}' if prev else None]
        chg_txt, yoy_fact = None, None
        if metric == 'roe' and pp:
            chg_txt = f'{pp.group(1)}{pp.group(2)}个百分点'
            yoy_fact = fact('yoy', metric, label, year, ptype, '个百分点', pp.group(2), sign=1 if pp.group(1) == '增加' else -1, base=(None, unit, cur))
        elif col.get('yoy') and metric != 'roe':
            chg_txt = f"增减{col['yoy']}%"
            extra = {'vs': '上年度末'} if metric == 'assets' else {}
            yoy_fact = fact('yoy', metric, label, year, ptype, '%', col['yoy'], base=(None, unit, cur), **extra)
        row = f'{label}（{unit}）：' + '，'.join(p for p in parts + [chg_txt] if p) + '。'
        new_lines.append(row)
        done.add(metric)
        facts.append({**fact('value', metric, label, year, ptype, unit, cur), 'anchor': row})
        if prev:
            prev_period = (year - 1, 'FY') if metric == 'assets' else (year - 1, ptype)
            facts.append({**fact('prev', metric, label, *prev_period, unit, prev), 'anchor': row})
        if yoy_fact:
            facts.append({**yoy_fact, 'anchor': row})
    return '\n'.join(new_lines), facts


def main():
    parser = argparse.ArgumentParser(description="巨潮资讯公告抓取")
    parser.add_argument('--codes', type=str, required=True, help="股票代码，逗号分隔")
    parser.add_argument('--categories', type=str, default='ndbg,bndbg,yjdbg,sjdbg', help=f"公告类别：{','.join(CATEGORIES)}")
    parser.add_argument('--start', type=str, default='2022-01-01')
    parser.add_argument('--end', type=str, default='2024-12-31')
    parser.add_argument('--out', type=str, default='data/raw/cninfo.jsonl')
    parser.add_argument('--pdf_dir', type=str, default='data/raw/pdf')
    parser.add_argument('--max_pages', type=int, default=12, help="每个 PDF 最多解析的页数（主要财务指标通常在前几页）")
    parser.add_argument('--skip_summary', type=int, default=0, help="年报/半年报只保留摘要（1）或全文（0）")
    parser.add_argument('--sleep', type=float, default=1.5, help="请求间隔（秒）")
    args = parser.parse_args()

    os.makedirs(args.pdf_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    session = requests.Session()
    org_ids = load_org_ids(session)
    n_docs, n_facts = 0, 0
    with open(args.out, 'a', encoding='utf-8') as fout:
        for code in args.codes.split(','):
            if code not in org_ids:
                print(f'[skip] 未找到 {code} 的 orgId')
                continue
            for cat in args.categories.split(','):
                category, doc_type, _ = CATEGORIES[cat]
                for ann in query_announcements(session, code, org_ids[code], category, args.start, args.end, sleep=args.sleep):
                    title = re.sub(r'<[^>]+>', '', ann['announcementTitle'])
                    if any(k in title for k in ('英文', '取消', '已取消')) or (args.skip_summary and '摘要' not in title and doc_type == 'periodic'):
                        continue
                    pdf_path = os.path.join(args.pdf_dir, f"{ann['announcementId']}.pdf")
                    if not os.path.exists(pdf_path):
                        resp = session.get(PDF_BASE + ann['adjunctUrl'], headers=HEADERS, timeout=60)
                        with open(pdf_path, 'wb') as f:
                            f.write(resp.content)
                        time.sleep(args.sleep)
                    try:
                        text = pdf_text(pdf_path, args.max_pages)
                    except Exception as e:  # 加密/损坏的 PDF
                        print(f'[skip] {title}: {e}')
                        continue
                    year, ptype = parse_title(title)
                    facts = []
                    if doc_type == 'periodic':
                        text, facts = linearize_key_table(text, year, ptype)
                    doc = {'ann_id': f"CNINFO{ann['announcementId']}", 'code': code, 'short_name': ann['secName'], 'full_name': ann['secName'],
                           'industry': '', 'title': f"{ann['secName']}：{title}", 'doc_type': doc_type, 'year': year, 'ptype': ptype,
                           'publish_date': time.strftime('%Y-%m-%d', time.localtime(ann['announcementTime'] / 1000)),
                           'text': text, 'facts': facts}
                    fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    n_docs, n_facts = n_docs + 1, n_facts + len(facts)
                    print(f'[ok] {doc["title"]} chars={len(text)} facts={len(facts)}')
    print(f'docs={n_docs} facts={n_facts} -> {args.out}')


if __name__ == '__main__':
    main()
