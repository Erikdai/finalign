"""
问答构造核心：把结构化事实组织成"问题 + 检索证据 + 带引用答案"的多轮会话

每个 turn 记录可判分的 target：
  values   可接受的归一化数值（同一事实在表格/正文/次年报表中可能以不同单位、精度出现）
  cls      单位类别（currency / percent / pp / per_share / share）
  period   金标报告期 (year, ptype)，回购等按日期披露的事实为 None
  support  证据中可作为合法引用的编号集合
另外记录 slots（答案中目标数字串、单位、报告期表述、引用编号），供 DPO 构造 rejected 时做定点扰动。
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from collections import defaultdict

from data_pipeline.fin_text import PERIOD_END, SYSTEM_PROMPT, UNIT_CLASS, period_str
from rag.retriever import ChunkRetriever, format_evidence

LABEL_ALIASES = {
    'revenue': ['营业收入', '营收'],
    'np': ['归属于上市公司股东的净利润', '归母净利润', '净利润'],
    'dnp': ['扣除非经常性损益的净利润', '扣非归母净利润', '扣非净利润'],
    'ocf': ['经营活动产生的现金流量净额', '经营性现金流净额', '经营现金流'],
    'eps': ['基本每股收益', '每股收益'],
    'roe': ['加权平均净资产收益率', '净资产收益率', 'ROE'],
    'assets': ['总资产', '资产总额'],
    'rd': ['研发投入', '研发投入金额'],
    'rd_ratio': ['研发投入占营业收入的比例', '研发投入占营收比例'],
}
CANON_LABEL = {'dnp': '归属于上市公司股东的扣除非经常性损益的净利润', 'rd': '研发投入金额'}
PERIOD_Q_ALIASES = {'Q1': ['第一季度', '一季度', '1-3月'], 'H1': ['上半年', '半年度', '1-6月'],
                    'Q3': ['前三季度', '1-9月'], 'FY': ['年度', '全年']}
TABLE_METRICS = ('revenue', 'np', 'dnp', 'ocf', 'eps', 'roe', 'assets')


def q_period(year, ptype, rng):
    return f'{year}年{rng.choice(PERIOD_Q_ALIASES[ptype])}'


def label_of(metric, f):
    return CANON_LABEL.get(metric, f['label'])


def yoy_phrase(f):
    """把 yoy 事实渲染成"同比增长6.24%"/"较上年同期减少0.52个百分点"/"较上年度末下降2.16%"""
    v = f['value']
    if f['unit'] == '个百分点':
        return f"较上年同期{'增加' if v >= 0 else '减少'}{f['display'].lstrip('-')}个百分点"
    head = '较上年度末' if f.get('vs') == '上年度末' else '同比'
    return f"{head}{'增长' if v >= 0 else '下降'}{f['display'].lstrip('-')}%"


class FactStore:
    """keep(rec) 同时作用于证据块与事实：训练集只能看到 train 公司的 hist 公告，杜绝检索干扰项泄漏评测数据"""

    def __init__(self, data_dir='data/processed', keep=lambda rec: True):
        self.chunks = {}
        with open(os.path.join(data_dir, 'chunks.jsonl'), encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                if keep(c):
                    self.chunks[c['chunk_id']] = c
        self.facts = []
        with open(os.path.join(data_dir, 'facts.jsonl'), encoding='utf-8') as f:
            for line in f:
                fct = json.loads(line)
                if keep(fct):
                    self.facts.append(fct)
        self.by_key = defaultdict(list)    # (code, key) -> facts
        self.by_chunk = defaultdict(list)  # chunk_id -> facts
        self.by_ann = defaultdict(list)    # ann_id -> facts
        for f in self.facts:
            self.by_key[(f['code'], f['key'])].append(f)
            self.by_chunk[f['chunk_id']].append(f)
            self.by_ann[f['ann_id']].append(f)
        self.retriever = ChunkRetriever(list(self.chunks.values()))

    def support(self, code, key):
        return self.by_key.get((code, key), [])

    def in_chunk(self, chunk_id, key):
        return next((f for f in self.by_chunk[chunk_id] if f['key'] == key), None)


class Turn:
    def __init__(self, question, key, gold_chunk, answer_fn, target, slots):
        self.question, self.key, self.gold_chunk = question, key, gold_chunk
        self.answer_fn, self.target, self.slots = answer_fn, target, slots


class QABuilder:
    def __init__(self, store, rng, k_evidence=3):
        self.s, self.rng, self.k = store, rng, k_evidence

    # ---------------- target ----------------
    def _target(self, f, period):
        sup = self.s.support(f['code'], f['key'])
        values = [f['value']] if f['kind'] == 'forecast' else sorted({round(x['value'], 6) for x in sup})
        return {'kind': f['kind'], 'short': f['short_name'], 'metric': f['metric'], 'values': values, 'cls': UNIT_CLASS[f['unit']],
                'period': period, 'support_chunks': sorted({x['chunk_id'] for x in sup}),
                'displays': sorted({(x['display'] if isinstance(x['display'], str) else '|'.join(x['display']), x['unit']) for x in sup})}

    # ---------------- 各类问题 ----------------
    def value_turn(self, f, question=None, with_yoy=None):
        """f: kind in (value, prev) 的事实；答案中报告期一律用 f 自身的 (year, ptype)"""
        rng, s = self.rng, f['short_name']
        label, alias = label_of(f['metric'], f), rng.choice(LABEL_ALIASES[f['metric']])
        year, ptype = f['year'], f['ptype']
        yoy = self.s.in_chunk(f['chunk_id'], f"{f['metric']}|{year}|{ptype}|yoy")
        with_yoy = (rng.random() < 0.5) if with_yoy is None else with_yoy
        yoy_txt = f'，{yoy_phrase(yoy)}' if (with_yoy and yoy) else ''
        unit = f['unit']
        if f['metric'] == 'assets':
            ptxt = f"{year}年{PERIOD_END[ptype]}"
            question = question or rng.choice([f'截至{ptxt}，{s}的{alias}是多少？', f'{s}{ptxt}的{alias}为多少？', f'{s}截至{ptxt}的{alias}有多少？'])
            answer = lambda k: f"截至{ptxt}，{s}{label}为{f['display']}{unit}{yoy_txt}[{k}]。"
        else:
            ptxt = period_str(year, ptype)
            qp = q_period(year, ptype, rng)
            question = question or rng.choice([f'{s}{qp}的{alias}是多少？', f'{s}{qp}{alias}为多少？', f'请问{s}{qp}实现{alias}多少？',
                                               f'根据公告，{s}{qp}的{alias}是多少？', f'{s}在{qp}的{alias}有多少？'])
            answer = lambda k: f"{s}{ptxt}{label}为{f['display']}{unit}{yoy_txt}[{k}]。"
        return Turn(question, f['key'], f['chunk_id'], answer, self._target(f, (year, ptype)),
                    {'disp': f['display'], 'unit': unit, 'period_text': ptxt, 'short': f['short_name'], 'metric': f['metric'], 'year': year, 'ptype': ptype})

    def yoy_turn(self, f, question=None):
        rng, s = self.rng, f['short_name']
        label, alias = label_of(f['metric'], f), rng.choice(LABEL_ALIASES[f['metric']])
        year, ptype = f['year'], f['ptype']
        _, base_unit, base_disp = f['base']
        if f['metric'] == 'assets':
            ptxt = f"{year}年{PERIOD_END[ptype]}"
            question = question or rng.choice([f'截至{ptxt}，{s}{alias}较上年度末变化多少？', f'{s}{ptxt}的{alias}比上年度末增长了多少？'])
            answer = lambda k: f'截至{ptxt}，{s}{label}为{base_disp}{base_unit}，{yoy_phrase(f)}[{k}]。'
        else:
            ptxt, qp = period_str(year, ptype), q_period(year, ptype, rng)
            if f['unit'] == '个百分点':
                cands = [f'{s}{qp}{alias}较上年同期变化了多少个百分点？', f'{s}{qp}的{alias}比上年同期增减多少？']
            else:
                cands = [f'{s}{qp}{alias}同比变化多少？', f'{s}{qp}的{alias}同比增长了多少？', f'{s}{qp}{alias}较上年同期变动幅度是多少？']
            question = question or rng.choice(cands)
            answer = lambda k: f'{s}{ptxt}{label}为{base_disp}{base_unit}，{yoy_phrase(f)}[{k}]。'
        return Turn(question, f['key'], f['chunk_id'], answer, self._target(f, (year, ptype)),
                    {'disp': f['display'].lstrip('-'), 'unit': f['unit'], 'period_text': ptxt, 'short': f['short_name'], 'metric': f['metric'], 'year': year, 'ptype': ptype})

    def forecast_turn(self, f):
        rng, s = self.rng, f['short_name']
        year, ptype = f['year'], f['ptype']
        ptxt, qp = period_str(year, ptype), q_period(year, ptype, rng)
        lo, hi = f['display']
        question = rng.choice([f'{s}预计{qp}归属于上市公司股东的净利润是多少？', f'{s}{qp}业绩预告中预计的归母净利润区间是多少？',
                               f'{s}对{qp}的净利润预告是多少？'])
        answer = lambda k: f"根据业绩预告，{s}预计{ptxt}实现归属于上市公司股东的净利润{lo}万元至{hi}万元，{f['growth']}[{k}]。"
        return Turn(question, f['key'], f['chunk_id'], answer, self._target(f, (year, ptype)),
                    {'disp': lo, 'unit': '万元', 'period_text': ptxt, 'short': f['short_name'], 'metric': 'np_forecast', 'year': year, 'ptype': ptype})

    def dividend_turn(self, f):
        rng, s, y, d = self.rng, f['short_name'], f['year'], f['display']
        ptxt = f'{y}年度'
        if f['metric'] == 'dps10':
            question = rng.choice([f'{s}{y}年度利润分配预案中每10股派发多少现金红利？', f'{s}{y}年度的分红方案是每10股派多少元？'])
            answer = lambda k: f'{s}{ptxt}利润分配预案为每10股派发现金红利{d}元（含税）[{k}]。'
        elif f['metric'] == 'div_total':
            question = rng.choice([f'{s}{y}年度合计拟派发多少现金红利？', f'{s}{y}年度分红总额是多少？'])
            answer = lambda k: f'{s}{ptxt}合计拟派发现金红利{d}元（含税）[{k}]。'
        else:  # payout
            question = rng.choice([f'{s}{y}年度的现金分红比例是多少？', f'{s}{y}年度现金分红占净利润的比例为多少？'])
            answer = lambda k: f'{s}{ptxt}现金分红比例为{d}%[{k}]。'
        return Turn(question, f['key'], f['chunk_id'], answer, self._target(f, (y, 'FY')),
                    {'disp': d, 'unit': f['unit'], 'period_text': ptxt, 'short': f['short_name'], 'metric': f['metric'], 'year': y, 'ptype': 'FY'})

    def buyback_turn(self, f):
        rng, s, date, d = self.rng, f['short_name'], f['date'], f['display']
        if f['metric'] == 'bb_shares':
            ratio = self.s.in_chunk(f['chunk_id'], f'bb_ratio|{date}|bb')
            question = rng.choice([f'截至{date}，{s}累计回购了多少股？', f'{s}截至{date}已回购公司股份多少股？'])
            answer = lambda k: f"截至{date}，{s}已累计回购公司股份{d}股，占公司总股本的比例为{ratio['display']}%[{k}]。"
        elif f['metric'] == 'bb_amount':
            question = rng.choice([f'截至{date}，{s}回购股份已支付的总金额是多少？', f'{s}截至{date}回购花了多少钱？'])
            answer = lambda k: f'截至{date}，{s}回购股份已支付的总金额为{d}元（不含交易费用）[{k}]。'
        else:
            question = f'截至{date}，{s}已回购股份占总股本的比例是多少？'
            answer = lambda k: f'截至{date}，{s}已回购股份占公司总股本的比例为{d}%[{k}]。'
        return Turn(question, f['key'], f['chunk_id'], answer, self._target(f, None),
                    {'disp': d, 'unit': f['unit'], 'period_text': date, 'short': f['short_name'], 'metric': f['metric'], 'year': f['year'], 'ptype': None})

    def refusal_turn(self, f):
        """证据中刻意不放任何支持块，答案为拒答"""
        t = self.value_turn(f)
        s, what, label = f['short_name'], t.slots['period_text'], label_of(f['metric'], f)
        t.answer_fn = lambda k: f'根据所提供的证据，未找到{s}{what}{label}的相关数据，无法回答该问题。'
        t.gold_chunk = None
        t.target = {**t.target, 'kind': 'refusal'}
        return t

    # ---------------- 多轮追问 ----------------
    def follow_up(self, prev_fact, asked_metrics=()):
        """基于上一轮（表格行）事实生成追问：同比 / 上年同期 / 同报告期另一指标（不重复已问过的指标）"""
        rng = self.rng
        chunk_id, m, y, p = prev_fact['chunk_id'], prev_fact['metric'], prev_fact['year'], prev_fact['ptype']
        options = []
        yoy = self.s.in_chunk(chunk_id, f'{m}|{y}|{p}|yoy')
        if yoy:
            options.append('yoy')
        prev_key = f"{m}|{y - 1}|{'FY' if m == 'assets' else p}|val"
        prev = self.s.in_chunk(chunk_id, prev_key)
        if prev:
            options.append('prev')
        others = [f for f in self.s.by_ann[prev_fact['ann_id']] if f['kind'] == 'value' and f['metric'] in TABLE_METRICS
                  and f['metric'] != m and f['metric'] not in asked_metrics and not f.get('narrative')]
        if others:
            options.append('other')
        if not options:
            return None, None
        choice = rng.choice(options)
        if choice == 'yoy':
            q = rng.choice(['较上年度末变化多少？', '比上年度末增长了多少？']) if m == 'assets' else \
                rng.choice(['较上年同期变化了多少个百分点？', '比上年同期增减多少？']) if m == 'roe' else \
                rng.choice(['同比变化呢？', '同比增长了多少？', '较上年同期变化多少？'])
            return self.yoy_turn(yoy, question=q), prev_fact
        if choice == 'prev':
            q = rng.choice(['上年度末呢？', '那上年度末是多少？']) if m == 'assets' else rng.choice(['上年同期是多少？', '那上年同期呢？'])
            return self.value_turn(prev, question=q, with_yoy=False), prev_fact
        nf = rng.choice(others)
        alias = rng.choice(LABEL_ALIASES[nf['metric']])
        q = rng.choice([f'那{alias}呢？', f'{alias}是多少？', f'同期的{alias}呢？'])
        return self.value_turn(nf, question=q), nf

    # ---------------- 证据拼装 ----------------
    def assemble(self, turns, code, query, n_evidence=None):
        """gold 块 + 同公司 BM25 难负例（排除所有可支持任一 turn 的块，保证引用唯一可判）"""
        k = n_evidence or self.k
        gold = []
        for t in turns:
            if t.gold_chunk and t.gold_chunk not in gold:
                gold.append(t.gold_chunk)
        exclude = set(gold)
        for t in turns:
            exclude.update(t.target['support_chunks'])
        # 难负例：同公司、含数值事实的块（多为同一指标的其他报告期/其他口径）；少量样本放开限制，保留页眉等普通噪声块
        where = (lambda c: c['chunk_id'] in self.s.by_chunk) if self.rng.random() < 0.85 else None
        distract = [c for c, _ in self.s.retriever.search(query, code=code, topk=max(0, k - len(gold)), exclude=exclude, where=where)]
        evidence = [self.s.chunks[c] for c in gold] + distract
        self.rng.shuffle(evidence)
        pos = {c['chunk_id']: i + 1 for i, c in enumerate(evidence)}
        conv = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        metas = []
        for i, t in enumerate(turns):
            user = f'参考证据：\n{format_evidence(evidence)}\n\n问题：{t.question}' if i == 0 else t.question
            cite = pos.get(t.gold_chunk)
            conv.append({'role': 'user', 'content': user})
            conv.append({'role': 'assistant', 'content': t.answer_fn(cite)})
            support = sorted(pos[c] for c in t.target['support_chunks'] if c in pos)
            metas.append({**t.target, 'support': support, 'cite': cite, 'slots': t.slots})
        return {'conversations': conv, 'turns': metas, 'evidence_ids': [c['chunk_id'] for c in evidence]}
