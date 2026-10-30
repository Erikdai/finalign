"""
合成上市公司公告语料（离线、可复现）

真实公告（巨潮资讯）可用 data_pipeline/crawl_cninfo.py 抓取，但其中的 QA 金标需要人工或教师模型标注。
为了让 CPT→SFT→DPO→评测 全链路可以在无外网、无标注的条件下复现，这里按真实公告的版式生成合成公告：
  * 公司、行业、财务时间序列内部自洽（累计口径、同比、上年同期、每股收益、分红均可互相推出）
  * 同一指标在表格（元/万元）与正文（亿元/万元）中以不同单位出现；年报表格同一行同时出现本年与上年数值
    —— 这正是"数字、单位、报告期易混淆"的来源
  * 注入转载重复、更正公告近重复、页眉页脚、全角数字等噪声，供清洗与去重环节处理
每篇公告附带结构化 facts（不进入训练文本），用于构造带金标的问答、偏好对与评测集。

注意：公司名称、代码与财务数据均为随机生成，与任何真实上市公司无关。

用法：python data_pipeline/synth_announcements.py --num_companies 300 --out data/raw/announcements.jsonl
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import math
import random

from data_pipeline.fin_text import PERIOD_END, PERIOD_REPORT_NAME, UNIT_SCALE, fmt_amount, fmt_number, parse_number, period_str

YEARS = [2021, 2022, 2023, 2024]
PTYPES = ['Q1', 'H1', 'Q3', 'FY']
PUBLISH_MONTH = {'Q1': (0, 4), 'H1': (0, 8), 'Q3': (0, 10), 'FY': (1, 4)}  # (年份偏移, 月份)

NAME_CHARS = list('恒晟嘉瑞华鼎泰弘远拓澜锦衡昊越盛德汇宏中金达宇航星海天安裕丰联博创润祥和信鸿凯腾岳屹睿铭钧峰川禾云启明辰')
INDUSTRIES = {
    '电子': (['电子', '微电', '光电', '芯源'], (0.03, 0.15), (0.06, 0.12)),
    '医药生物': (['医药', '生物', '药业', '康泰'], (0.05, 0.22), (0.05, 0.15)),
    '食品饮料': (['食品', '酒业', '乳业', '饮品'], (0.06, 0.28), (0.005, 0.02)),
    '机械设备': (['机械', '重工', '装备', '精工'], (0.03, 0.12), (0.03, 0.07)),
    '计算机': (['软件', '信息', '数据', '智联'], (-0.05, 0.15), (0.08, 0.2)),
    '电力设备': (['新能', '电气', '储能', '光能'], (0.02, 0.14), (0.03, 0.06)),
    '汽车': (['汽车', '汽配', '动力', '车联'], (0.01, 0.09), (0.03, 0.06)),
    '基础化工': (['化工', '材料', '新材', '化学'], (0.02, 0.13), (0.02, 0.05)),
    '建筑装饰': (['建设', '建工', '工程', '路桥'], (0.01, 0.05), (0.01, 0.03)),
    '家用电器': (['电器', '家电', '厨电', '智家'], (0.04, 0.12), (0.02, 0.05)),
}
AUDITORS = ['立信会计师事务所（特殊普通合伙）', '天健会计师事务所（特殊普通合伙）', '容诚会计师事务所（特殊普通合伙）',
            '大华会计师事务所（特殊普通合伙）', '信永中和会计师事务所（特殊普通合伙）']
DISCLAIMER = '本公司董事会及全体董事保证本公告内容不存在任何虚假记载、误导性陈述或者重大遗漏，并对其内容的真实性、准确性和完整性依法承担法律责任。'
METRICS = [  # key, 表格行标签, 是否区间(期末)指标
    ('revenue', '营业收入'),
    ('np', '归属于上市公司股东的净利润'),
    ('dnp', '归属于上市公司股东的扣除非经常性损益的净利润'),
    ('ocf', '经营活动产生的现金流量净额'),
    ('eps', '基本每股收益'),
    ('roe', '加权平均净资产收益率'),
    ('assets', '总资产'),
]


def pct_change(cur, prev):
    return (cur - prev) / abs(prev) * 100 if prev else 0.0


def signed(x, decimals=2):
    return fmt_number(x, decimals, thousands=False)


class Company:
    def __init__(self, idx, rng, used_names):
        self.rng = rng
        self.industry = rng.choice(list(INDUSTRIES))
        words, margin_rng, rd_rng = INDUSTRIES[self.industry]
        while True:
            short = ''.join(rng.sample(NAME_CHARS, 2)) + rng.choice(words)
            if short not in used_names:
                used_names.add(short)
                break
        self.short = short
        self.full = f'{short}股份有限公司'
        self.code = f'9{idx:05d}'  # 合成代码段，避免与真实 A 股代码冲突
        self.margin_rng, self.rd_rng = margin_rng, rd_rng
        self.series = self._simulate()

    def _simulate(self):
        """生成 2020-2024 年各报告期的累计口径财务数据（单位：元）"""
        rng = self.rng
        rev_fy = math.exp(rng.uniform(math.log(3e8), math.log(2e11)))
        self.shares = round(min(1e10, max(8e7, rev_fy / rng.uniform(3, 15))), -4)  # 总股本与营收规模挂钩，使每股收益落在合理区间
        assets_ratio = rng.uniform(1.0, 3.0)
        equity_ratio = rng.uniform(0.35, 0.65)
        series = {}
        for year in [2020] + YEARS:
            rev_fy *= 1 + max(-0.4, min(0.8, rng.gauss(0.08, 0.18)))
            margin = rng.uniform(*self.margin_rng) + rng.gauss(0, 0.02)
            q1 = rng.uniform(0.18, 0.27)
            h1 = q1 + rng.uniform(0.22, 0.28)
            q3 = h1 + rng.uniform(0.22, 0.28)
            assets_fy = rev_fy * assets_ratio * rng.uniform(0.95, 1.08)
            for ptype, frac in zip(PTYPES, [q1, h1, q3, 1.0]):
                rev = rev_fy * frac * rng.uniform(0.98, 1.02)
                np_ = rev * (margin + rng.gauss(0, 0.01))
                assets = assets_fy * (0.9 + 0.1 * frac) * rng.uniform(0.98, 1.02)
                equity = assets * equity_ratio
                series[(year, ptype)] = {
                    'revenue': round(rev, 2),
                    'np': round(np_, 2),
                    'dnp': round(np_ * rng.uniform(0.75, 1.05), 2),
                    'ocf': round(np_ * rng.uniform(-0.6, 2.2), 2),
                    'eps': round(np_ / self.shares, 2),
                    'roe': round(np_ / equity * 100, 2),
                    'assets': round(assets, 2),
                    'rd': round(rev * rng.uniform(*self.rd_rng), 2),
                }
        return series


class DocBuilder:
    """逐行拼装公告正文，同时记录 facts（anchor 为该事实所在行的原文，用于清洗后定位证据块）"""

    def __init__(self, company, title, doc_type, year, ptype, publish_date):
        self.c = company
        self.meta = {'code': company.code, 'short_name': company.short, 'full_name': company.full, 'industry': company.industry,
                         'title': title, 'doc_type': doc_type, 'year': year, 'ptype': ptype, 'publish_date': publish_date}
        self.lines, self.facts = [], []

    def add(self, line, facts=()):
        self.lines.append(line)
        for f in facts:
            self.facts.append({**f, 'anchor': line})

    def build(self):
        return {**self.meta, 'text': '\n'.join(self.lines), 'facts': self.facts}


def fact(kind, metric, label, year, ptype, unit, display, sign=1, **extra):
    """display 为原文中的数字串（不含单位）；value 由 display 按单位归一化（金额→元、股数→股、比例→百分数），
    因此正文中"12.35亿元"这类四舍五入后的数值与表格中的精确值各自独立判分。
    sign 用于"下降7.83%""减少0.52个百分点"这类以文字表达方向的数值"""
    to_value = lambda s: sign * parse_number(s) * UNIT_SCALE[unit]
    value = [to_value(x) for x in display] if isinstance(display, list) else to_value(display)
    return dict(kind=kind, metric=metric, label=label, year=year, ptype=ptype, value=value, unit=unit, display=display, **extra)


def periodic_report(c, year, ptype, rng, seq):
    cur, prev = c.series[(year, ptype)], c.series[(year - 1, ptype)]
    prev_fy = c.series[(year - 1, 'FY')]
    is_summary = ptype in ('H1', 'FY')
    report = f"{year}年{PERIOD_REPORT_NAME[ptype]}{'摘要' if is_summary else ''}"
    oy, om = PUBLISH_MONTH[ptype]
    d = DocBuilder(c, f'{c.short}：{report}', 'periodic', year, ptype, f'{year + oy}-{om:02d}-{rng.randint(10, 29):02d}')
    unit = rng.choice(['元', '万元']) if ptype != 'FY' else rng.choice(['元', '元', '万元'])
    d.add(f'证券代码：{c.code}　证券简称：{c.short}　公告编号：{year + oy}-{seq:03d}')
    d.add(f'{c.full}{report}')
    d.add(DISCLAIMER)
    d.add('重要内容提示：公司负责人、主管会计工作负责人及会计机构负责人保证财务信息的真实、准确、完整。')
    d.add('一、主要财务数据')
    d.add('（一）主要会计数据和财务指标')
    if ptype == 'FY':
        cur_lbl, prev_lbl = f'{year}年', f'{year - 1}年'
    elif ptype == 'Q3':
        cur_lbl, prev_lbl = '年初至报告期末', '上年同期'
    else:
        cur_lbl, prev_lbl = '本报告期', '上年同期'
    for key, label in METRICS:
        cv, pv = cur[key], prev[key]
        if key == 'assets':  # 期末时点指标：与上年度末比较
            pv = prev_fy[key]
            row_unit, cd, pd_ = unit, fmt_amount(cv, unit), fmt_amount(pv, unit)
            chg = pct_change(cv, pv)
            line = f'{label}（{row_unit}）：本报告期末为{cd}，上年度末为{pd_}，本报告期末比上年度末增减{signed(chg)}%。'
            d.add(line, [fact('value', key, label, year, ptype, row_unit, cd, date=f'{year}年{PERIOD_END[ptype]}'),
                         fact('prev', key, label, year - 1, 'FY', row_unit, pd_, date=f'{year - 1}年12月31日', report_year=year, report_ptype=ptype),
                         fact('yoy', key, label, year, ptype, '%', signed(chg), date=f'{year}年{PERIOD_END[ptype]}', base=(cv, row_unit, cd), vs='上年度末')])
            continue
        if key == 'eps':
            row_unit, cd, pd_ = '元/股', fmt_number(cv), fmt_number(pv)
        elif key == 'roe':
            row_unit, cd, pd_ = '%', fmt_number(cv), fmt_number(pv)
        else:
            row_unit, cd, pd_ = unit, fmt_amount(cv, unit), fmt_amount(pv, unit)
        if key == 'roe':
            diff = cv - pv
            chg_txt = f"{'增加' if diff >= 0 else '减少'}{fmt_number(abs(diff))}个百分点"
            yoy = fact('yoy', key, label, year, ptype, '个百分点', fmt_number(abs(diff)), sign=1 if diff >= 0 else -1, base=(cv, '%', cd))
        else:
            chg = pct_change(cv, pv)
            chg_txt = f'增减{signed(chg)}%'
            yoy = fact('yoy', key, label, year, ptype, '%', signed(chg), base=(cv, row_unit, cd))
        line = f'{label}（{row_unit}）：{cur_lbl}为{cd}，{prev_lbl}为{pd_}，{chg_txt}。'
        d.add(line, [fact('value', key, label, year, ptype, row_unit, cd),
                     fact('prev', key, label, year - 1, ptype, row_unit, pd_, report_year=year, report_ptype=ptype),
                     yoy])
    d.add('注：本报告期指' + ('本年度' if ptype == 'FY' else f'{period_str(year, ptype)}') + '，"上年同期"数据未经重述。')
    # 正文经营情况：与表格使用不同单位（亿元/万元）
    if ptype in ('H1', 'FY') or rng.random() < 0.5:
        nar_unit = '亿元' if cur['revenue'] >= 1e9 else '万元'
        np_unit = '亿元' if abs(cur['np']) >= 1e9 else '万元'
        rev_d, np_d = fmt_amount(cur['revenue'], nar_unit), fmt_amount(cur['np'], np_unit)
        rv, nv = pct_change(cur['revenue'], prev['revenue']), pct_change(cur['np'], prev['np'])
        d.add('二、经营情况讨论与分析')
        line = (f"报告期内，公司实现营业收入{rev_d}{nar_unit}，同比{'增长' if rv >= 0 else '下降'}{fmt_number(abs(rv))}%；"
                f"实现归属于上市公司股东的净利润{np_d}{np_unit}，同比{'增长' if nv >= 0 else '下降'}{fmt_number(abs(nv))}%。")
        d.add(line, [fact('value', 'revenue', '营业收入', year, ptype, nar_unit, rev_d, narrative=True),
                     fact('yoy', 'revenue', '营业收入', year, ptype, '%', fmt_number(abs(rv)), sign=1 if rv >= 0 else -1, base=(cur['revenue'], nar_unit, rev_d), narrative=True),
                     fact('value', 'np', '归属于上市公司股东的净利润', year, ptype, np_unit, np_d, narrative=True),
                     fact('yoy', 'np', '归属于上市公司股东的净利润', year, ptype, '%', fmt_number(abs(nv)), sign=1 if nv >= 0 else -1, base=(cur['np'], np_unit, np_d), narrative=True)])
        if ptype in ('H1', 'FY'):
            rd_unit = '亿元' if cur['rd'] >= 1e9 else '万元'
            rd_d = fmt_amount(cur['rd'], rd_unit)
            ratio = cur['rd'] / cur['revenue'] * 100
            line = f'报告期内，公司持续加大研发投入，研发投入金额为{rd_d}{rd_unit}，占营业收入的比例为{fmt_number(ratio)}%。'
            d.add(line, [fact('value', 'rd', '研发投入', year, ptype, rd_unit, rd_d),
                         fact('value', 'rd_ratio', '研发投入占营业收入的比例', year, ptype, '%', fmt_number(ratio))])
    d.add('三、股东信息')
    d.add(f'报告期末普通股股东总数为{rng.randint(8000, 260000):,}户，前十名股东中不存在回购专户。')
    d.add('四、其他提醒事项')
    d.add('需提醒投资者关注的关于公司报告期经营情况的其他重要信息：无。')
    d.add(f'{c.full}董事会')
    return d.build()


def forecast(c, year, ptype, rng, seq):
    cur, prev = c.series[(year, ptype)], c.series[(year - 1, ptype)]
    if cur['np'] <= 2e6 or prev['np'] <= 2e6:
        return None
    lo = math.floor(cur['np'] * rng.uniform(0.85, 0.97) / 1e6) * 100  # 万元，取整到百万元
    hi = math.ceil(cur['np'] * rng.uniform(1.03, 1.15) / 1e6) * 100
    prev_w = prev['np'] / 1e4
    a, b = pct_change(lo, prev_w), pct_change(hi, prev_w)
    name = '年度' if ptype == 'FY' else '半年度'
    pub = f'{year + 1}-01-{rng.randint(10, 30):02d}' if ptype == 'FY' else f'{year}-07-{rng.randint(5, 30):02d}'
    d = DocBuilder(c, f'{c.short}：{year}年{name}业绩预告', 'forecast', year, ptype, pub)
    d.add(f'证券代码：{c.code}　证券简称：{c.short}　公告编号：{pub[:4]}-{seq:03d}')
    d.add(f'{c.full}{year}年{name}业绩预告')
    d.add(DISCLAIMER)
    d.add('一、本期业绩预告情况')
    end = '12月31日' if ptype == 'FY' else '6月30日'
    d.add(f'（一）业绩预告期间：{year}年1月1日至{year}年{end}。')
    lo_d, hi_d = fmt_number(lo), fmt_number(hi)
    if a >= 0:
        growth = f'同比增长{fmt_number(a)}%至{fmt_number(b)}%'
    elif b <= 0:
        growth = f'同比下降{fmt_number(-b)}%至{fmt_number(-a)}%'
    else:
        growth = f'同比变动{signed(a)}%至{signed(b)}%'
    line = (f'（二）业绩预告情况：经财务部门初步测算，预计{period_str(year, ptype)}实现归属于上市公司股东的净利润{lo_d}万元至{hi_d}万元，'
            f'与上年同期相比，{growth}。')
    d.add(line, [fact('forecast', 'np_forecast', '归属于上市公司股东的净利润', year, ptype, '万元', [lo_d, hi_d], growth=growth)])
    d.add('（三）本次业绩预告未经注册会计师审计。')
    d.add('二、上年同期业绩情况')
    pd_ = fmt_number(prev_w)
    line = f'{period_str(year - 1, ptype)}归属于上市公司股东的净利润：{pd_}万元；基本每股收益：{fmt_number(prev["eps"])}元/股。'
    d.add(line, [fact('prev', 'np', '归属于上市公司股东的净利润', year - 1, ptype, '万元', pd_, report_year=year, report_ptype=ptype)])
    d.add('三、本期业绩变化的主要原因')
    d.add(rng.choice(['报告期内，公司主营业务市场需求稳步增长，产品结构持续优化。', '报告期内，公司持续推进降本增效，规模效应逐步显现。',
                      '报告期内，公司新产能逐步释放，海外市场拓展取得进展。']))
    d.add('四、风险提示：本次业绩预告数据仅为初步核算数据，具体准确的财务数据以公司正式披露的定期报告为准。')
    d.add(f'{c.full}董事会')
    return d.build()


def dividend(c, year, rng, seq):
    fy = c.series[(year, 'FY')]
    if fy['np'] <= 0:
        return None
    per10 = round(fy['np'] * rng.uniform(0.2, 0.6) / c.shares * 10, 2)
    if per10 < 0.1:
        return None
    total = round(per10 * c.shares / 10, 2)
    ratio = total / fy['np'] * 100
    pub = f'{year + 1}-04-{rng.randint(10, 29):02d}'
    d = DocBuilder(c, f'{c.short}：{year}年度利润分配预案公告', 'dividend', year, 'FY', pub)
    d.add(f'证券代码：{c.code}　证券简称：{c.short}　公告编号：{year + 1}-{seq:03d}')
    d.add(f'{c.full}{year}年度利润分配预案公告')
    d.add(DISCLAIMER)
    d.add('重要内容提示：本次利润分配以实施权益分派股权登记日登记的总股本为基数，具体日期将在权益分派实施公告中明确。')
    d.add('一、利润分配预案内容')
    d.add(f'经{rng.choice(AUDITORS)}审计，截至{year}年12月31日，公司期末可供分配利润为{fmt_number(fy["np"] * rng.uniform(1.5, 4))}元。')
    per10_d, total_d, shares_d = fmt_number(per10), fmt_number(total), fmt_number(c.shares, 0)
    line = f'公司拟向全体股东每10股派发现金红利{per10_d}元（含税）。截至本公告披露日，公司总股本{shares_d}股，以此计算合计拟派发现金红利{total_d}元（含税）。'
    d.add(line, [fact('dividend', 'dps10', '每10股派发现金红利', year, 'FY', '元', per10_d),
                 fact('dividend', 'div_total', '合计拟派发现金红利', year, 'FY', '元', total_d),
                 fact('dividend', 'shares', '总股本', year, 'FY', '股', shares_d)])
    line = f'本年度公司现金分红比例为{fmt_number(ratio)}%，本年度不送红股，不以资本公积金转增股本。'
    d.add(line, [fact('dividend', 'payout', '现金分红比例', year, 'FY', '%', fmt_number(ratio))])
    d.add('二、公司履行的决策程序：本次利润分配预案已经公司董事会审议通过，尚需提交公司年度股东大会审议。')
    d.add(f'{c.full}董事会')
    return d.build()


def buybacks(c, year, rng, seq):
    """同一年披露多期回购进展，累计数逐期递增 —— 制造"截至日期"混淆"""
    if rng.random() > 0.3:
        return []
    months = sorted(rng.sample(range(2, 12), rng.randint(2, 3)))
    price = max(3.0, c.series[(year, 'FY')]['eps'] * rng.uniform(12, 30))
    n, amt, docs = 0, 0.0, []
    for i, m in enumerate(months):
        add = round(c.shares * rng.uniform(0.001, 0.004), -2)
        lo_p, hi_p = round(price * rng.uniform(0.85, 0.95), 2), round(price * rng.uniform(1.05, 1.15), 2)
        n += add
        amt += add * rng.uniform(lo_p, hi_p)
        amt = round(amt, 2)
        last_day = [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
        date = f'{year}年{m}月{last_day}日'
        pub = f'{year}-{m + 1:02d}-0{rng.randint(1, 5)}'
        d = DocBuilder(c, f'{c.short}：关于回购公司股份的进展公告', 'buyback', year, None, pub)
        d.add(f'证券代码：{c.code}　证券简称：{c.short}　公告编号：{year}-{seq + i:03d}')
        d.add(f'{c.full}关于回购公司股份的进展公告')
        d.add(DISCLAIMER)
        d.add('一、回购股份的基本情况：公司于前期召开董事会，审议通过了以集中竞价交易方式回购公司股份的方案，回购股份将用于员工持股计划或股权激励。')
        n_d, p_d, amt_d = fmt_number(n, 0), fmt_number(n / c.shares * 100), fmt_number(amt)
        line = (f'二、回购股份的进展情况：截至{date}，公司通过集中竞价交易方式已累计回购公司股份{n_d}股，占公司总股本的比例为{p_d}%，'
                f'回购成交的最高价为{fmt_number(hi_p)}元/股，最低价为{fmt_number(lo_p)}元/股，已支付的总金额为{amt_d}元（不含交易费用）。')
        d.add(line, [fact('buyback', 'bb_shares', '累计回购股份', year, None, '股', n_d, date=date),
                     fact('buyback', 'bb_amount', '已支付的总金额', year, None, '元', amt_d, date=date),
                     fact('buyback', 'bb_ratio', '占公司总股本的比例', year, None, '%', p_d, date=date)])
        d.add('三、其他事项：公司将根据市场情况在回购期限内继续实施回购计划，并及时履行信息披露义务。')
        d.add(f'{c.full}董事会')
        docs.append(d.build())
    return docs


# ---------------- 噪声注入：模拟 PDF 抽取与多源转载 ----------------
FULLWIDTH = str.maketrans('0123456789,.%', '０１２３４５６７８９，．％')


def add_noise(doc, rng):
    lines = doc['text'].split('\n')
    out, n_pages = [], max(1, len(lines) // 6)
    for i, line in enumerate(lines):
        if rng.random() < 0.08:
            line = line.translate(FULLWIDTH)
        if rng.random() < 0.1:
            line = '  ' + line.replace('，', '， ', 1) + ' '
        out.append(line)
        if i % 6 == 5:
            out.append(f'{doc["full_name"]} {doc["title"].split("：")[-1]}  第{i // 6 + 1}页 共{n_pages}页')
            if rng.random() < 0.5:
                out.append('')
    return {**doc, 'text': '\n'.join(out)}


def make_near_dup(doc):
    """更正公告：仅改动个别措辞，事实不变"""
    text = doc['text'].replace('不存在任何虚假记载', '不存在虚假记载').replace('依法承担法律责任', '承担个别及连带责任')
    return {**doc, 'title': doc['title'] + '（更正后）', 'text': text + '\n特此公告。（本公告为更正后版本）',
            'publish_date': doc['publish_date'][:8] + '28'}


def main():
    parser = argparse.ArgumentParser(description="FinAlign 合成公告语料生成")
    parser.add_argument('--num_companies', type=int, default=300, help="公司数量")
    parser.add_argument('--dup_ratio', type=float, default=0.06, help="完全重复（多源转载）比例")
    parser.add_argument('--near_dup_ratio', type=float, default=0.04, help="近重复（更正公告）比例")
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', type=str, default='data/raw/announcements.jsonl')
    args = parser.parse_args()
    rng = random.Random(args.seed)
    used, docs = set(), []
    companies = [Company(i, rng, used) for i in range(args.num_companies)]
    for c in companies:
        seq = 1
        for year in YEARS:
            for ptype in PTYPES:
                docs.append(periodic_report(c, year, ptype, rng, seq))
                seq += 1
                if ptype in ('H1', 'FY') and rng.random() < 0.4:
                    f = forecast(c, year, ptype, rng, seq)
                    if f:
                        docs.append(f)
                        seq += 1
            dv = dividend(c, year, rng, seq) if rng.random() < 0.75 else None
            if dv:
                docs.append(dv)
                seq += 1
            bbs = buybacks(c, year, rng, seq)
            docs.extend(bbs)
            seq += len(bbs)
    out = []
    for d in docs:
        out.append(add_noise(d, rng))
        if rng.random() < args.dup_ratio:
            out.append(add_noise(d, rng))  # 转载：同文不同噪声
        if rng.random() < args.near_dup_ratio:
            out.append(add_noise(make_near_dup(d), rng))
    rng.shuffle(out)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for i, d in enumerate(out):
            f.write(json.dumps({'ann_id': f'SYN{i:07d}', **d}, ensure_ascii=False) + '\n')
    print(f'companies={len(companies)} unique_docs={len(docs)} written={len(out)} -> {args.out}')


if __name__ == '__main__':
    main()
