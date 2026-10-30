"""
金融文本基础工具：数值格式化、单位归一、报告期解析、引用解析。
数据构造（gold 答案、偏好对扰动）与评测（数值/引用/报告期判分）共用同一套规则，保证口径一致。
"""
import re

SYSTEM_PROMPT = "你是金融公告问答助手。请仅依据给定证据作答，数值须注明单位和报告期，并用[编号]标注引用来源。"

# ---------------- 单位 ----------------
CURRENCY_SCALE = {'元': 1, '千元': 1e3, '万元': 1e4, '百万元': 1e6, '亿元': 1e8}
SHARE_SCALE = {'股': 1, '万股': 1e4, '亿股': 1e8}
# 单位类别：同类可换算比较，跨类视为单位错误
UNIT_CLASS = {**dict.fromkeys(CURRENCY_SCALE, 'currency'), **dict.fromkeys(SHARE_SCALE, 'share'),
              '%': 'percent', '个百分点': 'pp', '元/股': 'per_share'}
UNIT_SCALE = {**CURRENCY_SCALE, **SHARE_SCALE, '%': 1, '个百分点': 1, '元/股': 1}
# 长单位在前，防止"元/股"被"元"截断
_UNIT_PATTERN = '|'.join(sorted(map(re.escape, UNIT_CLASS), key=len, reverse=True))
NUM_UNIT_RE = re.compile(rf'(?<![\d.])(-?\d[\d,]*(?:\.\d+)?)\s*({_UNIT_PATTERN})')
CITE_RE = re.compile(r'\[(\d{1,2})\]')

# ---------------- 报告期 ----------------
# ptype: Q1 一季度 / H1 半年度 / Q3 前三季度(年初至报告期末) / FY 年度
PERIOD_REPORT_NAME = {'Q1': '第一季度报告', 'H1': '半年度报告', 'Q3': '第三季度报告', 'FY': '年度报告'}
PERIOD_TEXT = {'Q1': '第一季度', 'H1': '上半年', 'Q3': '前三季度', 'FY': '年度'}
PERIOD_END = {'Q1': '3月31日', 'H1': '6月30日', 'Q3': '9月30日', 'FY': '12月31日'}
_PERIOD_ALIASES = {
    'Q1': ['第一季度', '一季度', '1-3月', '一季报'],
    'H1': ['上半年', '半年度', '1-6月', '中报', '半年报'],
    'Q3': ['前三季度', '1-9月', '第三季度', '三季报', '三季度'],
    'FY': ['年度', '全年', '年报', '1-12月'],
}
_ALIAS2P = {a: p for p, aliases in _PERIOD_ALIASES.items() for a in aliases}
PERIOD_RE = re.compile(r'(20\d{2})\s*年\s*(' + '|'.join(sorted(map(re.escape, _ALIAS2P), key=len, reverse=True)) + ')?')


def period_str(year, ptype):
    """回答中使用的规范报告期表述，如 2024年第一季度 / 2024年度"""
    return f"{year}年度" if ptype == 'FY' else f"{year}年{PERIOD_TEXT[ptype]}"


_DATE_TAIL_RE = re.compile(r'(\d{1,2})\s*月\s*(\d{1,2})\s*日')
_MONTH2P = {3: 'Q1', 6: 'H1', 9: 'Q3', 12: 'FY'}


def parse_periods(text):
    """抽取文本中出现的全部报告期，返回 [(year, ptype)]；仅写"2024年"视为年度，
    "2024年3月31日"这类期末日期映射到对应报告期，其余日期返回 ptype=None"""
    out = []
    for m in PERIOD_RE.finditer(text):
        if m.group(2) is None:
            d = _DATE_TAIL_RE.match(text, m.end())
            if d:
                month, day = int(d.group(1)), int(d.group(2))
                is_period_end = month in _MONTH2P and day == (31 if month in (3, 12) else 30)
                out.append((int(m.group(1)), _MONTH2P[month] if is_period_end else None))
                continue
        out.append((int(m.group(1)), _ALIAS2P.get(m.group(2), 'FY')))
    return out


# ---------------- 数值 ----------------
def fmt_number(value, decimals=2, thousands=True):
    s = f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"
    return s


def fmt_amount(value_yuan, unit, decimals=2):
    """把以"元"计的金额换算到指定单位并格式化：1.23亿元、12,345.67万元、1,234,567,890.12元"""
    v = value_yuan / CURRENCY_SCALE[unit]
    return fmt_number(v, decimals, thousands=(unit != '亿元'))


def parse_number(s):
    return float(s.replace(',', ''))


_NEG_PREFIX = ('下降', '减少', '下滑', '降低')


def extract_values(text):
    """抽取 (原始字符串, 归一化数值, 单位, 单位类别, 起始位置)；金额统一换算成元、股数换算成股；
    "同比下降7.83%""减少0.52个百分点"按负值处理，与金标的带符号数值对齐"""
    out = []
    for m in NUM_UNIT_RE.finditer(text):
        num, unit = m.group(1), m.group(2)
        try:
            v = parse_number(num) * UNIT_SCALE[unit]
        except ValueError:
            continue
        if text[max(0, m.start() - 2):m.start()] in _NEG_PREFIX and v > 0:
            v = -v
        out.append({'raw': m.group(0), 'value': v, 'unit': unit, 'cls': UNIT_CLASS[unit], 'pos': m.start()})
    return out


def values_equal(a, b, rel_tol=1e-6):
    return abs(a - b) <= rel_tol * max(1.0, abs(a), abs(b))


def extract_citations(text):
    return [int(x) for x in CITE_RE.findall(text)]


def is_refusal(text):
    return any(k in text for k in ('无法回答', '无法确定', '未包含', '未提供', '未披露'))
