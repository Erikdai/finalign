"""
金融问答判分（评测与偏好对校验共用）

  数值正确 num_ok    回答中第一个"同单位类别"的数值，归一化后等于任一可接受金标值（单位换算后比较，
                     因此 1.23万元 与 1.23亿元 判为不同，12,345.67万元 与 123,456,700元 判为相同）
                     业绩预告区间要求前两个金额依次等于下限、上限
  单位正确 unit_ok   回答中与金标数字串相同的数值，其单位是否与证据一致（诊断项，无匹配数字串时为 None）
  报告期正确 period_ok 回答中第一个报告期表述是否等于金标报告期（按日期披露的事实为 None，不计入）
  引用正确 cite_ok   引用编号非空，且全部落在能支持该事实的证据块内
  拒答 refusal_ok    证据不含答案时是否拒答
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.fin_text import extract_citations, extract_values, is_refusal, parse_periods, values_equal


def _numeral(s):
    return s.replace(',', '').lstrip('-')


def score_answer(pred, target):
    if target['kind'] == 'refusal':
        return {'refusal_ok': is_refusal(pred) and not extract_values(pred)}
    mentions = extract_values(pred)
    same_cls = [m for m in mentions if m['cls'] == target['cls']]
    if target['kind'] == 'forecast':
        lo, hi = target['values'][0]
        num_ok = len(same_cls) >= 2 and values_equal(same_cls[0]['value'], lo) and values_equal(same_cls[1]['value'], hi)
    else:
        num_ok = bool(same_cls) and any(values_equal(same_cls[0]['value'], g) for g in target['values'])

    gold_units = {}
    for disp, unit in target['displays']:
        for d in disp.split('|'):
            gold_units.setdefault(_numeral(d), set()).add(unit)
    unit_ok = None
    for m in mentions:
        num = _numeral(m['raw'][:len(m['raw']) - len(m['unit'])].strip())
        if num in gold_units:
            unit_ok = m['unit'] in gold_units[num]
            break

    period_ok = None
    if target.get('period'):
        periods = parse_periods(pred)
        period_ok = bool(periods) and tuple(periods[0]) == tuple(target['period'])

    cites = set(extract_citations(pred))
    cite_ok = bool(cites) and cites <= set(target['support'])
    correct = num_ok and cite_ok and period_ok is not False
    return {'num_ok': num_ok, 'unit_ok': unit_ok, 'period_ok': period_ok, 'cite_ok': cite_ok, 'correct': correct}


def aggregate(scores):
    """对一组 score_answer 结果求各项准确率（None 不计入分母）"""
    out = {}
    for key in ('num_ok', 'unit_ok', 'period_ok', 'cite_ok', 'correct', 'refusal_ok'):
        vals = [s[key] for s in scores if s.get(key) is not None]
        if vals:
            name = key[:-3] if key.endswith('_ok') else key
            out[f'{name}_acc'] = round(100 * sum(vals) / len(vals), 2)
            out[f'{name}_n'] = len(vals)
    return out
