"""
汇总多个评测结果为 Markdown 表格：python scripts/compare_results.py results/sft results/sft_dpo
"""
import json
import os
import sys

ROWS = [('num_acc', '数值准确率'), ('cite_acc', '引用正确率'), ('period_acc', '报告期准确率'), ('unit_acc', '单位正确率'), ('correct_acc', '完全正确率')]


def load_metrics(d):
    with open(os.path.join(d, 'metrics.json'), encoding='utf-8') as f:
        return json.load(f)


def main(dirs):
    ms = [load_metrics(d) for d in dirs]
    for m, d in zip(ms, dirs):
        m['tag'] = os.path.basename(os.path.normpath(d))  # 以目录名作为列名
    n = ms[0]['answerable'].get('num_n')
    lines = [f"| 指标（{n} 条可回答样本） | " + ' | '.join(m['tag'] for m in ms) + ' |', '|---' * (len(ms) + 1) + '|']
    for key, name in ROWS:
        lines.append(f'| {name} | ' + ' | '.join(f"{m['answerable'].get(key, '-')}%" for m in ms) + ' |')
    lines.append('| 拒答准确率（{} 条） | '.format(ms[0]['refusal'].get('refusal_n', 0)) +
                 ' | '.join(f"{m['refusal'].get('refusal_acc', '-')}%" for m in ms) + ' |')
    lines.append('| 数值错误中"取了其他报告期的同名指标" | ' +
                 ' | '.join(f"{m['num_errors_from_distractor']}/{m['num_errors']}" for m in ms) + ' |')
    lines += ['', '| 题型 | ' + ' | '.join(f"{m['tag']} 数值/引用" for m in ms) + ' |', '|---' * (len(ms) + 1) + '|']
    for t in ms[0]['by_type']:
        if t == 'refusal':
            continue
        cells = [f"{m['by_type'][t].get('num_acc', '-')} / {m['by_type'][t].get('cite_acc', '-')}" for m in ms]
        lines.append(f"| {t} ({ms[0]['by_type'][t].get('num_n')}) | " + ' | '.join(cells) + ' |')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1:])
