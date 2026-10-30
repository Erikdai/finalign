"""
单元测试（不依赖训练好的权重）：python -m pytest tests -q  或  python tests/test_pipeline.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy
import random
import re

import torch

from data_pipeline.build_dpo import permute_evidence
from data_pipeline.clean_dedup import MinHashLSH, clean_line, clean_text, numeric_jaccard
from data_pipeline.crawl_cninfo import linearize_key_table
from data_pipeline.fin_text import extract_citations, extract_values, parse_periods, values_equal
from evaluation.metrics import score_answer
from model.model_lora import apply_lora, lora_disabled, lora_parameters, merge_lora_inplace
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from rag.retriever import ChunkRetriever, chunk_document

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_value_normalization():
    v = extract_values('营业收入为12,345.67万元，同比下降7.83%，每股收益0.35元/股，ROE减少0.52个百分点')
    assert [m['cls'] for m in v] == ['currency', 'percent', 'per_share', 'pp']
    assert values_equal(v[0]['value'], 123456700.0)
    assert v[1]['value'] == -7.83 and v[3]['value'] == -0.52
    assert extract_values('1.23亿元')[0]['value'] != extract_values('1.23万元')[0]['value']


def test_period_parsing():
    assert parse_periods('恒晟电子2024年第一季度营业收入') == [(2024, 'Q1')]
    assert parse_periods('2023年1-6月') == [(2023, 'H1')]
    assert parse_periods('截至2024年9月30日，总资产') == [(2024, 'Q3')]
    assert parse_periods('截至2024年8月31日') == [(2024, None)]
    assert parse_periods('2022年度利润分配') == [(2022, 'FY')]


def _target(**kw):
    t = {'kind': 'value', 'cls': 'currency', 'values': [123456700.0], 'period': (2024, 'Q1'), 'support': [2],
         'displays': [['12,345.67', '万元']]}
    t.update(kw)
    return t


def test_score_answer():
    gold = '某公司2024年第一季度营业收入为12,345.67万元[2]。'
    assert score_answer(gold, _target())['correct']
    assert score_answer(gold.replace('[2]', '[1]'), _target())['cite_ok'] is False
    unit_err = score_answer(gold.replace('万元', '亿元'), _target())
    assert unit_err['num_ok'] is False and unit_err['unit_ok'] is False
    assert score_answer(gold.replace('2024年第一季度', '2023年第一季度'), _target())['period_ok'] is False
    assert score_answer('某公司2024年第一季度营业收入为123,456,700.00元[2]。', _target())['num_ok']  # 单位换算后相等
    assert score_answer('根据所提供的证据，无法回答该问题。', {'kind': 'refusal'})['refusal_ok']
    assert extract_citations('a[1]b[3]') == [1, 3]


def test_cleaning_and_dedup():
    assert clean_line('营业收入：１，２３４．５６ 万元， 同比') == '营业收入：1,234.56万元，同比'
    assert clean_text('正文\nXX公司 年度报告  第3页 共12页\n正文2') == '正文\n正文2'
    a = '营业收入（万元）：本报告期为1,234.56，上年同期为1,111.11，增减11.11%。' * 5
    b = a.replace('本报告期', '本期', 1)
    c = a.replace('1,234.56', '2,345.67').replace('1,111.11', '2,222.22').replace('11.11', '5.55')
    lsh = MinHashLSH()
    sa, sb = lsh.signature(a), lsh.signature(b)
    assert (sa == sb).mean() > 0.85 and numeric_jaccard(a, b) == 1.0
    assert numeric_jaccard(a, c) < 0.5  # 同模板不同数值：不是重复


def test_lora_merge_equivalence():
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=128))
    model.eval()
    x = torch.randint(0, 128, (2, 10))
    base = model(x).logits
    apply_lora(model, rank=4, alpha=8)
    for p in lora_parameters(model):
        torch.nn.init.normal_(p, std=0.05)
    with_lora = model(x).logits
    with lora_disabled(model):
        assert torch.allclose(model(x).logits, base, atol=1e-5)
    merged = merge_lora_inplace(copy.deepcopy(model))
    assert not lora_parameters(merged)
    assert torch.allclose(merged(x).logits, with_lora, atol=1e-4)
    assert not torch.allclose(with_lora, base, atol=1e-3)


def test_sft_mask_only_trains_assistant():
    from transformers import AutoTokenizer

    from dataset.lm_dataset import tokenize_with_mask
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
    msgs = [{'role': 'system', 'content': '系统'}, {'role': 'user', 'content': '问题A'}, {'role': 'assistant', 'content': '回答A[1]。'},
            {'role': 'user', 'content': '问题B'}, {'role': 'assistant', 'content': '回答B[1]。'}]
    ids, mask = tokenize_with_mask(tok, msgs, 512)
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    assert trained == '回答A[1]。<|im_end|>回答B[1]。<|im_end|>'
    ids, mask = tokenize_with_mask(tok, msgs, 512, train_last_only=True)
    assert tok.decode([i for i, m in zip(ids, mask) if m]) == '回答B[1]。<|im_end|>'


def test_chunk_and_retrieve():
    rows = ['营业收入（万元）：本报告期为1,234.56，上年同期为1,111.11，增减11.11%。',
            '归属于上市公司股东的净利润（万元）：本报告期为234.56，上年同期为211.11，增减11.11%。']
    chunks = chunk_document('\n'.join(rows), max_chars=60)
    assert chunks == rows  # 行不被切断
    meta = {'code': '900001', 'short_name': '测试公司', 'publish_date': '2024-04-20'}
    docs = [{'chunk_id': str(i), 'title': '测试公司：2024年第一季度报告', 'text': t, **meta} for i, t in enumerate(rows)]
    r = ChunkRetriever(docs)
    assert r.search('2024年一季度归母净利润', code='900001', topk=1)[0][0]['chunk_id'] == '1'
    assert r.find_code('测试公司的营收') == '900001'


def test_linearize_real_layout():
    text = ('单位：元 币种：人民币\n营业收入 45,775,517,043.29 18.11\n归属于上市公司股东的扣除非经常性损\n24,051,471,185.69 15.75\n益的净利润\n'
            '加权平均净资产收益率（%） 10.57 增加0.57个百分点\n总资产 285,524,543,268.38 272,699,660,092.25 4.70')
    out, facts = linearize_key_table(text, 2024, 'Q1')
    assert '营业收入（元）：本报告期为45,775,517,043.29，增减18.11%。' in out
    got = {(f['kind'], f['metric']): f['value'] for f in facts}
    assert got[('value', 'dnp')] == 24051471185.69
    assert got[('yoy', 'roe')] == 0.57 and got[('prev', 'assets')] == 272699660092.25


def test_permute_evidence_keeps_citations_on_the_same_text():
    blocks = ['《A：2024年第一季度报告》营业收入（元）：本报告期为100.00。',
              '《A：2023年第一季度报告》营业收入（元）：本报告期为90.00。\n基本每股收益（元/股）：本报告期为0.10。',
              '《A：2024年半年度报告摘要》营业收入（元）：本报告期为200.00。']
    user = '参考证据：\n' + '\n'.join(f'[{i + 1}]{b}' for i, b in enumerate(blocks)) + '\n\n问题：A公司2024年一季度营收？'
    prompt = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': user},
              {'role': 'assistant', 'content': '营业收入为100.00元[1]。'}, {'role': 'user', 'content': '那每股收益呢？'}]
    pair = {'chosen': prompt + [{'role': 'assistant', 'content': '基本每股收益为0.10元/股[2]。'}],
            'rejected': prompt + [{'role': 'assistant', 'content': '基本每股收益为0.10元/股[3]。'}],
            'target': {'kind': 'value', 'support': [2], 'cite': 2}}

    def cited_text(messages, answer):
        ev = messages[1]['content'].split('\n\n问题：')[0]
        found = dict(re.findall(r'\[(\d)\](《.*?)(?=\n\[\d\]《|$)', ev, re.S))
        return [found[k] for k in re.findall(r'\[(\d)\]', answer)]

    for seed in range(6):
        out = permute_evidence(pair, random.Random(seed))
        assert out['chosen'][:-1] == out['rejected'][:-1]
        for side in ('chosen', 'rejected'):
            for before, after in zip(pair[side], out[side]):
                if before['role'] == 'assistant':
                    assert cited_text(pair[side], before['content']) == cited_text(out[side], after['content'])
        assert cited_text(out['chosen'], f"[{out['target']['cite']}]") == [blocks[1]]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
