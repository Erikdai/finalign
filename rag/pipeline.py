"""
FinRAG：检索公告证据 → 拼装带编号的证据 → 对齐后的模型生成带引用答案 → 回填引用来源

两种接入方式：
  1. 自带检索：FinRAG(chunks_path=...) 使用公告块索引（BM25 + 公司元数据过滤）
  2. 外部检索：上游研报 RAG 流程已召回证据时，直接调用 answer(question, evidence=[{title, text}, ...])
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

import torch

from data_pipeline.fin_text import SYSTEM_PROMPT, extract_citations
from dataset.lm_dataset import IM_END, render_chat
from rag.retriever import ChunkRetriever, format_evidence


class FinRAG:
    def __init__(self, model, tokenizer, chunks_path=None, topk=3, device='cpu', max_new_tokens=160):
        self.model, self.tokenizer, self.topk, self.device = model.eval(), tokenizer, topk, device
        self.max_new_tokens = max_new_tokens
        self.retriever = None
        if chunks_path:
            with open(chunks_path, encoding='utf-8') as f:
                self.retriever = ChunkRetriever([json.loads(line) for line in f])

    def retrieve(self, question, code=None):
        if self.retriever is None:
            return []
        code = code or self.retriever.find_code(question)
        query = question
        if code:
            short = self.retriever.chunks[self.retriever.by_code[code][0]]['short_name']
            query = question.replace(short, '')
        return [c for c, _ in self.retriever.search(query, code=code, topk=self.topk)]

    def build_messages(self, question, evidence, history=()):
        """history: 之前轮次的 [{'role','content'}]，证据只放在首轮用户消息中（与训练格式一致）"""
        if history:
            return [{'role': 'system', 'content': SYSTEM_PROMPT}, *history, {'role': 'user', 'content': question}]
        user = f'参考证据：\n{format_evidence(evidence)}\n\n问题：{question}'
        return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]

    @torch.inference_mode()
    def generate(self, messages, temperature=0.0, top_p=0.9):
        prompt, _ = render_chat(messages, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(self.device)
        im_end = self.tokenizer.convert_tokens_to_ids(IM_END)
        kwargs = {'do_sample': temperature > 0, 'eos_token_id': im_end, 'pad_token_id': self.tokenizer.pad_token_id,
                      'max_new_tokens': self.max_new_tokens}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p)
        out = self.model.generate(inputs=enc['input_ids'], attention_mask=enc['attention_mask'], **kwargs)
        ids = out[0, enc['input_ids'].shape[1]:].tolist()
        ids = ids[:ids.index(im_end)] if im_end in ids else ids
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def answer(self, question, evidence=None, history=(), code=None, **gen_kwargs):
        if not history and evidence is None:
            evidence = self.retrieve(question, code)
        evidence = [e if 'title' in e else {**e, 'title': e.get('source', '外部证据')} for e in (evidence or [])]
        messages = self.build_messages(question, evidence, history)
        text = self.generate(messages, **gen_kwargs)
        cited = sorted(set(extract_citations(text)))
        sources = [{'index': i, 'title': evidence[i - 1]['title'], 'text': evidence[i - 1]['text']}
                   for i in cited if evidence and 1 <= i <= len(evidence)]
        return {'answer': text, 'citations': sources, 'evidence': evidence, 'messages': messages + [{'role': 'assistant', 'content': text}]}
