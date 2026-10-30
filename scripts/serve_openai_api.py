"""
OpenAI 兼容推理服务（参考 minimind/scripts/serve_openai_api.py），内置金融公告 RAG

  POST /v1/chat/completions
    标准 OpenAI 请求体；额外字段：
      rag: bool        为 true 时对最后一条用户消息自动检索公告证据（需启动时指定 --chunks_path）
      evidence: list   上游研报 RAG 已召回的证据 [{"title":..., "text":...}]，优先于自动检索
    响应中 choices[0].message.content 为带 [n] 引用的回答，额外字段 citations 给出被引用的证据原文
  POST /v1/finqa      精简接口：{"question": "...", "evidence": [...]?}

python scripts/serve_openai_api.py --from_weight finalign --loras none --chunks_path data/processed/chunks.jsonl --port 8998
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import FinRAG
from scripts.merge_lora import add_merge_args, load_merged


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = 'finalign'
    messages: list[Message]
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 160
    rag: bool = False
    evidence: list[dict] | None = None


class FinQARequest(BaseModel):
    question: str
    evidence: list[dict] | None = None


app = FastAPI(title='FinAlign API')
rag: FinRAG = None


@app.post('/v1/chat/completions')
def chat(req: ChatRequest):
    msgs = [m.model_dump() for m in req.messages if m.role != 'system']
    rag.max_new_tokens = req.max_tokens
    citations = []
    if req.evidence is not None or req.rag:
        # 只把最后一条用户消息当作问题，之前的多轮对话作为历史（首轮已含证据时直接沿用）
        question, history = msgs[-1]['content'], msgs[:-1]
        result = rag.answer(question, evidence=req.evidence, history=history, temperature=req.temperature, top_p=req.top_p)
        text, citations = result['answer'], result['citations']
    else:
        text = rag.generate([m.model_dump() for m in req.messages], temperature=req.temperature, top_p=req.top_p)
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex[:12]}', 'object': 'chat.completion', 'created': int(time.time()), 'model': req.model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
        'citations': citations,
    }


@app.post('/v1/finqa')
def finqa(req: FinQARequest):
    r = rag.answer(req.question, evidence=req.evidence)
    return {'answer': r['answer'], 'citations': r['citations'], 'evidence': [{'title': e['title'], 'text': e['text']} for e in r['evidence']]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FinAlign OpenAI 兼容服务")
    add_merge_args(parser)
    parser.add_argument('--chunks_path', type=str, default='', help="公告块索引（data/processed/chunks.jsonl），为空则只支持外部证据")
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8998)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    model, tokenizer = load_merged(args, device=args.device)
    rag = FinRAG(model, tokenizer, chunks_path=args.chunks_path or None, topk=args.topk, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)
