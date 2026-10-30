"""
公告切块 + BM25 检索（无第三方依赖，中文按"单字+双字"切词）
数据构造（为问题检索证据、挖掘难负例）与在线 RAG 服务共用同一套检索逻辑。
"""
import math
import re
from collections import Counter, defaultdict

_SENT_SPLIT_RE = re.compile(r'(?<=[。；！？])')
_TOKEN_RE = re.compile(r'[一-鿿]|[A-Za-z]+|\d+(?:[.,]\d+)*')


def chunk_document(text, max_chars=100, hard_max=300):
    """按行（表格行/段落）贪心打包成块；单行超过 hard_max 时再按句号切分。行内不切断，保证一条事实完整落在一个块中"""
    units = []
    for line in text.split('\n'):
        if len(line) <= hard_max:
            units.append(line)
        else:
            units.extend(s for s in _SENT_SPLIT_RE.split(line) if s)
    chunks, buf = [], ''
    for u in units:
        if buf and len(buf) + len(u) + 1 > max_chars:
            chunks.append(buf)
            buf = u
        else:
            buf = f'{buf}\n{u}' if buf else u
    if buf:
        chunks.append(buf)
    return chunks


def tokenize(text):
    toks = _TOKEN_RE.findall(text)
    han = [t for t in toks if len(t) == 1 and '一' <= t <= '鿿']
    return toks + [a + b for a, b in zip(han, han[1:])]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / max(1, len(self.lens))
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.inv = defaultdict(list)
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.inv[t].append(i)

    def scores(self, query):
        q = Counter(tokenize(query))
        s = defaultdict(float)
        for t in q:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i in self.inv[t]:
                f = self.tfs[i][t]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
        return s

    def search(self, query, topk=5, candidates=None):
        s = self.scores(query)
        ids = candidates if candidates is not None else s.keys()
        return sorted(((i, s.get(i, 0.0)) for i in ids), key=lambda x: -x[1])[:topk]


class ChunkRetriever:
    """带元数据过滤的检索器：先按公司过滤（对应 RAG 中的实体识别/元数据过滤），再在候选块中做 BM25"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.by_code = defaultdict(list)
        for i, c in enumerate(chunks):
            self.by_code[c['code']].append(i)
        self._global, self._per_code = None, {}

    @staticmethod
    def index_text(c):
        """公司名由元数据过滤负责，索引中去掉，避免每个块都因公司名命中而让页眉块排到前面"""
        return f"{c['title'].split('：')[-1]}\n{c['text'].replace(c['short_name'], '')}"

    def _index(self, code):
        """按公司懒加载独立的 BM25 索引；不指定公司时使用全局索引"""
        if code is None:
            if self._global is None:
                self._global = (list(range(len(self.chunks))), BM25([self.index_text(c) for c in self.chunks]))
            return self._global
        if code not in self._per_code:
            ids = self.by_code.get(code, [])
            self._per_code[code] = (ids, BM25([self.index_text(self.chunks[i]) for i in ids]))
        return self._per_code[code]

    def search(self, query, code=None, topk=5, exclude=(), where=None):
        """where(chunk) 为额外的过滤条件（例如只取含数值事实的块作为难负例）"""
        ids, bm25 = self._index(code)
        cand = None
        if exclude or where:
            cand = [j for j, i in enumerate(ids) if self.chunks[i]['chunk_id'] not in exclude and (where is None or where(self.chunks[i]))]
        return [(self.chunks[ids[j]], s) for j, s in bm25.search(query, topk, cand)]

    def find_code(self, text):
        """从问题中识别公司（简称/代码匹配），返回 code 或 None"""
        best = None
        for code, ids in self.by_code.items():
            c = self.chunks[ids[0]]
            if (c['short_name'] in text or code in text) and (best is None or len(c['short_name']) > len(best[1])):
                best = (code, c['short_name'])
        return best[0] if best else None


def format_evidence(chunks):
    """证据统一格式：[n]《公告标题》正文（标题携带公司与报告期，是区分报告期的关键线索）"""
    return '\n'.join(f"[{i + 1}]《{c['title']}》{c['text']}" for i, c in enumerate(chunks))
