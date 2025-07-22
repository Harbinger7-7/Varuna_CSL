"""
===============================================================================
 Project: Varuna — Shipyard Compliance Assistant

 Description:
     Local LLM-powered assistant to evaluate engineering procurement queries
     based on regulatory frameworks like GFR, DPP, GeM SOP, BIS/ISO standards.

 LLM: Mistral-7B-Instruct (GGUF via llama.cpp)
 RAG: ChromaDB + all-MiniLM-L6-v2 embeddings
 Prompt: Rule-based strict audit prompt

 Developers:
     - Anand Raj
       B.Tech Artificial Intelligence and Data Science, 2025
       Rajagiri School of Engineering and Technology, Kochi

     - Kestelyn Sunil Jacob
       B.Tech Artificial Intelligence and Data Science, 2025
       Rajagiri School of Engineering and Technology, Kochi
===============================================================================
"""
import os, re, heapq, threading
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Paths and settings
BASE_DIR    = "C:/Users/user/Desktop/Internship_Projects/LLM_Mistral_base"
MODEL_PATH  = os.path.join(BASE_DIR, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
PROMPT_PATH = os.path.join(BASE_DIR, "Prompt_template.txt")
DB_DIR      = os.path.join(BASE_DIR, "Storage_m_varuna")
COLLECTION  = "rules"

# Model settings
CPU_THREADS  = max(os.cpu_count() - 1, 1)
CTX_SIZE     = 4096
MAX_NEW_TOK  = 640
TEMP         = 0.2
TOP_P        = 0.8
REPEAT_PEN   = 1.1
N_BATCH      = 64
TOP_K_RETR   = 3
TOP_K_FOOTER = 3
N_GPU_LAYERS = 0

# Load model
llm = Llama(
    model_path   = MODEL_PATH,
    n_ctx        = CTX_SIZE,
    n_threads    = CPU_THREADS,
    n_batch      = N_BATCH,
    n_gpu_layers = N_GPU_LAYERS,
    use_mlock    = True,
    verbose      = False,
)

# Load embedder and DB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client   = chromadb.PersistentClient(path=DB_DIR)
col      = client.get_or_create_collection(COLLECTION)

# Interrupt control
_interrupt = threading.Event()
def set_interrupt() -> None: _interrupt.set()
def _clr() -> None: _interrupt.clear()

# Text cleaners
_ws  = re.compile(r"[ \t]+")
_dnl = re.compile(r"\n\s*\n+")
def _clean(t: str) -> str: return _ws.sub(" ", _dnl.sub("\n\n", t)).strip()
def _sim(a, b) -> float: return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# Auto-fact extraction
MONTHS = {m: i+1 for i, m in enumerate([
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"])}

def _extract_facts(q: str) -> str:
    nums = list(map(int, re.findall(r"\b\d+\b", q)))
    dates = re.findall(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}", q, flags=re.I)
    out = []
    if len(nums) >= 2:
        out.append(f"Original qty={nums[0]}, Requested qty={nums[1]}")
    if len(dates) >= 2:
        def _p(d): m, y = d.split(); return MONTHS[m[:3].lower()], int(y)
        (m1, y1), (m2, y2) = _p(dates[0]), _p(dates[1])
        out.append(f"Gap={(y2 - y1) * 12 + (m2 - m1)} months between {dates[0]} and {dates[1]}")
    return "; ".join(out)

@lru_cache(maxsize=1)
def _template() -> str:
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError(f"Prompt missing: {PROMPT_PATH}")
    return open(PROMPT_PATH, encoding="utf-8").read()

def _build_prompt(user_q: str, ctx: str) -> str:
    facts = _extract_facts(user_q)
    if facts:
        user_q += f"\n\n[🔎 Auto‑Facts] {facts}"
    return (_template()
            .replace("{{current_date}}", datetime.now().strftime("%d %B %Y"))
            .replace("{{contextual_chunks_here}}", ctx)
            .replace("{{user_question_here}}", user_q))

# Retrieval
def _retrieve(query: str, k: int = TOP_K_RETR) -> List[str]:
    qv = embedder.encode([query])[0]
    res = col.query(query_embeddings=[qv.tolist()], n_results=k, include=["documents"])
    return res.get("documents", [[]])[0]

# Re-ranking
def _rerank(query: str, chunks: List[str], k: int = TOP_K_FOOTER) -> str:
    if not chunks: return ""
    qv = embedder.encode([query])[0]
    cvs = embedder.encode(chunks)
    qt = set(re.findall(r"\b\w+\b", query.lower()))
    sc = []
    for emb, txt in zip(cvs, chunks):
        cos = _sim(qv, emb)
        toks = set(re.findall(r"\b\w+\b", txt.lower()))
        jac = len(qt & toks) / (len(qt | toks) + 1e-10)
        sc.append((0.7 * cos + 0.3 * jac, txt))
    return "\n\n".join(t for _, t in heapq.nlargest(k, sc))

# Format enforcement
def _enforce_structure(t: str) -> bool:
    req = ["Action Required", "Procedure", "Clause Justification"]
    return all(s.lower() in t.lower() for s in req)

def _contains_weak(t: str) -> bool:
    return any(re.search(p, t, re.I) for p in [r"\bmay\b", r"\bshould\b", r"\bcould\b", r"\bdepends on\b"])

def _fix_format(t: str) -> str:
    parts = {s: "" for s in ["Action Required", "Procedure", "Clause Justification"]}
    for s in parts:
        m = re.search(f"{s}:(.*?)(?=(Action Required:|Procedure:|Clause Justification:|$))", t, re.S)
        if m: parts[s] = m.group(1).strip()
    return f"Action Required:\n{parts['Action Required']}\n\nProcedure:\n{parts['Procedure']}\n\nClause Justification:\n{parts['Clause Justification']}"

# Streaming generation
def answer_stream(question: str, history: List[Tuple[str, str]] | None = None):
    _clr()
    chunks = _retrieve(question)
    ctx_text = "\n\n".join(chunks)
    prompt = _build_prompt(question, ctx_text)
    footer = ("\n\n---\nRelevant Clauses:\n" + _rerank(question, chunks)) if chunks else "\n\n---\nRelevant Clauses: Not found."

    buf, pending, last, full = "", "", "", ""

    for part in llm(prompt,
                    max_tokens=MAX_NEW_TOK,
                    temperature=TEMP,
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PEN,
                    stop=["###"],
                    stream=True):
        if _interrupt.is_set():
            _clr(); return
        pending += part["choices"][0]["text"]
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if line and line != last:
                last = line
                buf += line + "\n"
                full += line + "\n"
                yield _clean(buf)

    _clr()

    if pending.strip() and pending.strip() != last:
        full += pending.strip() + "\n"
        buf += pending.strip() + "\n"
        yield _clean(buf)

    structured = _clean(full).split("###")[0].rstrip()
    if not _enforce_structure(structured) or _contains_weak(structured):
        yield f"\n\n[Corrected Format Applied]\n{_fix_format(structured)}"
    else:
        yield structured + footer

def get_answer_stream(q: str, history=None):
    return answer_stream(q, history)

# CLI interface
if __name__ == "__main__":
    print("Varuna CLI — type 'exit' to quit")
    while True:
        try: q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt): break
        if q.lower() in {"exit", "quit"}: break
        for chunk in answer_stream(q): print(chunk)
