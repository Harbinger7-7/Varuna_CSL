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

import os
import re
import argparse
import hashlib
import fitz  # PyMuPDF
import chromadb
from uuid import uuid4
from sentence_transformers import SentenceTransformer

# Paths and settings
BASE_DIR     = "C:/Users/user/Desktop/Internship_Projects/LLM_Mistral_base"
DEFAULT_PDF  = os.path.join(BASE_DIR, "data", "Materials_Manual.pdf")
DEFAULT_DB   = os.path.join(BASE_DIR, "Storage_m_varuna")
COLLECTION   = "rules"

# Extracts raw text from PDF
def iter_pdf_text(path):
    with fitz.open(path) as doc:
        for page in doc:
            yield page.get_text()

# Splits long text into chunks
def split_into_chunks(text, max_words=150):
    paras = re.split(r"\n{2,}", text)
    for para in paras:
        sentences = re.split(r"(?<=[.!?]) +", para)
        current, length = [], 0
        for s in sentences:
            words = s.strip().split()
            if length + len(words) > max_words:
                if current:
                    yield " ".join(current)
                    current, length = [], 0
            current.append(s.strip())
            length += len(words)
        if current:
            yield " ".join(current)

# Computes PDF file
def pdf_sha256(path, buf_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()

# Builds ChromaDB index from PDF
def build_index(pdf_path=DEFAULT_PDF, db_dir=DEFAULT_DB):
    client = chromadb.PersistentClient(path=db_dir)
    pdf_hash = pdf_sha256(pdf_path)

    try:
        collection = client.get_collection(COLLECTION)
        if collection.metadata.get("pdf_sha256") == pdf_hash and collection.count() > 0:
            return
    except:
        pass

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = []
    for page_text in iter_pdf_text(pdf_path):
        chunks.extend(split_into_chunks(page_text))

    embeddings = embedder.encode(chunks, batch_size=64, show_progress_bar=True).tolist()

    try:
        client.delete_collection(COLLECTION)
    except:
        pass

    collection = client.create_collection(name=COLLECTION, metadata={"pdf_sha256": pdf_hash})
    ids = [str(uuid4()) for _ in chunks]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or refresh the ChromaDB index.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the PDF file.")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to Chroma storage directory.")
    args = parser.parse_args()
    build_index(args.pdf, args.db)
