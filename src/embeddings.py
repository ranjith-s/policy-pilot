"""
embeddings.py — Semantic retrieval layer (scales to the full ~4000-scheme corpus).

Design:
  * Embeddings computed OFFLINE once via `python src/embeddings.py build`
    (uses local Ollama's nomic-embed-text — `ollama pull nomic-embed-text`).
  * Stored as data/embeddings.npy (float32 matrix) + data/embedding_ids.json.
    At 4000 schemes x 768 dims that's ~12 MB — no vector DB needed.
  * At query time: cosine similarity via numpy. Milliseconds at this scale.
  * If the index or Ollama is unavailable, callers fall back to keyword search
    (see tools.search_schemes) — retrieval degrades, never dies.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMB_PATH = DATA_DIR / "embeddings.npy"
IDS_PATH = DATA_DIR / "embedding_ids.json"


class OllamaEmbedder:
    def __init__(self, model="nomic-embed-text", host="http://localhost:11434"):
        self.model, self.host = model, host

    def embed_batch(self, texts, retries=3):
        """Embed a list of texts in one call (/api/embed batch endpoint —
        far faster than one HTTP request per text)."""
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps({"model": self.model,
                             "input": [t[:2000] for t in texts]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        # first call after a model swap can be slow (model loading on a small
        # GPU); retry transient timeouts with backoff instead of dying
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read())
                return np.array(data["embeddings"], dtype=np.float32)
            except (TimeoutError, OSError):
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)

    def embed(self, text, retries=3):
        return self.embed_batch([text], retries=retries)[0]


def _doc_text(doc):
    """What we embed per scheme: name + tags + brief + eligibility.

    Deliberately NOT the full document — names/briefs/eligibility carry the
    semantics users search with; application-process steps just add noise.
    """
    return " | ".join(filter(None, [
        doc["scheme_name"],
        " ".join(doc.get("tags", [])),
        " ".join(doc.get("categories", [])),
        doc.get("brief_description", ""),
        doc.get("eligibility_text", "")[:600],
    ]))


def build_index(embedder=None, corpus_path=None, verbose=True):
    """One-off offline job. Re-run whenever rag_corpus.json changes."""
    embedder = embedder or OllamaEmbedder()
    with open(corpus_path or DATA_DIR / "rag_corpus.json", encoding="utf-8") as f:
        docs = json.load(f)

    ids = [d["id"] for d in docs]
    texts = [_doc_text(d) for d in docs]

    BATCH = 64
    chunks = []
    for i in range(0, len(texts), BATCH):
        chunks.append(embedder.embed_batch(texts[i:i + BATCH]))
        if verbose:
            print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)

    mat = np.vstack(chunks)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)  # pre-normalise
    np.save(EMB_PATH, mat)
    IDS_PATH.write_text(json.dumps(ids))
    if verbose:
        print(f"saved {mat.shape} -> {EMB_PATH}")
    return mat, ids


class SemanticIndex:
    """Loaded once; query() returns [(scheme_id, score)] best-first."""

    def __init__(self, embedder=None):
        self.embedder = embedder or OllamaEmbedder()
        self.mat = np.load(EMB_PATH)
        self.ids = json.loads(IDS_PATH.read_text())

    @staticmethod
    def available():
        return EMB_PATH.exists() and IDS_PATH.exists()

    def query(self, text, top_k=10, restrict_ids=None):
        q = self.embedder.embed(text)
        q /= (np.linalg.norm(q) + 1e-9)
        scores = self.mat @ q                      # cosine (rows pre-normalised)
        order = np.argsort(-scores)
        out = []
        allowed = set(restrict_ids) if restrict_ids is not None else None
        for idx in order:
            sid = self.ids[idx]
            if allowed is not None and sid not in allowed:
                continue
            out.append((sid, float(scores[idx])))
            if len(out) >= top_k:
                break
        return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        print("usage: python src/embeddings.py build")
