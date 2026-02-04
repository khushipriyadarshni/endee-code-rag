import os
import json
from typing import List, Dict, Tuple
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

ENDEE_URL = os.getenv("ENDEE_URL")
ENDEE_API_KEY = os.getenv("ENDEE_API_KEY")

# load local embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


class EndeeStore:
    """A minimal Endee wrapper. If ENDEE_URL and ENDEE_API_KEY are set, uses REST endpoints.
    Otherwise falls back to an in-memory local store persisted under `repo/endee_local_store.json`.
    """

    def __init__(self):
        self.endee_url = ENDEE_URL
        self.endee_api_key = ENDEE_API_KEY
        self.local_path = os.path.join("repo", "endee_local_store.json")
        self._vectors = []
        self._chunks = []
        self._metadatas = []

        if not os.path.exists("repo"):
            os.makedirs("repo", exist_ok=True)

        if not self.endee_url and os.path.exists(self.local_path):
            try:
                with open(self.local_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._chunks = data.get("chunks", [])
                    self._metadatas = data.get("metadatas", [])
                    vectors = data.get("vectors", [])
                    self._vectors = [np.array(v, dtype=float) for v in vectors]
            except Exception:
                self._vectors = []

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _persist_local(self):
        data = {
            "chunks": self._chunks,
            "metadatas": self._metadatas,
            "vectors": [v.tolist() for v in self._vectors],
        }
        with open(self.local_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def add_chunks(self, chunks: List[str], metadatas: List[Dict]):
        if not chunks:
            return

        embeddings = self._embed(chunks)

        if self.endee_url and self.endee_api_key:
            url = self.endee_url.rstrip("/") + "/vectors/upsert"
            headers = {
                "Authorization": f"Bearer {self.endee_api_key}",
                "Content-Type": "application/json",
            }
            payload = {"vectors": []}
            for i, emb in enumerate(embeddings):
                payload["vectors"].append({
                    "id": f"local-{len(self._vectors)+i}",
                    "values": emb,
                    "metadata": metadatas[i],
                    "payload": {"text": chunks[i]},
                })
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                r.raise_for_status()
            except Exception:
                self._vectors.extend([np.array(e, dtype=float) for e in embeddings])
                self._chunks.extend(chunks)
                self._metadatas.extend(metadatas)
                self._persist_local()
        else:
            self._vectors.extend([np.array(e, dtype=float) for e in embeddings])
            self._chunks.extend(chunks)
            self._metadatas.extend(metadatas)
            self._persist_local()

    def search_chunks(self, query: str, k: int = 3) -> List[Tuple[str, Dict, float]]:
        q_emb = self._embed([query])[0]

        if self.endee_url and self.endee_api_key:
            url = self.endee_url.rstrip("/") + "/vectors/query"
            headers = {
                "Authorization": f"Bearer {self.endee_api_key}",
                "Content-Type": "application/json",
            }
            payload = {"top_k": k, "vector": q_emb}
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
                matches = data.get("matches") or data.get("results") or []
                out = []
                for m in matches[:k]:
                    text = m.get("payload", {}).get("text") or ""
                    meta = m.get("metadata") or {}
                    score = m.get("score") or 0.0
                    out.append((text, meta, float(score)))
                return out
            except Exception:
                pass

        if not self._vectors:
            return []

        qv = np.array(q_emb, dtype=float)
        mats = np.vstack(self._vectors)
        qnorm = np.linalg.norm(qv)
        norms = np.linalg.norm(mats, axis=1)
        dots = mats.dot(qv)
        sims = dots / (norms * qnorm + 1e-12)
        idx = np.argsort(-sims)[:k]

        results = []
        for i in idx:
            results.append((self._chunks[i], self._metadatas[i], float(sims[i])))
        return results


store = EndeeStore()

def add_chunks(chunks: List[str], metadatas: List[Dict]):
    return store.add_chunks(chunks, metadatas)

def search_chunks(query: str, k: int = 3):
    return store.search_chunks(query, k)
