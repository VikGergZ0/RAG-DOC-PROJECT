import json
from pathlib import Path
import numpy as np

from .embedder import embed_texts, embed_query


class VectorIndex:
    def __init__(self, records: list[dict], embeddings: np.ndarray):
        self.records = records
        self.embeddings = embeddings

    @classmethod
    def build(cls, records: list[dict]) -> "VectorIndex":
        texts = [r["text"] for r in records]
        embeddings = embed_texts(texts)
        return cls(records, embeddings)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        q = embed_query(query)
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            rec = dict(self.records[i])
            rec["score"] = float(scores[i])
            results.append(rec)
        return results

    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.embeddings)
        (out / "records.json").write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, out_dir: str) -> "VectorIndex":
        out = Path(out_dir)
        embeddings = np.load(out / "embeddings.npy")
        records = json.loads((out / "records.json").read_text(encoding="utf-8"))
        return cls(records, embeddings)