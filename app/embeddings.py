"""Embedding back-ends (LangChain `Embeddings` objects, 384 dimensions each).

* FastEmbedEmbeddings: real semantic vectors (BAAI/bge-small-en-v1.5), CPU only.
* HashEmbeddings: dependency-free fallback using hashed character 3-grams and words.
  It is deterministic, needs no download, and is good at typo matching ("gingko" ~ "ginkgo"),
  so the demo never fails because a model could not be downloaded.
"""
from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from . import config

DIM = config.EMBED_DIM


class HashEmbeddings(Embeddings):
    name = "hash-384"
    suggest_max_dist = 0.65  # cosine distance; typos land ~0.5-0.6, unrelated names >= 0.78

    @staticmethod
    def _vec(text: str) -> list[float]:
        t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        t = re.sub(r"\s+", " ", t).strip()
        padded = f" {t} "
        feats = [padded[i : i + 3] for i in range(len(padded) - 2)]
        feats += [f"w:{w}" for w in t.split() for _ in range(2)]
        v = [0.0] * DIM
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FastEmbedEmbeddings(Embeddings):
    name = "fastembed-bge-small-en-v1.5"
    suggest_max_dist = 0.30  # bge similarities are compressed high, so the cut-off is tighter

    def __init__(self) -> None:
        from fastembed import TextEmbedding  # imported lazily

        self._model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    mode = config.EMBEDDINGS
    if mode == "hash":
        return HashEmbeddings()
    try:
        return FastEmbedEmbeddings()
    except Exception as exc:  # no fastembed, no network, unsupported Python, ...
        if mode == "fastembed":
            raise
        print(f"[embeddings] fastembed unavailable ({exc.__class__.__name__}); using hashing embedder")
        return HashEmbeddings()
