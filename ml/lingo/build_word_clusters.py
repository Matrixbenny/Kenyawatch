"""Learn word clusters from unlabelled Kenyan text (Sheng, Kiswahili and others).

Words used in similar contexts get similar vectors (positive pointwise mutual
information over a sliding window, reduced with a truncated SVD), and the
vectors are grouped into clusters with k-means. Adding a word's cluster as a
feature lets the model treat an unseen street word ("mullah") like a seen one
("pesa") when they are used alike (Turian, Ratinov & Bengio 2010).

Sources (all local, git-ignored, used with permission or under their licence):
LughaGen corpus (CC BY-SA 4.0: Sheng, Kiswahili, Luo, Kikuyu, Kamba), the
English-Sheng parallel corpus and Sheng dictionary examples (owner's
permission), and the licence-checked MasakhaNEWS and Wikipedia samples. The
project's own labelled corpora are deliberately excluded, so no evaluation text
influences the clusters.

    python -m ml.lingo.build_word_clusters
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans

from ml.src.text_prep import normalize_text

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml" / "data" / "raw"
OUT = ROOT / "ml" / "lingo" / "word_clusters.json"
NEIGHBOURS = RAW / "lughagen" / "neighbours_sample.json"
WORD_CAPS = {"Sheng_synthetic.jsonl": None, "swahili_synthetic.jsonl": 8_000_000,
             "luo_synthetic.jsonl": 1_000_000, "kikuyu_synthetic.jsonl": 1_000_000,
             "kamba_synthetic.jsonl": 1_000_000}
MIN_COUNT, MAX_VOCAB, WINDOW, DIM, CLUSTERS = 5, 60000, 4, 128, 1000


def token_list(text: str) -> list[str]:
    return [t for t in normalize_text(text).split() if t.isalpha() and len(t) >= 2]


def iter_documents():
    for name, cap in WORD_CAPS.items():
        path = RAW / "lughagen" / name
        if not path.is_file():
            print(f"  (missing {name}, skipped)")
            continue
        words = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    toks = token_list(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
                words += len(toks)
                yield toks
                if cap and words >= cap:
                    break
        print(f"  {name}: {words:,} words")
    parallel = ROOT / "ml" / "sheng.md"
    if parallel.is_file():
        with parallel.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                yield token_list(row.get("Sheng", ""))
                yield token_list(row.get("English", ""))
    dictionary = RAW / "sheng" / "sheng_dictionary.txt"
    if dictionary.is_file():
        for line in dictionary.read_text(encoding="utf-8").splitlines():
            yield token_list(line)
    for path in sorted((RAW / "lingo").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                yield token_list(row.get("text", "") or row.get("content", ""))


def main() -> int:
    start = time.time()
    print("reading corpora ...")
    docs = [d for d in iter_documents() if len(d) >= 2]
    counts = Counter(t for d in docs for t in d)
    vocab = [w for w, c in counts.most_common(MAX_VOCAB) if c >= MIN_COUNT]
    index = {w: i for i, w in enumerate(vocab)}
    print(f"documents {len(docs):,} | tokens {sum(counts.values()):,} | vocabulary {len(vocab):,}")

    print("counting co-occurrences ...")
    rows, cols, vals = [], [], []
    batch_rows, batch_cols, batch_vals = [], [], []
    matrix = sparse.csr_matrix((len(vocab), len(vocab)), dtype=np.float32)
    for n, d in enumerate(docs, 1):
        ids = np.array([index.get(t, -1) for t in d], dtype=np.int64)
        for offset in range(1, WINDOW + 1):
            a, b = ids[:-offset], ids[offset:]
            keep = (a >= 0) & (b >= 0)
            if keep.any():
                w = np.full(keep.sum(), 1.0 / offset, dtype=np.float32)
                batch_rows += [a[keep], b[keep]]
                batch_cols += [b[keep], a[keep]]
                batch_vals += [w, w]
        if n % 20000 == 0 or n == len(docs):
            matrix = matrix + sparse.csr_matrix(
                (np.concatenate(batch_vals), (np.concatenate(batch_rows), np.concatenate(batch_cols))),
                shape=matrix.shape)
            batch_rows, batch_cols, batch_vals = [], [], []

    print("PPMI and SVD ...")
    matrix = matrix.tocoo()
    total = matrix.sum()
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    col_sum = np.asarray(matrix.sum(axis=0)).ravel() ** 0.75
    col_sum = col_sum / col_sum.sum()
    pmi = np.log((matrix.data / total) / ((row_sum[matrix.row] / total) * col_sum[matrix.col]))
    keep = pmi > 0
    ppmi = sparse.csr_matrix((pmi[keep].astype(np.float32), (matrix.row[keep], matrix.col[keep])), shape=matrix.shape)
    u, s, _ = svds(ppmi, k=DIM, random_state=42)
    vectors = u * np.sqrt(s)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9

    print("clustering ...")
    labels = MiniBatchKMeans(n_clusters=CLUSTERS, random_state=42, batch_size=4096, n_init=3).fit_predict(vectors)

    OUT.write_text(json.dumps({
        "version": 1,
        "source": ("Word clusters (PPMI + SVD + k-means) learned from unlabelled Kenyan text: the LughaGen "
                   "Multilingual Kenyan Language Corpus (LughaGen-Organization, CC BY-SA 4.0), an English-Sheng "
                   "parallel corpus and Sheng dictionary examples used with permission, and licence-checked "
                   "MasakhaNEWS and Wikipedia samples. Shared under CC BY-SA 4.0. No labelled project data was used."),
        "clusters": {w: int(c) for w, c in zip(vocab, labels)},
    }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    probe = ["pesa", "doo", "ganji", "mullah", "chapaa", "polisi", "karao", "kanjo", "hongo", "rushwa", "chai",
             "msee", "manzi", "tender", "kazi", "ofisi", "tishia", "iba", "fake", "cheti", "binamu", "ndugu"]
    neighbours = {}
    for w in probe:
        if w in index:
            sims = vectors @ vectors[index[w]]
            neighbours[w] = [vocab[i] for i in np.argsort(-sims)[1:9]]
    NEIGHBOURS.write_text(json.dumps(neighbours, ensure_ascii=False, indent=1), encoding="utf-8")
    for w, ns in neighbours.items():
        print(f"  {w:8s} -> {', '.join(ns)}")
    print(f"done in {time.time() - start:.0f}s; clusters written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
