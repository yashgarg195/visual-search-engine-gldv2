# Landmarq — visual landmark search over 5M images

Identify a landmark from a single photograph by matching it against **5,052,992
images** from the [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark),
then naming it, placing it on a map, and showing the evidence behind the answer.

Built with **OpenCLIP ViT-B/32** for embeddings, **FAISS** (HNSW-PQ and IVF-PQ)
for approximate search, exact reranking from memory-mapped shards, and optional
local-feature geometric verification. Ships with a FastAPI service and a
dependency-free web UI.

> **Project status — read this first.** The full pipeline is built and runs end
> to end at 5M scale. **Retrieval accuracy has not yet been measured at scale.**
> The only committed evaluation is a 303-vector smoke test, which is far too
> small to mean anything. No recall or accuracy figure is claimed anywhere in
> this repository. See [Evaluation](#9-evaluation) for how to produce a real one.

---

## Contents

1. [What it does](#1-what-it-does)
2. [How it works](#2-how-it-works)
3. [Quickstart — prebuilt index](#3-quickstart--prebuilt-index)
4. [Quickstart — synthetic sample](#4-quickstart--synthetic-sample)
5. [The web app](#5-the-web-app)
6. [Building the index from scratch](#6-building-the-index-from-scratch)
7. [Stage-2 reranking](#7-stage-2-reranking)
8. [Index design & storage math](#8-index-design--storage-math)
9. [Evaluation](#9-evaluation)
10. [Repository layout](#10-repository-layout)
11. [Tests](#11-tests)
12. [Deploying on GCP](#12-deploying-on-gcp)

---

## 1. What it does

Give it a photo of a place you can't identify. It returns:

- **A name**, chosen by similarity-weighted voting across the top matches —
  not just the single nearest neighbour — plus the confidence margin over the
  runner-up.
- **A location**: city, country and coordinates for the identified landmark.
- **The evidence**: the matching images, their scores, and per-stage latency.

**Text queries are a late addition, included to demonstrate CLIP's shared
embedding space** rather than as a goal of the project. Because CLIP maps images
and text into the same 512-d space, the *same* index answers a typed query with
no extra structures — see `--query-text` and `POST /api/search/text`. Retrieval
quality for text queries is untuned and noticeably weaker than for photos.

## 2. How it works

The system is split into a slow offline build and a fast online query path.

**Offline, once:**

```
GLDv2 images ──► OpenCLIP ViT-B/32 ──► 512-d L2-normalised float32 vectors
                                              │
                                   atomic shards (.npy + .parquet)
                                              │
                                   FAISS HNSW-PQ / IVF-PQ index
                                   + IndexIDMap2 (permanent int64 → image_id)
```

**Online, per query:**

```
query photo ──► same encoder ──► ANN search over 5.05M compressed vectors
                                              │
                                   top-K candidates (approximate)
                                              │
                        exact rerank from memory-mapped fp32 shards
                                              │
                        optional geometric verification (SIFT/ORB/LightGlue)
                                              │
                        similarity-weighted voting ──► landmark + location
```

The core trade: product quantization shrinks each vector **28.4×** so all five
million fit in memory and can be skimmed in milliseconds, and the resulting
precision loss is bought back by rescoring only the shortlist against the
original uncompressed vectors.

**Pipeline stages in detail**

1. **Metadata preparation** — parses the official GLDv2 CSVs, preserves string
   IDs with leading zeros, resolves sharded paths
   (`{split}/{c0}/{c1}/{c2}/{id}.jpg`), and decodes readable landmark names from
   Wikimedia Commons category URLs.
2. **Feature extraction** — batched OpenCLIP ViT-B/32 encoding to 512-d
   L2-normalised float32.
3. **Atomic sharding** — `shard_XXXXX_emb.npy` + `shard_XXXXX_meta.parquet`
   with a global `embedding_manifest.json`, so an interrupted run resumes without
   duplicating work.
4. **Index build** — HNSW-PQ (current) or IVF-PQ, inner product over normalised
   vectors (equivalent to cosine), with `IndexIDMap2` for a permanent 1:1 map
   between FAISS integer IDs and GLDv2 string image IDs.
5. **Exact reranking** — candidate vectors are loaded on demand from
   memory-mapped shards and rescored by exact cosine similarity.
6. **Voting** — scores are aggregated per landmark to pick a winner and report
   the margin over the runner-up.

## 3. Quickstart — prebuilt index

Skip embedding 5M images and pull the published index from
[Releases](https://github.com/yashgarg195/visual-search-engine-gldv2/releases/tag/v1.0.0):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/download_release_assets.py      # ~520 MB total
```

This fetches the populated 5.05M-vector IVF-PQ index and the `id_mapping.parquet`
into `outputs/index/`. Then:

```bash
python3 scripts/search.py --query-image path/to/photo.jpg --top-k 10 --nprobe 16
```

> Searching returns landmark **names and IDs** from the metadata tables. Showing
> the matched *images* additionally requires the GLDv2 image files on disk; the
> web UI serves them from `data/raw/images/`.

## 4. Quickstart — synthetic sample

To exercise the whole pipeline offline with no dataset download:

```bash
./scripts/run_small_pipeline.sh none
```

This runs every stage end to end — environment audit, synthetic sample
generation, metadata validation, embedding, index training, ingestion, a demo
query with voting, and evaluation against an exact `IndexFlatIP` baseline.

## 5. The web app

A FastAPI service with a dependency-free frontend (three static files, no build
step, no framework).

```bash
python3 src/api/run.py                 # http://127.0.0.1:8000
python3 src/api/run.py --host 0.0.0.0 --port 8080 --reload
```

Interactive API docs at `/docs`.

| Route | Purpose |
|---|---|
| `GET /` | The web UI |
| `GET /api/info` | Index stats and loaded configuration |
| `POST /api/search/image` | Search by uploaded photo |
| `POST /api/search/text` | Search by text query |
| `GET /api/image/{split}/{a}/{b}/{c}/{id}.jpg` | Serve a dataset image |

Landmark enrichment (city, country, coordinates) runs through
`src/llm/location_resolver.py`. It is **optional** — without `GEMINI_API_KEY`
set, or without `google-genai` installed, the service degrades gracefully and
simply omits the location panel.

## 6. Building the index from scratch

```bash
# 1. Environment and hardware audit
python3 scripts/check_environment.py

# 2. Metadata (add --download-metadata to fetch the official CSVs)
python3 scripts/prepare_metadata.py --splits train index --verify-images

# 3. Embeddings — resumable, safe to interrupt
python3 scripts/embed_images.py --split train --batch-size 64 \
    --shard-size 10000 --device auto

# 4a. Train an HNSW-PQ index (what the 5M build uses)
python3 scripts/train_hnsw_pq.py --m 64 --hnsw-m 64 --ef-construction 256

# 4b. ...or an IVF-PQ index
python3 scripts/train_ivfpq.py --nlist 1024 --m 64 --nbits 8 --sample-size 50000

# 5. Ingest shards into the index
python3 scripts/add_embeddings.py
```

Helper runners: `scripts/run_embedding_pipeline.sh` (resumable large-scale
embedding) and `scripts/run_index_pipeline.sh` (build + evaluate).

**Searching**

```bash
python3 scripts/search.py \
    --query-image data/sample_queries/query_eiffel_tower.jpg \
    --index-type hnsw_pq --ef-search 128 --top-k 10 \
    --rerank --rerank-candidates 100
```

<details>
<summary>Sample output</summary>

```
==========================================================================================
 VISUAL LANDMARK SEARCH RESULTS (Image: query_eiffel_tower.jpg)
 nprobe: 8 | Latency: 4.88 ms search (125.12 ms embedding)
==========================================================================================

Rank  Score    Landmark Name                    Landmark ID    Image ID       Split
------------------------------------------------------------------------------------------
1     0.0890   Eiffel Tower                     1001           00002d         train
2     0.0886   Eiffel Tower                     1001           00001e         train
3     0.0884   Eiffel Tower                     1001           00003c         train

------------------------------------------------------------------------------------------
 SIMILARITY-WEIGHTED VOTING IDENTIFICATION
------------------------------------------------------------------------------------------
  Leading Candidate   : Eiffel Tower (Split: train, ID: 1001)
  Total Vote Score    : 0.4409 (from 5 supporting image matches)
  Confidence Margin   : 0.4409 over runner-up
  Category Source     : http://commons.wikimedia.org/wiki/Category:Eiffel_Tower
  Hierarchy           : monument_tower

  NOTE: Similarity scores are cosine similarities in [-1, 1] for normalized
        embeddings, not calibrated probabilities.
==========================================================================================
```

Figures above are from the synthetic sample, not the 5M index.
</details>

**Landmark metadata lookup**

```bash
python3 scripts/lookup_landmark.py --landmark-id 1001 --split train
python3 scripts/lookup_landmark.py \
    --category-url "http://commons.wikimedia.org/wiki/Category:%C3%89cole_Militaire"
```

## 7. Stage-2 reranking

`src/reranking/` holds optional, fully decoupled precision stages. **If the
package is removed, the engine falls back to coarse search with zero errors.**

- **Geometric verification** (`--geometric-rerank`) — SIFT (default) or ORB
  keypoints, FLANN matching with Lowe's 2-NN ratio test at 0.75, then
  `USAC_MAGSAC` homography estimation. Requires ≥12 inliers and checks spatial
  dispersion entropy over a 4×4 grid, so a tight cluster of matches on a
  watermark or repeated texture does not pass as a real match.
- **LightGlue matching** (`--matcher lightglue`) — learned local feature
  matching as an alternative to SIFT/FLANN.
- **Alpha query expansion** (`--alpha-qe`) — re-queries using a weighted
  combination of the query and its top results.
- **Hybrid fusion** (`src/index/fusion.py`, `src/reranking/hybrid_fusion.py`) —
  combines global-embedding and local-feature evidence into one ranking.

```bash
python3 scripts/search.py --query-image photo.jpg --geometric-rerank --top-k 10
```

> These stages are **not** enabled by default and have not been shown to improve
> accuracy on this dataset at scale. Treat them as experimental.

## 8. Index design & storage math

For 512-dimensional float32 embeddings:

- Raw vector: `512 × 4 B = 2,048 B` ≈ 2.0 KB
- PQ-coded vector (`m=64, nbits=8`): `64 B + 8 B (ID) = 72 B`
- **Compression: ≈ 28.4×**

| Vectors | Raw float32 | PQ index | Approx. RAM |
|---:|---:|---:|---:|
| 100 K | ~200 MB | ~7.2 MB | < 250 MB |
| 1 M | ~2.0 GB | ~72 MB | < 1.0 GB |
| 5 M | ~10.0 GB | ~360 MB | < 3.0 GB |

**Current 5M build** (`outputs/index/*.json`): `hnsw_pq`, `d=512`, `m=64`,
`M=64`, `ef_construction=256`, `ef_search=128`, inner product, 5,052,992
vectors, trained.

**Choosing parameters**

- `M` (HNSW) — graph connectivity. Higher is more accurate and more memory.
- `ef_construction` / `ef_search` — build and query breadth. `ef_search` is the
  accuracy/latency dial at query time; raise it first when tuning.
- `nlist` (IVF) — coarse centroids. Rule of thumb `4√N`–`16√N`; for N=1M that is
  4,096–16,384. For tiny test sets (N < 1,000) use 8–32 so each cell trains.
- `nprobe` (IVF) — cells searched per query. The recall/latency dial.
- `m` (PQ) — subquantizers; must divide `d`. For `d=512`, `m=64` gives 8-d
  sub-vectors.
- `nbits` (PQ) — bits per subquantizer; 8 means 256 centroids per codebook.

## 9. Evaluation

```bash
python3 scripts/evaluate.py --num-queries 100 --nprobe-sweep 1 4 8 16 32 64
```

Reports Recall@1/5/10 against an exact `IndexFlatIP` baseline, Top-1/Top-5
landmark recognition accuracy, and a latency sweep.

**The committed report in `outputs/evaluation/` used only 303 vectors.** It is a
smoke test that proves the harness runs; it is not a result. To produce a number
worth quoting, run this against the full 5M index with a held-out query set of
meaningful size.

If you are trying to improve accuracy, in rough order of effort:

1. **Widen the rerank pool.** Exact reranking is the strongest precision tool
   here, but it currently only ever sees the top 100 candidates.
2. **Raise `ef_search` / `nprobe`.** Costs latency, costs nothing to try.
3. **Rebuild the index.** Several build-time parameters sit below their
   recommended values, and there is no OPQ rotation.
4. **Replace the encoder.** ViT-B/32 is the ceiling on retrieval quality —
   nothing downstream can recover information it never captured. This is the
   largest available gain and the most expensive, but **does not require
   re-downloading the dataset**.

## 10. Repository layout

```
visual-search-engine-gldv2/
├── src/
│   ├── config.py                 # Typed dataclass configuration
│   ├── api/
│   │   ├── app.py                # FastAPI routes
│   │   ├── service.py            # Search orchestration
│   │   ├── run.py                # Server entrypoint
│   │   └── static/               # index.html, style.css, app.js (no build step)
│   ├── index/
│   │   ├── base.py               # BaseVectorIndex interface
│   │   ├── hnsw_pq.py            # HNSW + PQ (current 5M build)
│   │   ├── ivfpq.py              # IVF-PQ + IndexIDMap2
│   │   ├── flat.py               # Exact IndexFlatIP baseline
│   │   ├── rerank.py             # Memory-mapped exact reranker
│   │   ├── fusion.py             # Score fusion
│   │   ├── voting.py             # Similarity-weighted voting
│   │   └── threading_utils.py
│   ├── reranking/                # Optional stage 2 — safe to delete
│   │   ├── geometric_verifier.py # SIFT/ORB + MAGSAC homography
│   │   ├── lightglue_verifier.py
│   │   ├── alpha_qe.py
│   │   ├── hybrid_fusion.py
│   │   ├── pipeline.py
│   │   └── {reference,thumbnail}_fetcher.py
│   ├── models/clip_encoder.py    # OpenCLIP ViT-B/32
│   ├── storage/embedding_store.py# Atomic shards, manifest, mmap reader
│   ├── metadata/                 # CSV ingestion, URL decoding, O(1) lookups
│   ├── llm/location_resolver.py  # Optional landmark enrichment
│   └── utils/                    # hardware, io, logging
├── scripts/                      # 22 CLIs and runners (see §6)
├── tests/                        # 35 tests across 10 modules
├── web/run.py                    # Alias for src/api/run.py
└── outputs/                      # Index, manifests, evaluation reports
```

## 11. Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

35 tests across 10 modules cover URL decoding and split preservation, atomic
shard writes and resumption, HNSW-PQ and IVF-PQ parameter bounds and search,
similarity-weighted voting, geometric verification and homography estimation,
LightGlue and alpha-QE, hybrid fusion, and an end-to-end integration run.

## 12. Deploying on GCP

- **VM**: `g2-standard-8` (1× NVIDIA L4, 8 vCPU, 32 GB) for embedding;
  `n1-standard-8` is sufficient for CPU-only search and evaluation.
- **Disk**: Balanced or SSD persistent disk (`pd-balanced` / `pd-ssd`), ext4.

> [!IMPORTANT]
> **Keep images and shards on the local persistent disk, not on GCS FUSE.**
> GCS is right for cold storage and backups, but reading millions of individual
> JPEGs through `gcsfuse` incurs severe random-access latency and per-request
> metadata overhead during embedding extraction.
>
> Recommended flow: keep dataset tarballs in a regional bucket, `gcloud storage
> rsync` chunks down to the local SSD, then run `scripts/embed_images.py` at full
> local read speed.

**GPU install**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu open_clip_torch pandas pyarrow pillow tqdm pyyaml
```

---

## Acknowledgements

- [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
  — Weyand et al., CVPR 2020
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LightGlue](https://github.com/cvg/LightGlue)

Landmark names and category metadata originate from Wikimedia Commons.
