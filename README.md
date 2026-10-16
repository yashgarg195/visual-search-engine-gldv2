# Visual Landmark Search Pipeline (GLDv2 + CLIP + FAISS IVF-PQ)

A production-grade, reproducible, script-based visual landmark retrieval and recognition pipeline built with **OpenCLIP** (ViT-B/32) and **FAISS IVF-PQ** (Inverted File Product Quantization) on the **Google Landmarks Dataset v2 (GLDv2)**. Designed for high-throughput scaling to millions of landmark vectors on Linux VMs on Google Cloud, while supporting rapid local development.

---

## 1. System Architecture

```
                                            [ Query Image / Text ]
                                                      │
                                           CLIP Image/Text Encoder
                                                      │
                                           L2 Normalized Vector
                                                      │
                                                      ▼
[ GLDv2 Images ] ──► [ CLIP ViT-B/32 ] ──► [ Persistent Shards ] ──► [ FAISS IVF-PQ Index ]
                             │                      │                           │
                   L2-Normalized Float32      Embedding Array (.npy)       Coarse Quantizer (FlatIP)
                       Unit Vectors          Companion Meta (.parquet)     Product Quantizer (PQ)
                                             Global Manifest (.json)       Explicit ID Mapping (int64)
                                                                                │
                                                                                ▼
                                                                     [ Top-K Nearest Matches ]
                                                                                │
                                                                   [ Landmark Lookup Table ]
                                                                       Category URL Decoded
                                                                       Hierarchical Labels
                                                                                │
                                                                                ▼
                                                                   [ Similarity-Weighted Voting ]
                                                                   Leading Landmark Candidate
```

### Retrieval & Indexing Pipeline:
1. **Metadata Preparation**: Parses official GLDv2 CSV files (`train`, `index`), checks duplicate IDs, preserves string IDs with leading zeros, resolves sharded directory paths (`{split}/{c0}/{c1}/{c2}/{id}.jpg`), and extracts readable landmark names from Wikimedia Commons category URLs.
2. **Feature Extraction**: Encodes images into 512-dimensional L2-normalized float32 vectors using OpenCLIP ViT-B/32 in batches.
3. **Atomic Sharding & Checkpointing**: Persists vectors and companion metadata into atomic shards (`shard_XXXXX_emb.npy` and `shard_XXXXX_meta.parquet`) with a global `embedding_manifest.json` for safe resumption without duplicates.
4. **FAISS IVF-PQ Indexing**:
   - Quantizer: `IndexFlatIP` (Inner product = cosine similarity for normalized vectors).
   - Inverted File Product Quantization (`IndexIVFPQ`): Quantizes 512-dim vectors into `m=64` byte codes.
   - Explicit 64-bit ID Mapping (`IndexIDMap2`): Ensures permanent 1-to-1 mapping between FAISS integer IDs and string `image_id`.
5. **Similarity-Weighted Voting**: Aggregates retrieval scores across candidate landmarks to identify the leading landmark candidate.
6. **Optional Two-Stage Reranking**: Queries candidate pool with IVF-PQ, loads candidate vectors on demand from memory-mapped shards, and reranks by exact cosine similarity.
7. **ANN & Recognition Benchmark**: Evaluates Recall@1, Recall@5, Recall@10 against an exact `IndexFlatIP` baseline, along with Top-1 / Top-5 landmark recognition accuracy and latency sweeps.

---

## 2. Scalability & Storage Calculations

For **512-dimensional float32** embeddings:
- **Raw Float32 Vector**: $512 \times 4\text{ bytes} = 2,048\text{ bytes} \approx 2.0\text{ KB/vector}$.
- **Quantized IVF-PQ Vector** ($m=64, \text{nbits}=8$): $64\text{ bytes} + 8\text{ bytes (ID)} = 72\text{ bytes/vector}$.
- **Compression Ratio**: $\approx 28.4\times$ memory reduction!

| Vector Count | Raw Float32 Vectors | IVF-PQ Index Size | Approximate RAM Required |
|:---|:---|:---|:---|
| **100,000 (100K)** | ~200 MB | ~7.2 MB | < 250 MB |
| **1,000,000 (1M)** | ~2.0 GB | ~72.0 MB | < 1.0 GB |
| **5,000,000 (5M)** | ~10.0 GB | ~360.0 MB | < 3.0 GB |

### IVF-PQ Hyperparameter Design:
- **`nlist` (Number of Voronoi cells / coarse centroids)**:
  - Partitions the embedding space. Rule of thumb: $4\sqrt{N}$ to $16\sqrt{N}$ (e.g. for 1M vectors, $nlist=4,096$ to $16,384$).
  - For small testing sets ($N < 1,000$), use $nlist=8$ to $32$ so each cell has sufficient training points.
- **`nprobe` (Clusters searched at query time)**:
  - Controls the recall vs. latency trade-off. Searching $nprobe=1$ is fastest; $nprobe=16$ or $32$ typically recovers 85-98% of exact nearest neighbors.
- **`m` (Number of PQ subquantizers)**:
  - Must divide embedding dimension $d$ ($512 \pmod m == 0$). For $d=512$, $m=64$ creates 8-dimensional sub-vectors.
- **`nbits` (Bits per subquantizer)**:
  - Typically $nbits=8$ ($2^8 = 256$ centroids per subquantizer codebook).

---

## 3. Directory Layout

```
visual-search-engine-gldv2/
├── .env.example                       # Configuration template
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development & test dependencies
├── pytest.ini                         # PyTest configuration
├── README.md                          # Comprehensive documentation
├── src/
│   ├── config.py                      # Master typed dataclass configuration
│   ├── utils/
│   │   ├── hardware.py                # Hardware, RAM, GPU, disk, and dependency checker
│   │   ├── io.py                      # Atomic JSON, Parquet, and NumPy file persistence
│   │   └── logging.py                 # Structured console and file logger
│   ├── metadata/
│   │   ├── resolver.py                # Wikimedia URL decoding & hierarchy resolution
│   │   ├── preparation.py             # Official CSV ingestion & path validation
│   │   └── lookup.py                  # Split-aware O(1) landmark & coordinate lookups
│   ├── models/
│   │   └── clip_encoder.py            # OpenCLIP ViT-B/32 loader, normalization, batching
│   ├── storage/
│   │   └── embedding_store.py         # Sharded storage, manifests, and memory-mapped reader
│   └── index/
│       ├── base.py                    # BaseVectorIndex interface (modular for future HNSW)
│       ├── ivfpq.py                   # FAISS IndexIVFPQ + IndexIDMap2 implementation
│       ├── flat.py                    # Exact IndexFlatIP baseline
│       ├── rerank.py                  # Two-stage memory-mapped candidate reranker
│       └── voting.py                  # Similarity-weighted voting across Top-K matches
├── scripts/
│   ├── check_environment.py           # Environment and hardware inspection CLI
│   ├── prepare_metadata.py            # CSV parsing and Parquet generation CLI
│   ├── embed_images.py                # Resumable sharded embedding extraction CLI
│   ├── train_ivfpq.py                 # IVF-PQ parameter validation & training CLI
│   ├── add_embeddings.py              # Batch shard ingestion into FAISS index CLI
│   ├── search.py                      # Visual query CLI with voting & landmark lookup
│   ├── evaluate.py                    # ANN recall, recognition accuracy, and latency CLI
│   ├── lookup_landmark.py             # Standalone landmark metadata resolution CLI
│   ├── generate_synthetic_sample.py   # Realistic offline sample generator
│   ├── run_small_pipeline.sh          # End-to-end local test pipeline runner
│   ├── run_embedding_pipeline.sh      # Resumable large-scale embedding runner
│   └── run_index_pipeline.sh          # Index build and evaluation runner
└── tests/
    ├── test_metadata_resolver.py      # Unit tests for URL decoding & split preservation
    ├── test_embedding_store.py        # Unit tests for atomic shards & resumption
    ├── test_ivfpq_index.py            # Unit tests for parameter checks & search
    └── test_end_to_end.py             # Complete end-to-end integration test
```

---

## 4. Setup & Installation

### Option A: Setup on Linux VM (Google Cloud)
```bash
# 1. Update OS packages and install Python 3 venv & build tools
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip libopenblas-dev

# 2. Clone repository & create virtual environment
cd /opt/visual-search-engine-gldv2
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install dependencies
# For CPU-only machine:
pip install -r requirements.txt

# For CUDA GPU-enabled machine (e.g. L4, T4, A100):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install faiss-gpu open_clip_torch pandas pyarrow pillow tqdm pyyaml pytest
```

### Option B: Local Testing (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

---

## 5. Quickstart & Verification

Run the automated end-to-end runner on the synthetic GLDv2 sample:
```bash
./scripts/run_small_pipeline.sh none
```
This executes all stages:
1. Environment and hardware audit.
2. Synthetic GLDv2 sample generation (`data/raw/images` and `data/raw/metadata`).
3. Metadata validation and Parquet generation.
4. CLIP embedding generation in persistent shards.
5. FAISS IVF-PQ index training ($d=512, nlist=8, m=64, nbits=8$).
6. Ingesting embeddings into FAISS index with explicit IDs.
7. Visual query demonstration on `data/sample_queries/query_eiffel_tower.jpg` with similarity-weighted voting.
8. Evaluation against exact `IndexFlatIP` baseline.

---

## 6. Running Individual Pipeline Steps

### 1. Environment Audit
```bash
python3 scripts/check_environment.py
```

### 2. Prepare GLDv2 Metadata
```bash
python3 scripts/prepare_metadata.py --splits train index --verify-images
# To auto-download official GLDv2 CSV files if missing:
python3 scripts/prepare_metadata.py --splits train index --download-metadata
```

### 3. Extract CLIP Embeddings (Resumable)
```bash
python3 scripts/embed_images.py \
    --split train \
    --batch-size 64 \
    --shard-size 10000 \
    --device auto
```

### 4. Train FAISS IVF-PQ Index
```bash
python3 scripts/train_ivfpq.py \
    --nlist 1024 \
    --m 64 \
    --nbits 8 \
    --sample-size 50000
```

### 5. Ingest Embedding Shards into Index
```bash
python3 scripts/add_embeddings.py
```

### 6. Visual Query Search with Landmark Resolution & Voting
```bash
python3 scripts/search.py \
    --query-image data/sample_queries/query_eiffel_tower.jpg \
    --top-k 10 \
    --nprobe 16
```
Sample Search & Voting Output:
```
==========================================================================================
 VISUAL LANDMARK SEARCH RESULTS (Image: query_eiffel_tower.jpg)
 IVF-PQ nprobe: 8 | Latency: 4.88 ms search (125.12 ms embedding)
==========================================================================================

Rank  Score    Landmark Name                    Landmark ID    Image ID       Split
------------------------------------------------------------------------------------------
1     0.0890   Eiffel Tower                     1001           00002d         train
2     0.0886   Eiffel Tower                     1001           00001e         train
3     0.0884   Eiffel Tower                     1001           00003c         train
4     0.0882   Eiffel Tower                     1001           00000f         train
5     0.0878   Eiffel Tower                     1001           00001d         train

------------------------------------------------------------------------------------------
 SIMILARITY-WEIGHTED VOTING IDENTIFICATION
------------------------------------------------------------------------------------------
  Leading Candidate   : Eiffel Tower (Split: train, ID: 1001)
  Total Vote Score    : 0.4409 (from 5 supporting image matches)
  Confidence Margin   : 0.4409 over runner-up
  Category Source     : http://commons.wikimedia.org/wiki/Category:Eiffel_Tower
  Hierarchy           : monument_tower
  Supporting Images   : 00002d, 00001e, 00003c, 00000f, 00001d

  NOTE: Similarity scores represent cosine similarity in [-1, 1] for normalized embeddings,
        not calibrated statistical probabilities.
==========================================================================================
```

### 7. Benchmark Evaluation (Recall, Accuracy, Latency Sweep)
```bash
python3 scripts/evaluate.py \
    --num-queries 100 \
    --nprobe-sweep 1 4 8 16 32 64
```

### 8. Landmark Metadata Lookup CLI
```bash
python3 scripts/lookup_landmark.py --landmark-id 1001 --split train
python3 scripts/lookup_landmark.py --category-url "http://commons.wikimedia.org/wiki/Category:%C3%89cole_Militaire"
```

---

## 7. Running Tests

Run the complete test suite:
```bash
pytest tests/ -v
```
All 14 unit and integration tests validate URL decoding, split preservation, atomic shard writes, index parameter bounds, similarity-weighted voting, and end-to-end integration.

---

## 8. Deployment on Google Cloud Platform (GCP)

### Recommended GCP Architecture:
- **Compute Engine VM**: `g2-standard-8` (1 NVIDIA L4 GPU, 8 vCPUs, 32 GB RAM) or `n1-standard-8` (for CPU-only evaluation).
- **Persistent Disk**: Attach a **Balanced Persistent SSD (`pd-balanced` or `pd-ssd`)** formatted with `ext4`.
  > [!IMPORTANT]
  > **Storage Best Practice**: Keep active images and embedding shards on the local Persistent SSD rather than mounting Google Cloud Storage via Cloud Storage FUSE (`gcsfuse`). While GCS is ideal for cold storage and backups, reading millions of individual JPEG image files over Cloud Storage FUSE incurs severe random-access network latency and per-request metadata overhead during embedding extraction.
  >
  > **Recommended Workflow**:
  > 1. Store dataset tar archives in a regional GCS bucket (`gs://your-gldv2-bucket/`).
  > 2. Use `gcloud storage rsync` or `gsutil -m cp` to copy chunks to the local persistent SSD.
  > 3. Run `scripts/embed_images.py` at full NVMe SSD read speeds.
