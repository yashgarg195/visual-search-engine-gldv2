# Agent Handoff: Visual Landmark Search Engine (GLDv2)

This document provides a comprehensive operational and architectural handoff for any AI agent or software engineer taking over development, maintenance, or evaluation of this project.

---

## 1. Executive Summary

The **Visual Landmark Search Engine** is a high-performance visual retrieval and landmark identification system indexed over **5,052,992 landmark photos** from the **Google Landmarks Dataset v2 (GLDv2)**.

The system combines:
1. **Dense Vector Search**: Multi-index retrieval supporting **FAISS IVF-PQ** (Inverted File Product Quantization) and a **high-connectivity HNSW-PQ** graph ($M=64, \text{efConstruction}=256$), with **Reciprocal Rank Fusion (RRF)** for hybrid ensemble search.
2. **AI Location Intelligence**: Gemini 3.8 Flash combined with OpenStreetMap Nominatim and a pre-warmed disk cache for automated reverse geocoding, anchor resolution, and landmark summarization.
3. **Stage-2 Hybrid Multi-Signal Fusion Reranking**: Ultra-fast ($< 2\text{ ms}$) fusion combining visual cosine similarity, physical vicinity consensus, and geodesic spatial proximity to resolve split landmark IDs and eliminate geographic false positives.
4. **Interactive Web Dashboard**: Modern single-page application with drag-and-drop / clipboard image search, responsive thumbnail match cards, horizontal reference gallery scrolling, and direct on-page Google Maps iframe pin rendering.

---

## 2. Architecture & Pipeline Flow

```
[User Query: Image / Text]
             │
             ▼
   [OpenCLIP ViT-B-32 Encoder] ──► 512-dim Normalized Vector
             │
             ├─────────────────────────────────────────┐
             ▼                                         ▼
   [FAISS IVF-PQ Index]                       [FAISS HNSW-PQ Index]
   (nlist=4096, nprobe=64)                    (M=64, efSearch=128)
             │                                         │
             └───────────────────┬─────────────────────┘
                                 │
                         (If Ensemble Mode)
                                 ▼
                     [Reciprocal Rank Fusion]
                                 │
                                 ▼
                   [Top-k Candidate Retrievals]
                                 │
                                 ├───────────────────────────────┐
                                 ▼                               ▼
                 [Location Intelligence Resolver]     [GLDv2 Metadata Store]
                 (Gemini + Nominatim + Cache)         (Parquet IDs, Lat/Lon)
                                 │                               │
                                 └───────────────┬───────────────┘
                                                 ▼
                             [Hybrid Multi-Signal Fusion Reranker]
                               FinalScore = 0.50·VisualSim
                                          + 0.30·VicinityConsensus
                                          + 0.20·SpatialProximity
                                                 │
                                                 ▼
                             [Similarity-Weighted Voting Aggregator]
                                                 │
                                                 ▼
                                     [Interactive Web UI]
                                     - Fused Matches with Thumbnails
                                     - On-Page Google Maps Pin
                                     - Horizontal Reference Cards
                                     - Vicinity & Consensus Badges
```

---

## 3. Technology Stack & Specifications

| Component | Technology | Specification / Configuration |
| :--- | :--- | :--- |
| **Vision Backbone** | OpenCLIP (`ViT-B-32`, OpenAI pretrained) | 512-dim embedding, L2-normalized, MPS / CUDA / CPU |
| **Vector DB (Inverted)** | FAISS `IndexIVFPQ` | 5,052,992 vectors, $nlist=4096, m=64, \text{bits}=8$, default $nprobe=64$ |
| **Vector DB (Graph)** | FAISS `IndexHNSWPQ` | 5,052,992 vectors, $M=64, \text{efConstruction}=256, m=64$, default $efSearch=128$ |
| **Ensemble Reranker** | Reciprocal Rank Fusion (RRF) | Constant $k=60$ fusing IVF-PQ and HNSW-PQ ranks |
| **Fusion Reranker** | `HybridMultiSignalReranker` | $w_{\text{vis}}=0.50, w_{\text{vic}}=0.30, w_{\text{geo}}=0.20, \sigma_{\text{spatial}}=5.0\text{ km}$ ($< 2\text{ ms}$ latency) |
| **Location Intel** | Gemini 3.8 Flash + Nominatim | Persistent JSON cache in `outputs/cache/landmark_intel_cache.json` |
| **API Backend** | FastAPI + Uvicorn | Async endpoints (`/api/search/image`, `/api/search/text`, `/api/info`, `/api/image/...`) |
| **Frontend** | Vanilla JS + Modern CSS | No build step required; embedded Google Maps `<iframe>`; thumbnail cards |

---

## 4. Key Directory & File Map

```
visual-search-engine-gldv2/
├── .env                                  # Environment variables (GEMINI_API_KEY, IMAGE_DIR, paths)
├── .gitignore                            # Ignores .index, .npy, .parquet, data/cache, venv
├── requirements.txt                      # Core dependencies (faiss, torch, open_clip, etc.)
├── AGENT_HANDOFF.md                      # This handoff documentation
│
├── src/
│   ├── api/
│   │   ├── app.py                        # FastAPI application routes (search, info, image serving)
│   │   ├── service.py                    # SearchEngineService orchestrating FAISS + Fusion
│   │   └── static/
│   │       ├── index.html                # Single-page UI with Google Maps & metrics
│   │       ├── app.js                    # UI logic, search triggers, match card rendering with thumbnails
│   │       └── style.css                 # Dark-mode dashboard styling with card animations
│   ├── embeddings/
│   │   └── clip_encoder.py               # OpenCLIP ViT-B-32 wrapper with auto-device selection
│   ├── index/
│   │   ├── base.py                       # Abstract base class for FAISS vector indices
│   │   ├── ivfpq.py                      # IVF-PQ index implementation & serialization
│   │   ├── hnsw_pq.py                    # HNSW-PQ graph index implementation
│   │   └── fusion.py                     # Reciprocal Rank Fusion (RRF) implementation
│   ├── llm/
│   │   └── location_resolver.py          # Gemini 3.8 Flash + Nominatim reverse geocoder
│   ├── reranking/
│   │   ├── hybrid_fusion.py              # Multi-Signal Fusion Reranker (Visual + Vicinity + Geo)
│   │   ├── pipeline.py                   # Similarity-weighted voting and candidate grouping
│   │   ├── reference_fetcher.py          # Canonical Wikimedia image retriever
│   │   └── thumbnail_fetcher.py          # Thumbnail proxy & downloader
│   ├── storage/
│   │   └── embedding_store.py            # 494-shard memory-mapped NPY embedding store
│   └── utils/
│       ├── io.py                         # Atomic file save/load utilities
│       └── logging.py                    # Standardized colored console logger
│
├── scripts/
│   ├── search.py                         # CLI search script supporting text & image inputs
│   ├── train_hnsw_pq.py                  # Local / remote HNSW-PQ trainer over shards
│   ├── orchestrate_c3_spot_build.sh      # Automated GCP C3 Spot VM HNSW builder
│   └── vm_build_hnsw.sh                  # In-VM build script run by the orchestrator
│
├── outputs/
│   ├── cache/
│   │   └── landmark_intel_cache.json     # Pre-cached location intelligence (saves LLM quota)
│   ├── embeddings/
│   │   └── embedding_manifest.json       # Manifest indexing 494 embedding shards (10 GB)
│   └── index/
│       ├── ivfpq_populated.index         # Trained IVF-PQ index file (~480 MB)
│       ├── ivfpq_populated.meta.json     # IVF-PQ configuration & vector count
│       ├── hnswpq_populated.index        # Trained HNSW-PQ graph index file (~2.8 GB)
│       ├── hnswpq_populated.meta.json    # HNSW-PQ configuration (M=64, efConstruction=256)
│       └── id_mapping.parquet            # ID mapping table linking FAISS IDs to GLDv2 images
│
└── tests/
    ├── test_hybrid_fusion.py             # Unit tests for multi-signal fusion & Haversine math
    ├── test_hybrid_ensemble.py           # Unit tests for RRF IVF+HNSW fusion
    ├── test_hnsw_pq_index.py             # Unit tests for HNSW graph operations
    ├── test_ivfpq_index.py               # Unit tests for IVF-PQ operations
    ├── test_metadata_resolver.py         # Unit tests for metadata and GLDv2 mapping
    ├── test_geometric_verifier.py        # Unit tests for geometric verification utilities
    ├── test_lightglue_and_alpha_qe.py    # Unit tests for fallback matching & QE logic
    └── test_end_to_end.py                # End-to-end integration test (34 passed tests)
```

---

## 5. Critical Architectural Decisions & Post-Mortems

### A. Removal of SIFT, SuperPoint, and LightGlue
* **The Problem**: Initial attempts to use classical keypoint matching (SIFT) and neural deep feature matching (SuperPoint + LightGlue) for Stage-2 reranking showed poor accuracy on in-the-wild query photos (near 0% confidence).
* **The Root Cause**:
  1. Standard academic keypoint demos (e.g. `I023_CV_Lab_5.ipynb`) match an image against an *affine-transformed synthetic copy of itself* (100% pixel correlation without perspective shift).
  2. Real-world landmark photos exhibit **severe 3D perspective keystoning** (e.g. user looking up from street level vs. high aerial postcard photos), violating 2D planar homography assumptions.
  3. Repetitive architectural textures (identical cathedral windows, skyscraper balconies) generate false cross-matches that fail RANSAC geometric verification.
  4. Execution latency was **1.5 to 3.5 seconds per query** on Apple Silicon MPS / GPU.
* **The Solution**: Replaced geometric keypoint matching with **Hybrid Multi-Signal Fusion**:
  - Blazing fast ($< 2\text{ ms}$ in pure NumPy).
  - Unifies split GLDv2 IDs under physical vicinity consensus.
  - Eliminates geographic false positives via spatial proximity.

### B. High-Connectivity HNSW Graph Construction on GCP C3 Spot VM
* **The Challenge**: Constructing an HNSW graph ($M=64, \text{efConstruction}=256$) over 5.05 million 512-dimensional vectors requires ~30 GB RAM and significant multi-threaded compute. Running on local MacBook caused excessive thermal throttling.
* **The Solution**: Built [`scripts/orchestrate_c3_spot_build.sh`](file:///Users/yashgarg/Documents/visual-search-engine-gldv2/scripts/orchestrate_c3_spot_build.sh):
  - Provisions a Google Cloud **`c3-standard-22`** Spot VM (Sapphire Rapids, 22 vCPUs, 88 GB RAM) for $\sim \$0.25/\text{hr}$.
  - Streams code and 10 GB embedding shards to the instance.
  - Inserts 5.05M vectors across 22 threads in **6.8 minutes** (**12,410 vectors/sec**). Total remote execution: **9.0 minutes**.
  - Syncs the resulting 2.8 GB index back to `outputs/index/hnswpq_populated.index`.
  - Automatically destroys the Spot VM upon completion via `trap cleanup EXIT` (total compute cost $< \$0.12$).

### C. LLM Rate-Limit Protection
* To prevent burning user Gemini API quotas during testing and development:
  - Location intelligence is stored in a 2-tier cache: in-memory and persistent disk cache in `outputs/cache/landmark_intel_cache.json`.
  - Automated tests and search benchmarks use text-retrieval / mock paths that do not hit the live Gemini API.
  - **Standing constraint**: Do not loop against live Gemini API during automated regression tests.

### D. UI/UX Refinement & Front-End Architecture
* **CSS Syntax Correction**: Fixed an unclosed `.ref-gallery-sub` declaration in `style.css` that was risking cascading parse failures for gallery rules.
* **Leaflet Removal & Embedded Google Maps**: Removed legacy Leaflet JS/CSS CDN script tags (~150KB overhead). Google Maps pin is now rendered entirely on-page within an embedded `<iframe>` in `#mapContainer` without external page redirects.
* **Local Dataset Image Serving & Match Card Thumbnails**:
  - Added dedicated FastAPI endpoint `GET /api/image/{split}/{a}/{b}/{c}/{image_id}.jpg` to directly stream local GLDv2 images.
  - Added match card image thumbnails in `app.js` and responsive CSS in `style.css` (`.match-card-inner`, `.match-thumb-wrap`, `.match-thumb`), featuring subtle hover-zoom transitions and graceful `onerror` hiding.
* **Parameter & Metadata Hygiene**:
  - Removed deprecated `alpha_qe: false` request payloads from client search calls.
  - Synchronized vector counts across header pill and HNSW dropdown to accurately display `5.05M` / `5,052,992 Vectors`.
  - Cleaned dead DOM selector `gmapsLink`.
  - Bumped script cache-busting parameter to `app.js?v=2.6`.

---

## 6. How to Run & Verify

### Start the Local Application Server
```bash
# Activate virtual environment
source .venv/bin/activate

# Launch FastAPI daemon with single OpenMP thread (prevents macOS thread thrashing)
OMP_NUM_THREADS="1" uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000`**

### Verify Image Thumbnail Serving
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/image/train/0/0/0/000001.jpg
# Expected: 200
```

### Run CLI Searches
```bash
# Text query using HNSW graph index
python scripts/search.py --query "Saint Basil Cathedral Moscow" --index-type hnswpq --top-k 10

# Image query using IVF-PQ index
python scripts/search.py --image path/to/photo.jpg --index-type ivfpq --top-k 10

# Ensemble hybrid search (IVF + HNSW + RRF)
python scripts/search.py --query "Taj Mahal Agra" --index-type ensemble --top-k 10
```

### Run the Test Suite
```bash
# Run all unit and integration tests (34 tests, all passing)
pytest tests/
```

### Inspect Index Health via API
```bash
curl -s http://127.0.0.1:8000/api/info | jq .
```
Expected output:
```json
{
  "model_name": "ViT-B-32",
  "pretrained": "openai",
  "embedding_dim": 512,
  "device": "mps",
  "total_vectors": 5052992,
  "ivf_vectors": 5052992,
  "hnsw_vectors": 5052992,
  "available_indices": ["ivfpq", "hnswpq", "ensemble"],
  "default_index": "ivfpq",
  "default_nprobe": 64,
  "default_ef_search": 64,
  "default_top_k": 10,
  "is_ready": true
}
```

---

## 7. Recommended Next Steps for Augmentation

When augmenting this codebase with additional datasets:
1. **GSV-Cities**: Add 560k Google Street View urban images across 40 global cities. Exact GPS tags and perspective crops make it an ideal complement for spatial proximity.
2. **Wikimedia Commons Stream**: Use the category hierarchy to pull contemporary high-resolution photos of under-represented landmarks directly into the existing schema.
3. **MegaDepth**: Add dense multi-angle views of top 196 world monuments to bolster recall on extreme perspective queries.
4. **$\mathcal{R}$Oxford / $\mathcal{R}$Paris ($\mathcal{R}$Oxf/$\mathcal{R}$Par)**: Standardize a fixed benchmark script to calculate mAP@k improvements as new embeddings are ingested.
