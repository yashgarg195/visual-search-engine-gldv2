# Agent Handoff: Visual Landmark Search Engine (GLDv2)

This document provides a comprehensive operational and architectural handoff for any AI agent or software engineer taking over development, maintenance, or evaluation of this project.

---

## 1. Executive Summary

The **Visual Landmark Search Engine** is a high-performance visual retrieval and landmark identification system indexed over **5,052,992 landmark photos** from the **Google Landmarks Dataset v2 (GLDv2)**.

The system combines:
1. **Dense Vector Search**: Multi-index retrieval supporting **FAISS IVF-PQ** (Inverted File Product Quantization) and a **high-connectivity HNSW-PQ** graph ($M=64, \text{efConstruction}=256$), with **Reciprocal Rank Fusion (RRF)** for hybrid ensemble search.
2. **AI Location Intelligence**: Gemini 3.8 Flash combined with OpenStreetMap Nominatim and a pre-warmed disk cache for supplemental reverse geocoding, contextual summaries, and canonical reference-image queries. It never changes the identified landmark.
3. **Stage-2 Hybrid Multi-Signal Fusion Reranking**: Ultra-fast ($< 2\text{ ms}$) fusion combining visual cosine similarity, physical vicinity consensus, and geodesic spatial proximity to resolve split landmark IDs and eliminate geographic false positives.
4. **Interactive Web Dashboard**: Modern single-page application with drag-and-drop / clipboard image search, thumbnail match cards, an evidence-style horizontal reference atlas, and a rounded embedded Google Maps card.

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
                [50–100 Candidate Retrievals]
                                 │
                                 ├───────────────────────────────┐
                                 ▼                               ▼
                  [Exact Vector Reranking]       [GLDv2 Metadata Store]
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
                                                 ├───────────────► [Location Intelligence + Reference Images]
                                                 │                 (supplemental only)
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
| **Fusion Reranker** | `HybridMultiSignalReranker` | $w_{\text{vis}}=0.50, w_{\text{vic}}=0.30, w_{\text{geo}}=0.20, \sigma_{\text{spatial}}=5.0\text{ km}$; voting uses its `final_score` |
| **Location Intel** | Gemini 3.8 Flash + Nominatim | Persistent JSON cache in `outputs/cache/landmark_intel_cache.json` |
| **API Backend** | FastAPI + Uvicorn | Async endpoints (`/api/search/image`, `/api/search/text`, `/api/info`, `/api/image/...`) |
| **Frontend** | Vanilla JS + Modern CSS | No build step required; evidence atlas with horizontal scroll controls; rounded embedded Google Maps `<iframe>` |

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
│   │   ├── fusion.py                     # Reciprocal Rank Fusion (RRF) implementation
│   │   └── threading_utils.py            # Scoped OpenMP thread control context managers
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
    └── test_end_to_end.py                # End-to-end integration test (35 passed tests)
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
* **Landmark Lens Redesign**: Modernized UI with Fredoka (display) and Nunito (body) typography, organic biophilic curvature, warm coral/trust blue/cream palette, and persistent light/dark mode with `meta theme-color` synchronization.
* **Vector Icon System**: Standardized on inline SVG icons (`ICONS.pin`, `ICONS.star`, theme toggle icons) to eliminate layout shifts and ensure uniform color theming across light and dark modes.
* **Request Race Protection & Cancellation**: Added `AbortController` and request sequence tracking (`requestSeq`) to automatically cancel in-flight queries and prevent stale async responses from clobbering newer search results.
* **XSS Defense**: Sanitized all user and API-supplied fields with `escapeHtml` and bounded all URLs with `safeHttpUrl` to prevent `javascript:` or `data:` protocol smuggling.
* **Leaflet Removal & Embedded Google Maps**: Removed legacy Leaflet JS/CSS CDN script tags (~150KB overhead). Google Maps pin is rendered on-page within an embedded `<iframe>` in `#mapContainer` without external page redirects.
* **Local Dataset Image Serving & Match Card Thumbnails**:
  - Dedicated FastAPI endpoint `GET /api/image/{split}/{a}/{b}/{c}/{image_id}.jpg` streams local GLDv2 images.
  - Path traversal is strictly prevented via `_SAFE_SEGMENT` regex and `img_path.relative_to(IMAGE_DIR)` verification.
  - Match cards feature lazy-loaded dataset thumbnails with hover-zoom transitions and automatic fallback on missing images.
* **Parameter & Metadata Hygiene**:
  - Removed deprecated `alpha_qe: false` request payloads from client search calls.
  - Synchronized vector counts across header pill and HNSW dropdown to accurately display `5.05M` / `5,052,992 Vectors`.
  - Cleaned dead DOM selector `gmapsLink`.
  - Disabled auto-search on image upload (search triggers exclusively on user click of the search button).
  - Bumped script cache-busting parameter to `app.js?v=3.1`.

### E. Accuracy, Security, and Memory-Safe Runtime (Current)
* **Scoped OpenMP Threading**: Implemented [`src/index/threading_utils.py`](file:///Users/yashgarg/Documents/visual-search-engine-gldv2/src/index/threading_utils.py) context managers (`faiss_threads` and `faiss_single_thread_on_macos`) to prevent global thread contention and permanently pinning process OpenMP state.
* **Security & DoS Protection**:
  - Upload size is capped at 10 MB (`MAX_UPLOAD_BYTES`); oversized payloads are rejected immediately with `413`.
  - Image decoding is guarded against decompression bombs (`MAX_IMAGE_PIXELS = 89_478_485`).
  - 500 error responses no longer leak internal tracebacks to clients; they return a concise error with an 8-character correlation reference ID.
* **Thread-Safe Reranking & Caching**:
  - Mutex locks added to `SearchEngineService._index_lock`, `ThumbnailFetcher._lock`, and `LocationResolver._cache_lock`.
  - Atomic JSON disk persistence via temporary file write and replacement.
  - Image query caching is keyed on retrieved `lm:{landmark_id}` rather than user-supplied filenames to prevent cache poisoning.
* **Exact Reranking Default**: Approximate retrieval expands to `min(100, max(top_k * 5, 50))` candidates, runs exact vector reranking across that pool, and passes them to Multi-Signal Fusion.
* **Lazy HNSW Loading**: Startup loads IVF-PQ only (~3s boot time). HNSW-PQ graph loads on-demand when requested.
* **Memory-Safe Metadata**: Mapping tables are maintained as indexed pandas DataFrames rather than duplicate Python dictionaries.

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

Startup is complete only after the log prints `Application startup complete.` On a local Mac, expect `/api/info` to report `hnsw_vectors: 0` until an HNSW or Ensemble search is requested.

### Verify Image Thumbnail Serving
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/image/train/0/0/0/000001.jpg
# Expected: 200
```

### Run CLI Searches
```bash
# Text query using HNSW graph index (loads the graph on demand)
python scripts/search.py --query "Saint Basil Cathedral Moscow" --index-type hnswpq --top-k 10

# Image query using IVF-PQ index
python scripts/search.py --image path/to/photo.jpg --index-type ivfpq --top-k 10

# Ensemble hybrid search (IVF + HNSW + RRF)
python scripts/search.py --query "Taj Mahal Agra" --index-type ensemble --top-k 10
```

### Run the Test Suite
```bash
# Run all unit and integration tests
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
  "hnsw_vectors": 0,
  "available_indices": ["ivfpq", "hnswpq", "ensemble"],
  "default_index": "ivfpq",
  "default_nprobe": 64,
  "default_ef_search": 64,
  "default_top_k": 10,
  "is_ready": true
}
```

---

## 8. Frontend & UI Architecture Guide (v4.0 Swiss Minimal / MongoDB Style)

The frontend uses a clean Swiss-minimalist / MongoDB-inspired UI with zero gradients, zero box-shadows, and crisp 1px hairlines.

### Key UI File Locations
* **HTML Skeleton**: [`src/api/static/index.html`](file:///Users/yashgarg/Documents/visual-search-engine-gldv2/src/api/static/index.html)
* **Frontend Controller**: [`src/api/static/app.js`](file:///Users/yashgarg/Documents/visual-search-engine-gldv2/src/api/static/app.js)
* **Stylesheet**: [`src/api/static/style.css`](file:///Users/yashgarg/Documents/visual-search-engine-gldv2/src/api/static/style.css)

### Core UI Component Structure
1. **Top Bar & Theme Controls**:
   - Flat header with Brand mark, Theme Toggle (`#themeToggle`, persistent light/dark), and "System" drawer trigger button (`#drawerOpen`).
2. **System Slide-Over Drawer** (`#systemDrawer`):
   - Accessible modal drawer (`role="dialog" aria-modal="true"`) displaying live vector index statistics, device, OpenCLIP encoder, and dataset details fetched via `GET /api/info`.
   - Focus trap and Escape key handler with focus restoration to the trigger button.
3. **Query Section & Progressive Disclosure**:
   - Roving tabindex tablist for Image Search and Text Search (`#tabImageBtn`, `#tabTextBtn`).
   - Image drag-and-drop / clipboard upload with preview (search requires explicit click of "Search landmark").
   - Native `<details>` elements for progressive disclosure: *Search options* (algorithm, top-k, probes/depth), *Execution latency breakdown*, and *Evidence details*.
4. **Location Intelligence & Reference Atlas**:
   - Google Maps iframe pin embedded in a dedicated map container.
   - Reference photo gallery with horizontal carousel navigation controls (`#galleryPrev`, `#galleryNext`).
5. **Paginated Matches Grid & Accessible Navigation**:
   - Paginated display (`MATCHES_PER_PAGE = 6`) with previous/next controls, numbered page buttons, and dynamic announcement live region (`#srStatus`).
   - Cards display rank badge, local thumbnail image (`/api/image/...`), fused score bar, consensus & coordinate-supported tags, and collapsible match metadata details.

### Standing UI Constraints
* **Embedded Maps Only**: The on-page Google Maps `<iframe>` must remain embedded in the card; no external link redirects.
* **Cache-Busting Requirement**: Whenever you edit `app.js` or `style.css`, bump the version parameter in `index.html` (currently `?v=4.0`).
* **Preserve Zero-LLM Test Rule**: Use cached queries or text search rather than triggering live Gemini API repeatedly.
* **Preserve Security Hardening**: All user/API-supplied fields must pass through `escapeHtml()`, and URLs through `safeHttpUrl()`.



---

## 9. Recommended Next Steps for Augmentation

When augmenting this codebase with additional datasets:
1. **GSV-Cities**: Add 560k Google Street View urban images across 40 global cities. Exact GPS tags and perspective crops make it an ideal complement for spatial proximity.
2. **Wikimedia Commons Stream**: Use the category hierarchy to pull contemporary high-resolution photos of under-represented landmarks directly into the existing schema.
3. **MegaDepth**: Add dense multi-angle views of top 196 world monuments to bolster recall on extreme perspective queries.
4. **$\mathcal{R}$Oxford / $\mathcal{R}$Paris ($\mathcal{R}$Oxf/$\mathcal{R}$Par)**: Standardize a fixed benchmark script to calculate mAP@k improvements as new embeddings are ingested.
