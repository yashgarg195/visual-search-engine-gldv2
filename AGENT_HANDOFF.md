# Agent Handoff: Visual Landmark Search Engine (GLDv2)

This document provides a comprehensive operational, architectural, and engineering handoff for any AI agent or software engineer taking over development, maintenance, or evaluation of this project.

---

## 1. Executive Summary

The **Visual Landmark Search Engine** is a production-grade visual retrieval and landmark identification system indexed over **5,052,992 landmark photos** from the **Google Landmarks Dataset v2 (GLDv2)**.

The system combines:
1. **Dense Vector Search**: Multi-index retrieval supporting **FAISS IVF-PQ** (Inverted File Product Quantization) and **FAISS HNSW-PQ** (High-connectivity Hierarchical Navigable Small World, $M=64, \text{efConstruction}=256$), with **Reciprocal Rank Fusion (RRF)** for hybrid ensemble search.
2. **Stage-2 Exact Vector Reranking**: Re-evaluates top vector candidates against unquantized full-precision float32 embeddings from memory-mapped disk shards.
3. **Location Intelligence & Anchor Disambiguation**: Curated zero-latency prominent anchor knowledge base (`PROMINENT_ANCHOR_MAP`) with fallback to **Google Gemini** (`gemini-3.1-flash-lite`) and OpenStreetMap Nominatim for geographic attribution, historical summaries, and canonical reference images.
4. **Hybrid Multi-Signal Fusion**: Fuses visual cosine similarity, physical vicinity consensus, and geodesic spatial proximity into unified candidate scoring.
5. **Similarity-Weighted Voting Aggregation**: Consolidates multiple matches per landmark ID across training and index splits to determine the lead winning landmark.
6. **Modern Web Interface (v4.0)**: Swiss-minimalist responsive UI featuring drag-and-drop staging, manual search execution, paginated candidate grids, an embedded Google Maps location panel, and a sliding inspection drawer.

---

## 2. End-to-End Pipeline Flow

```
[User Query: Uploaded Image or Text Prompt]
             │
             ▼
   [OpenCLIP ViT-B-32 Encoder] ──► 512-dim L2-Normalized Vector
             │
             ├─────────────────────────────────────────┐
             ▼                                         ▼
   [FAISS IVF-PQ Index]                       [FAISS HNSW-PQ Index]
   (nlist=4096, nprobe=64)                    (M=64, efSearch=64–128)
             │                                         │
             └───────────────────┬─────────────────────┘
                                 │
                         (If Ensemble Mode)
                                 ▼
                     [Reciprocal Rank Fusion]
                                 │
                                 ▼
                    [Top 50–100 Candidates]
                                 │
                                 ▼
                    [Exact Float32 Reranking]
                                 │
                                 ▼
             [GLDv2 Metadata Store & Landmark Lookup]
                                 │
                                 ▼
             [Location Intelligence & Anchor Resolver]
             - Prominent Anchor Match (e.g. Saint Basil, Christ the Redeemer)
             - Gemini / Nominatim Supplemental Geocoding
             - Wikimedia Commons Canonical Reference Fetcher
                                 │
                                 ▼
             [Hybrid Multi-Signal Fusion Reranker]
             FinalScore = 0.50·VisualSim + 0.30·Vicinity + 0.20·Spatial
             (Generic building types excluded from vicinity clustering)
                                 │
                                 ▼
             [Similarity-Weighted Voting Aggregator]
             - Aggregates multi-image evidence per Landmark ID
             - Maps display_name = prominent_anchor
             - Preserves specific_landmark = dataset sub-entity
                                 │
                                 ▼
                        [FastAPI JSON Response]
                                 │
                                 ▼
                        [Frontend Dashboard]
                        - Identified Landmark Banner
                        - Location Intel & Embedded Map
                        - Canonical Photo Gallery
                        - Paginated Candidate Cards & Detail Drawer
```

---

## 3. Technology Stack & Specifications

| Component | Technology | Specification / Configuration |
| :--- | :--- | :--- |
| **Vision Backbone** | OpenCLIP (`ViT-B-32`, OpenAI weights) | 512-dim embedding, L2-normalized, Apple Silicon MPS / CUDA / CPU |
| **Vector DB (Inverted)** | FAISS `IndexIVFPQ` | 5,052,992 vectors, $nlist=4096, m=64, \text{bits}=8$, default $nprobe=64$ |
| **Vector DB (Graph)** | FAISS `IndexHNSWPQ` | 5,052,992 vectors, $M=64, \text{efConstruction}=256, m=64$, default $efSearch=64$ |
| **Ensemble Retrieval** | Reciprocal Rank Fusion (RRF) | Constant $k=60$ fusing IVF-PQ and HNSW-PQ ranks |
| **Exact Reranker** | Memory-mapped Float32 Shards | Two-stage candidate reranking against unquantized vectors |
| **Location Intelligence** | Gemini Flash + OSM Nominatim | Primary: `gemini-3.1-flash-lite`, timeout 5.5s; curated `PROMINENT_ANCHOR_MAP` |
| **Reference Fetcher** | Wikimedia Commons REST API | Canonical, high-resolution unmetered reference photos |
| **API Backend** | FastAPI + Uvicorn | High-concurrency async endpoints on port 8000 |
| **Web Frontend** | Vanilla JS + Modern Swiss CSS | Zero frontend dependencies/bundlers; slide-over drawer; pagination |

---

## 4. Key Architectural Decisions & Critical Fixes

### A. Prominent Landmark Identity Resolution
- **Problem**: In GLDv2, photos of major monuments are often catalogued under adjacent sub-entities (e.g., Saint Basil's Cathedral images are catalogued under `Monument to Minin and Pozharsky in Moscow`, and Christ the Redeemer under `Corcovado` or `Cristo Rei`). In a previous refactor, `display_name = prominent_anchor` was removed, causing the UI to display confusing sub-landmark titles.
- **Solution**: Restored in `src/api/service.py`:
  ```python
  display_name = prominent_anchor if (prominent_anchor and location_info) else lead.landmark_name
  voting_summary = {
      "landmark_name": display_name,
      "specific_landmark": lead.landmark_name,
      ...
  }
  ```
  The primary card displays the world-renowned landmark (e.g. *Saint Basil's Cathedral*, *Christ the Redeemer*), while retaining the specific dataset sub-entity in `specific_landmark`.

### B. Architectural Typology vs. Geographic Vicinity Clustering
- **Problem**: In `src/reranking/hybrid_fusion.py`, `_extract_vicinity_key()` previously split `hierarchical_label` on `>` and treated the first item as a geographic region. In GLDv2, `hierarchical_label` is an architectural category (e.g. `"church"`, `"tower"`, `"sculpture"`, `"mountain"`). This clustered every church on Earth into a single vicinity, giving random churches an artificial 50% score boost over the actual landmark.
- **Solution**: Defined `generic_typologies` in `src/reranking/hybrid_fusion.py`. Single-word building classifications are strictly excluded from vicinity clustering; clustering requires multi-level geographic hierarchies (`City > Neighborhood`) or prominent anchor matches.

### C. Prevention of Secondary Candidate Identity Leaks
- **Problem**: `LocationResolver.resolve()` previously scanned `candidates[:4]` against `PROMINENT_ANCHOR_MAP`. If candidate #1 was Church of the Saviour on the Blood and candidate #3 was Minin & Pozharsky, the anchor for candidate #3 hijacked candidate #1's resolution and cached it under `lm:190216`.
- **Solution**: In `src/llm/location_resolver.py`, anchor matching is strictly scoped to the query text or the lead candidate (`candidates[0]`). Secondary candidates cannot leak their identity into the winner.

### D. Cache Sanitization & Fallback Protection
- **Problem**: Transient timeouts or 503 errors previously triggered `_heuristic_fallback()`, which immediately wrote low-quality strings (e.g. *"located in tower"*, *"is a notable landmark"*) into `landmark_intel_cache.json`, permanently poisoning future searches.
- **Solution**:
  1. `_heuristic_fallback()` results are **never** persisted to the disk cache.
  2. Implemented `_is_degraded_cache_entry()` to detect and purge legacy placeholder strings dynamically.
  3. Pre-populated `PROMINENT_ANCHOR_MAP` with authoritative entries for top global landmarks:
     - *Gateway of India* (Mumbai, India)
     - *Eiffel Tower* (Paris, France)
     - *Saint Basil's Cathedral* (Moscow, Russia)
     - *Christ the Redeemer* (Rio de Janeiro, Brazil)
     - *Taj Mahal* (Agra, India)
     - *Colosseum* (Rome, Italy)
     - *Space Needle* (Seattle, USA)
     - *Church of the Savior on Spilled Blood* (Saint Petersburg, Russia)
     - *Grand Canyon* (Arizona, USA)

### E. Search Button Execution (No Auto-Search on Upload)
- **Problem**: Dragging or uploading an image immediately fired a search request, removing user control over parameter tuning (top_k, nprobe, index algorithm).
- **Solution**: In `src/api/static/app.js`, image upload stages the file and renders a thumbnail preview with a "Ready to search" pill. Execution only triggers when the user explicitly clicks the **Search** button.

### F. Playful Colour & Typography Re-skin (v5.0)
- **Palette**: Violet accent (`#6d28d9`), warm-cream background (`#fffcf8`), with semantic accents: coral (`#c91a52`, top-rank badge / marker), teal (`#0a7268`, score bars / confidence), amber (chip hover). In dark mode: deep indigo (`#1a1430`) with high-contrast light lavender action fills (`#b9a0ff`, 8.04:1 contrast).
- **Typography**: Display headings use **Fredoka**; body text uses **Nunito**; numerics and coordinates retain **JetBrains Mono** for fixed-width tabular alignment.
- **Geometry**: Border radii softened from 6px to 12px; layout is byte-for-byte identical with zero gradients, zero box-shadows, and zero backdrop-filter. Passes 33/33 WCAG AA contrast pairs.

### G. Candidate Pool Decoupling (`candidate_k=1000`) & Vectorized Reranking
- **Problem**: Previously `search_k = min(100, max(top_k * 5, 50))` was overloaded for both ANN search depth and downstream result size. Capping candidates at 100 artificially limited Recall@100 to only ~54%.
- **Solution**:
  1. Decoupled ANN search depth (`candidate_k = 1000`, env-tunable via `RERANK_CANDIDATE_K`) from downstream result size (`result_k = 100`).
  2. Vectorized `TwoStageReranker.rerank_candidates()` in `src/index/rerank.py` using batched pandas reindexing, grouping by shard, and fancy-indexed matrix-vector multiplication (`arr[row_indices] @ query_vec`).
  3. Increased reranker throughput by **2.4x–2.5x**, scoring 1,000 unquantized float32 vectors in ~140ms while keeping downstream metadata lookup, fusion, and voting fixed at 100 candidates.

---

## 5. Retrieval Sweep & Benchmark Analysis

Evaluated over all 5,052,992 vectors with exact ground truth computed across all 494 shards:

### IVF-PQ Sweep (M=64, nbits=8, nlist=4096)
| nprobe | candidate_k | Recall@10 vs Exact | Recall@100 vs Exact | p50 Latency (ms) | p95 Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | 100 | 85.20% | 51.32% | 0.65 ms | 1.13 ms |
| 8 | 500 | 88.80% | 77.60% | 0.66 ms | 0.85 ms |
| 8 | 1000 | 89.00% | 80.70% | 0.67 ms | 0.92 ms |
| 16 | 100 | 90.60% | 53.58% | 0.81 ms | 1.06 ms |
| 16 | 500 | 95.20% | 84.72% | 0.88 ms | 1.23 ms |
| 16 | 1000 | 95.80% | 89.56% | 0.91 ms | 1.20 ms |
| 32 | 100 | 92.60% | 54.22% | 1.25 ms | 1.82 ms |
| 32 | 500 | 97.80% | 87.66% | 1.23 ms | 1.64 ms |
| 32 | 1000 | 98.80% | 93.66% | 1.30 ms | 1.81 ms |
| **64** *(prod)* | 100 | 92.80% | 54.44% | 2.03 ms | 2.62 ms |
| **64** *(prod)* | 500 | 98.40% | 88.76% | 2.03 ms | 2.62 ms |
| **64** *(prod)* | **1000** | **99.40%** | **95.42%** | **2.09 ms** | **2.73 ms** |
| 128 | 1000 | 99.60% | 96.02% | 3.53 ms | 4.45 ms |
| 256 | 1000 | 99.60% | 96.24% | 6.55 ms | 7.90 ms |

### HNSW-PQ Sweep (M=64, efConstruction=256)
| efSearch | candidate_k | Recall@10 vs Exact | Recall@100 vs Exact | p50 Latency (ms) | p95 Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64 | 100 | 98.40% | 66.18% | 0.63 ms | 0.89 ms |
| 64 | 1000 | 98.60% | 88.42% | 0.53 ms | 0.78 ms |
| 128 | 100 | 99.20% | 66.74% | 0.77 ms | 1.16 ms |
| 128 | 500 | 99.60% | 94.60% | 0.78 ms | 1.08 ms |
| **128** *(prod)* | **1000** | **99.60%** | **95.28%** | **0.81 ms** | **1.15 ms** |
| **256** | **1000** | **100.00%** | **98.22%** | **1.26 ms** | **1.80 ms** |
| 512 | 1000 | 100.00% | 99.32% | 2.07 ms | 2.94 ms |

### Key Takeaway on Retrieval Health
- At `nprobe=64, candidate_k=1000`, IVF-PQ achieves **95.42% Recall@100 vs Exact** at only **2.09 ms**.
- At `efSearch=128, candidate_k=1000`, HNSW-PQ achieves **95.28% Recall@100 vs Exact** at **0.81 ms** (and **98.22%** at `efSearch=256`).
- **Conclusion**: Because ANN recall against exact search is $\ge 95\%$, the vector index is exceptionally healthy. Vector quantization causes negligible information loss; any remaining visual retrieval ambiguities stem from the CLIP ViT-B-32 vision backbone, confirming that index tuning is complete.

---

## 6. Repository Structure

```
visual-search-engine-gldv2/
├── data/
│   ├── raw/                       # Original GLDv2 CSVs (train.csv, index.csv)
│   └── processed/                 # Parquet metadata tables (train, index)
├── outputs/
│   ├── cache/
│   │   └── landmark_intel_cache.json  # Curated landmark intelligence cache
│   ├── embeddings/                # 494 float32 embedding shards (.npy + .parquet)
│   └── index/
│       ├── ivfpq_populated.index  # FAISS IVF-PQ index (5.05M vectors)
│       ├── hnswpq_populated.index # FAISS HNSW-PQ index (5.05M vectors, M=64)
│       └── id_mapping.parquet     # FAISS internal ID to GLDv2 image/landmark ID
├── scripts/
│   ├── orchestrate_c3_spot_build.sh # GCP Spot VM HNSW builder
│   ├── populate_hnsw_pq.py        # Multi-threaded HNSW graph population
│   └── search.py                  # CLI search tool
├── src/
│   ├── api/
│   │   ├── app.py                 # FastAPI route definitions, CORS, rate limits
│   │   ├── service.py             # Global singleton SearchEngineService
│   │   └── static/                # Single-page frontend (index.html, app.js, style.css)
│   ├── index/
│   │   ├── ivfpq.py               # IVFPQIndex wrapper
│   │   ├── hnsw_pq.py             # HNSWPQIndex wrapper
│   │   ├── rerank.py              # TwoStageReranker (float32 exact rerank)
│   │   └── voting.py              # compute_similarity_weighted_votes()
│   ├── llm/
│   │   └── location_resolver.py   # Curated anchors + Gemini 3.1-flash-lite fallback
│   ├── models/
│   │   └── clip_encoder.py        # OpenCLIP ViT-B-32 image & text feature extraction
│   └── reranking/
│       ├── hybrid_fusion.py       # HybridMultiSignalReranker (visual + vicinity + spatial)
│       └── reference_fetcher.py   # Canonical Wikimedia Commons photo retriever
├── tests/                         # 35 pytest unit & end-to-end integration tests
└── requirements.txt
```

---

## 6. Development & Run Guide

### Environment Setup
```bash
# Virtual environment with Python 3.9 / 3.10
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Starting the Server
```bash
# Set OpenMP thread count to 1 to avoid thread contention on macOS MPS
OMP_NUM_THREADS="1" .venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
- Web UI: `http://127.0.0.1:8000/`
- API Documentation (Swagger): `http://127.0.0.1:8000/docs`
- System Info: `http://127.0.0.1:8000/api/info`

### Running the Test Suite
```bash
.venv/bin/pytest tests/
```
All 35 unit tests should pass with 0 failures:
- `test_embedding_store.py`
- `test_end_to_end.py`
- `test_geometric_verifier.py`
- `test_hnsw_pq_index.py`
- `test_hybrid_ensemble.py`
- `test_hybrid_fusion.py`
- `test_ivfpq_index.py`
- `test_lightglue_and_alpha_qe.py`
- `test_metadata_resolver.py`

---

## 7. Operational Commitments & Constraints
1. **Never Cache Degraded Heuristics**: Only verified LLM responses or curated anchors may be stored in `landmark_intel_cache.json`.
2. **Never Trigger Repeated Automated LLM Loops**: Testing should utilize mocked responses or the curated anchor map to avoid rate limit exhaustion.
3. **Keep Ground Truth Distinct from Anchors**: Always preserve `specific_landmark` (raw GLDv2 entity) alongside `landmark_name` (prominent anchor) in voting results.
4. **Scoped OpenMP Threading**: Multi-threaded FAISS searches must be synchronized with `_index_lock` to prevent concurrency crashes in async FastAPI worker pools.
