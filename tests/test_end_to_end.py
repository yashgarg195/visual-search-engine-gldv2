"""End-to-End Pipeline Integration Test.

Validates the full workflow from synthetic image creation through
CLIP embedding, IVF-PQ training, addition, search, landmark voting, and evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.clip_encoder import CLIPEncoder
from src.index.flat import FlatIPIndex
from src.index.ivfpq import IVFPQIndex
from src.index.voting import compute_similarity_weighted_votes
from src.metadata.lookup import LandmarkLookupTable
from src.metadata.preparation import prepare_split_metadata
from src.storage.embedding_store import EmbeddingStore
from scripts.generate_synthetic_sample import generate_dataset_sample


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_full_pipeline_end_to_end(tmp_path):
    # Setup test directories
    data_dir = tmp_path / "data"
    emb_dir = tmp_path / "embeddings"
    index_dir = tmp_path / "index"
    processed_dir = tmp_path / "processed"
    eval_dir = tmp_path / "eval"

    for d in [data_dir, emb_dir, index_dir, processed_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic GLDv2 sample
    generate_dataset_sample(data_dir=data_dir, images_per_landmark=6)
    image_dir = data_dir / "raw" / "images"
    meta_dir = data_dir / "raw" / "metadata"

    assert (meta_dir / "train.csv").is_file()
    assert (image_dir / "train").is_dir()

    # 2. Metadata Preparation
    out_parquet = processed_dir / "metadata_train.parquet"
    summary = prepare_split_metadata(
        split="train",
        metadata_dir=meta_dir,
        image_dir=image_dir,
        output_parquet=out_parquet,
        verify_images=True,
    )
    assert summary["total_records"] == 30
    assert summary["existing_images"] == 30
    assert summary["resolved_landmark_names"] == 30
    assert out_parquet.is_file()

    # 3. Model & Embedding Generation (Use offline uninitialized model for fast reproducible test)
    encoder = CLIPEncoder(model_name="ViT-B-32", pretrained="", device="cpu")
    store = EmbeddingStore(emb_dir)
    store.set_model_metadata("ViT-B-32", "openai", encoder.embedding_dim)

    df_meta = pd.read_parquet(out_parquet)
    paths = df_meta["full_path"].tolist()

    embeddings, valid_paths, failed = encoder.encode_image_paths(paths)
    assert len(failed) == 0
    assert len(embeddings) == len(df_meta)
    assert embeddings.shape == (30, encoder.embedding_dim)

    # Verify L2 normalization: ||v||_2 must equal 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    # Write shard
    shard_info = store.write_shard(0, embeddings, df_meta)
    assert shard_info["num_vectors"] == 30
    assert len(store.get_completed_image_ids()) == 30

    # 4. Train IVF-PQ Index
    train_vecs = store.sample_training_vectors(max_samples=30, seed=42)
    # For 30 vectors, use small nlist=4, m=64, nbits=4 (2^4 = 16 < 30)
    index = IVFPQIndex(d=encoder.embedding_dim, nlist=4, m=64, nbits=4, nprobe=2)
    index.train(train_vecs)
    assert index.is_trained

    # 5. Add embeddings with explicit IDs
    faiss_ids = np.arange(100, 130, dtype=np.int64)
    index.add(embeddings, faiss_ids)
    assert index.total_vectors == 30

    saved_index_path = index_dir / "test_populated.index"
    index.save(saved_index_path)
    assert saved_index_path.is_file()

    # 6. Lookup Table & Search
    lookup = LandmarkLookupTable()
    lookup.load_from_parquet(out_parquet)

    # Query with the first training vector
    query_vec = embeddings[0:1]
    scores, ret_ids = index.search(query_vec, top_k=5)

    assert scores.shape == (1, 5)
    assert ret_ids.shape == (1, 5)
    # Vector 100 should be retrieved in the Top-5 approximate matches
    assert 100 in ret_ids[0]

    # Map retrieved matches
    matches = []
    for rank_idx, (fid, sc) in enumerate(zip(ret_ids[0], scores[0])):
        img_id = df_meta.iloc[fid - 100]["image_id"]
        lid = df_meta.iloc[fid - 100]["landmark_id"]
        landmark_info = lookup.lookup_landmark("train", lid)
        matches.append({
            "rank": rank_idx + 1,
            "faiss_id": int(fid),
            "score": float(sc),
            "image_id": img_id,
            "landmark_id": lid,
            "split": "train",
            "landmark_name": landmark_info.landmark_name,
            "is_resolved": landmark_info.is_resolved,
        })

    voting = compute_similarity_weighted_votes(matches)
    assert voting.leading_candidate is not None
    assert voting.leading_candidate.landmark_name == "Eiffel Tower"
    assert voting.leading_candidate.match_count >= 1

    # 7. Exact Baseline IndexFlatIP check
    flat_index = FlatIPIndex(d=encoder.embedding_dim)
    flat_index.add(embeddings, faiss_ids)
    f_scores, f_ids = flat_index.search(query_vec, top_k=5)
    # In exact baseline, query vector matches itself with score ~1.0
    assert f_ids[0][0] == 100
    assert f_scores[0][0] == pytest.approx(1.0, abs=1e-4)
