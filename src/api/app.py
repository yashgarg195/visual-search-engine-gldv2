"""FastAPI REST API & Web Frontend Server for GLDv2 Visual Search Engine."""

import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.service import SearchEngineService
from src.utils.logging import get_logger

logger = get_logger("fastapi_app")

# Global singleton service
search_service = SearchEngineService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize OpenCLIP and FAISS index once on server startup."""
    logger.info("FastAPI starting up: warming up OpenCLIP model and FAISS index...")
    try:
        search_service.initialize()
    except Exception as e:
        logger.error("Failed to initialize search service on startup: %s", e)
        raise
    yield
    logger.info("FastAPI shutting down.")


app = FastAPI(
    title="GLDv2 Visual Landmark Search API",
    description="High-throughput visual and text search across 5.05 million landmarks using OpenCLIP, Multi-Index Fusion, and Gemini Location Intelligence.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for developer flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount rendered keypoint match visualizations directly
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/visualizations", StaticFiles(directory=str(VIS_DIR)), name="visualizations")

# Mount static web assets
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Dataset image directory for serving thumbnails
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", str(PROJECT_ROOT / "data" / "raw" / "images")))


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language prompt describing the landmark")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to retrieve")
    nprobe: int = Field(64, ge=1, le=256, description="Number of IVF Voronoi clusters to probe")
    index_type: str = Field("ivfpq", description="Vector index algorithm: 'ivfpq', 'hnswpq', or 'ensemble'")
    ef_search: int = Field(64, ge=1, le=512, description="Exploration depth for HNSW graph search")
    rerank: bool = Field(False, description="Whether to apply two-stage exact re-ranking")
    alpha_qe: bool = Field(False, description="Whether to apply α-Query Expansion")


@app.get("/")
@app.head("/")
def root():
    """Serve the single-page frontend interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return {
            "message": "GLDv2 Visual Search Engine API is running.",
            "docs": "/docs",
            "info": "/api/info",
        }
    return FileResponse(index_file)


@app.get("/api/info")
def get_system_info():
    """Return runtime metadata, index metrics, and device details."""
    return search_service.get_info()


@app.get("/api/image/{split}/{a}/{b}/{c}/{image_id}.jpg")
@app.head("/api/image/{split}/{a}/{b}/{c}/{image_id}.jpg")
def serve_dataset_image(split: str, a: str, b: str, c: str, image_id: str):
    """Serve a dataset image from the local GLDv2 image directory for thumbnail display."""
    img_path = IMAGE_DIR / split / a / b / c / f"{image_id}.jpg"
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path, media_type="image/jpeg")


@app.post("/api/search/text")
def search_by_text(req: TextSearchRequest):
    """Retrieve nearest landmark images given a natural language text query."""
    if not search_service.is_ready:
        raise HTTPException(status_code=503, detail="Search service is still warming up.")

    try:
        results = search_service.search_text(
            query_text=req.query,
            top_k=req.top_k,
            nprobe=req.nprobe,
            index_type=req.index_type,
            ef_search=req.ef_search,
            rerank=req.rerank,
            alpha_qe=req.alpha_qe,
        )
        return results
    except Exception as e:
        logger.exception("Text search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/image")
def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    nprobe: int = Form(64),
    index_type: str = Form("ivfpq"),
    ef_search: int = Form(64),
    rerank: bool = Form(False),
    alpha_qe: bool = Form(False),
):
    """Upload a query photo to retrieve nearest landmark matches."""
    if not search_service.is_ready:
        raise HTTPException(status_code=503, detail="Search service is still warming up.")

    # Validate image mime type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload an image file (JPEG, PNG, WEBP).",
        )

    try:
        image_bytes = file.file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        results = search_service.search_image(
            image_bytes=image_bytes,
            filename=file.filename or "upload.jpg",
            top_k=top_k,
            nprobe=nprobe,
            index_type=index_type,
            ef_search=ef_search,
            rerank=rerank,
            alpha_qe=alpha_qe,
        )
        return results

    except Exception as e:
        logger.exception("Image search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
