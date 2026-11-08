"""SOM DRISHTI API.

    .venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import rasterio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.api.routes import router
from app.ingestion.indexer import discover_scenes

app = FastAPI(
    title="SOM DRISHTI",
    description="Lunar image correspondence & registration (SIH26166): SIFT, SuperPoint + LightGlue, LoFTR",
    version="0.2.0",
)

origins = [o.strip() for o in os.environ.get("SOM_DRISHTI_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    index = discover_scenes(config.DATA_ROOT)
    return {
        "status": "ok",
        "dataset_found": index.exists,
        "scene_count": len(index.scenes),
        "versions": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
        },
        "sift_available": hasattr(cv2, "SIFT_create"),
        "usac_magsac_available": hasattr(cv2, "USAC_MAGSAC"),
    }


app.include_router(router)

# Serve the built React app (frontend/dist) from the same process when present.
_dist = config.PROJECT_ROOT / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
