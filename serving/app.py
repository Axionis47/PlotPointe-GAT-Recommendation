#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from serving.runtime import ServingConfig, RecommenderRuntime


app = FastAPI(title="PlotPointe Recommender (baseline)")
runtime: Optional[RecommenderRuntime] = None


class StartupConfig(BaseModel):
    project_id: str
    item_embeddings_uri: str
    item_index_map_uri: Optional[str] = None
    topk: int = 20


class RecommendRequest(BaseModel):
    item_ids: List[int]
    k: Optional[int] = None


@app.on_event("startup")
def on_startup():
    # Environment-driven startup for Cloud Run
    project_id = os.getenv("PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "plotpointe"))
    item_embeddings_uri = os.getenv("ITEM_EMBEDDINGS_URI")
    item_index_map_uri = os.getenv("ITEM_INDEX_MAP_URI")
    topk = int(os.getenv("TOPK", "20"))

    if not item_embeddings_uri:
        # Delay startup if env not set; app will expose a manual /startup endpoint too
        return

    global runtime
    cfg = ServingConfig(
        project_id=project_id,
        item_embeddings_uri=item_embeddings_uri,
        item_index_map_uri=item_index_map_uri,
        topk=topk,
    )
    runtime = RecommenderRuntime(cfg)
    runtime.startup()


@app.post("/startup")
def manual_start(cfg: StartupConfig):
    global runtime
    runtime = RecommenderRuntime(ServingConfig(**cfg.dict()))
    runtime.startup()
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /startup or set env.")
    if not req.item_ids:
        raise HTTPException(status_code=400, detail="item_ids required")
    idx, scores = runtime.top_k_for_user_items(req.item_ids, k=req.k)
    return {"indices": idx.tolist(), "scores": [float(x) for x in scores.tolist()]}

