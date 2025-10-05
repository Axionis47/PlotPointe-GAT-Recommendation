#!/usr/bin/env python3
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ServingConfig:
    project_id: str
    item_embeddings_uri: str  # gs://.../item_embeddings.npy
    item_index_map_uri: Optional[str] = None  # gs://.../item_to_idx.json (optional)
    topk: int = 20


class RecommenderRuntime:
    def __init__(self, cfg: ServingConfig):
        self.cfg = cfg
        self.storage = None  # lazy
        self.item_vecs: Optional[np.ndarray] = None
        self.item_to_idx: Optional[dict] = None

    def _gs_download(self, gs_uri: str, local_path: str) -> None:
        if self.storage is None:
            try:
                from google.cloud import storage as gcs_storage
            except Exception as e:
                raise ImportError("google-cloud-storage is required for GCS access") from e
            self.storage = gcs_storage.Client(project=self.cfg.project_id)
        bucket = gs_uri.split('/')[2]
        blob_path = '/'.join(gs_uri.split('/')[3:])
        b = self.storage.bucket(bucket)
        bl = b.blob(blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        bl.download_to_filename(local_path)

    def startup(self) -> None:
        # Download item embeddings
        local = '/tmp/item_embeddings.npy'
        self._gs_download(self.cfg.item_embeddings_uri, local)
        self.item_vecs = np.load(local).astype(np.float32)
        # Optional: item_to_idx mapping for lookups
        if self.cfg.item_index_map_uri:
            mlocal = '/tmp/item_to_idx.json'
            self._gs_download(self.cfg.item_index_map_uri, mlocal)
            with open(mlocal, 'r') as f:
                self.item_to_idx = json.load(f)

    def _ensure_started(self):
        if self.item_vecs is None:
            raise RuntimeError('Runtime not started. Call startup().')

    def top_k_for_user_items(self, item_ids: List[int], k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a simple user vector as mean of provided item vectors, then score against all items via dot product.
        Returns (indices, scores) of top-K items (excluding the provided history if present).
        """
        self._ensure_started()
        assert len(item_ids) > 0, 'Need at least one item id from user history'
        K = k or self.cfg.topk
        item_ids = np.array(item_ids, dtype=np.int64)
        item_vecs = self.item_vecs[item_ids]
        user_vec = item_vecs.mean(axis=0)
        scores = self.item_vecs @ user_vec
        # Mask out history items
        scores[item_ids] = -1e9
        # Top-K
        top_idx = np.argpartition(scores, -K)[-K:]
        # Sort descending within the top-k
        order = np.argsort(scores[top_idx])[::-1]
        top_idx = top_idx[order]
        top_scores = scores[top_idx]
        return top_idx, top_scores

