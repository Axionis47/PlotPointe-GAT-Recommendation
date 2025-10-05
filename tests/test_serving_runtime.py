#!/usr/bin/env python3
import numpy as np
from serving.runtime import RecommenderRuntime, ServingConfig

class _DummyStorageClient:
    def __init__(self):
        pass


def test_top_k_for_user_items_monotonic(monkeypatch, tmp_path):
    # Prepare dummy item embeddings
    item_vecs = np.eye(10, dtype=np.float32)  # 10 items, orthonormal
    np.save(tmp_path / 'item_embeddings.npy', item_vecs)

    # Patch storage client download to copy from tmp_path
    def _fake_download(self, gs_uri: str, local_path: str):
        # ignore gs_uri; just write the test file
        src = tmp_path / 'item_embeddings.npy'
        import shutil
        shutil.copyfile(src, local_path)

    cfg = ServingConfig(project_id='dummy', item_embeddings_uri='gs://bucket/path/item_embeddings.npy', topk=3)
    rt = RecommenderRuntime(cfg)
    # monkeypatch the download method
    monkeypatch.setattr(RecommenderRuntime, '_gs_download', _fake_download)

    rt.startup()

    # Recommend for user with history items [0,1]
    idx, scores = rt.top_k_for_user_items([0, 1], k=3)
    assert len(idx) == 3
    # Best items should not include 0 or 1 and be consistent ordering by score
    assert 0 not in idx and 1 not in idx
    # Scores sorted descending
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

