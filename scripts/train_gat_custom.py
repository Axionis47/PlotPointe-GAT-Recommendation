#!/usr/bin/env python3
"""
Custom GAT (pure PyTorch) for PlotPointe user–item graph with BPR loss.
- Users: learned embeddings; Items: fused 128d features projected to model dim
- Train with one backward per epoch (large sampled BPR set) to avoid autograd graph reuse
- Inputs from GCS; outputs checkpoints and metrics to GCS
"""
import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from google.cloud import storage


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    project_id: str
    region: str
    staging_prefix: str
    graphs_prefix: str
    embeddings_prefix: str
    models_prefix: str
    hidden_dim: int = 128
    layers: int = 2
    attn_dropout: float = 0.1
    lr: float = 1e-3
    l2: float = 1e-4
    epochs: int = 20
    samples_per_epoch: int = 200_000
    seed: int = 42
    eval_neg_k: int = 1000
    item_features: str = "fused"  # {fused, txt}
    loss: str = "bpr"            # {bpr, bce}


class SimpleGATLayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, attn_dropout: float = 0.1):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = torch.nn.Parameter(torch.empty(out_dim))
        self.a_dst = torch.nn.Parameter(torch.empty(out_dim))
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        torch.nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = torch.nn.LeakyReLU(0.2)
        self.drop = torch.nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, F], edge_index: [2, E] with src=row(0), dst=row(1)
        h = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        e = (h[src] * self.a_src).sum(dim=-1) + (h[dst] * self.a_dst).sum(dim=-1)
        e = self.leaky(e)
        # Numerically stable exp via clamping (no extra deps)
        e = torch.clamp(e, min=-10.0, max=10.0)
        alpha_num = torch.exp(e)
        # softmax over incoming edges per destination node
        N = h.size(0)
        denom = torch.zeros(N, device=x.device, dtype=alpha_num.dtype)
        denom.scatter_add_(0, dst, alpha_num)
        alpha = alpha_num / (denom[dst] + 1e-9)
        alpha = self.drop(alpha)
        # weighted aggregate messages into dst nodes
        out = torch.zeros_like(h)
        out.index_add_(0, dst, (alpha.unsqueeze(-1) * h[src]))
        return out


class CustomGAT(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, item_feat_dim: int, hidden: int, layers: int):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.user_emb = torch.nn.Embedding(n_users, hidden)
        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        self.item_proj = torch.nn.Linear(item_feat_dim, hidden)
        self.layers = torch.nn.ModuleList([SimpleGATLayer(hidden, hidden) for _ in range(layers)])

    def node_features(self, item_feats: torch.Tensor) -> torch.Tensor:
        # Concatenate users then items to [N, hidden]
        u = self.user_emb.weight
        v = self.item_proj(item_feats)
        return torch.cat([u, v], dim=0)

    def forward(self, item_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_features(item_feats)
        for gat in self.layers:
            x = gat(x, edge_index)
        return x  # [n_users+n_items, hidden]


# -------- GCS helpers --------

def gcs_download(project_id: str, gcs_uri: str, local_path: str) -> None:
    storage_client = storage.Client(project=project_id)
    bucket_name = gcs_uri.split("/")[2]
    blob_path = "/".join(gcs_uri.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)


def gcs_upload(project_id: str, local_path: str, gcs_uri: str) -> None:
    storage_client = storage.Client(project=project_id)
    bucket_name = gcs_uri.split("/")[2]
    blob_path = "/".join(gcs_uri.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


# -------- Data utils --------

def build_splits(interactions: pd.DataFrame):
    by_user = interactions.sort_values("ts").groupby("user_id")
    train_pos, val_pos, test_pos = {}, {}, {}
    for u, g in by_user:
        items = g["asin"].values
        if len(items) >= 3:
            train_pos[u] = items[:-2]
            val_pos[u] = items[-2]
            test_pos[u] = items[-1]
        elif len(items) >= 2:
            train_pos[u] = items[:-1]
            test_pos[u] = items[-1]
        else:
            continue
    return train_pos, val_pos, test_pos


def to_indexed(interactions: pd.DataFrame, user_to_idx: Dict[str, int], item_to_idx: Dict[str, int]) -> pd.DataFrame:
    df = interactions[["user_id", "asin", "ts"]].copy()
    df["u"] = df["user_id"].map(user_to_idx)
    df["i"] = df["asin"].map(item_to_idx)
    df = df.dropna(subset=["u", "i"]).astype({"u": int, "i": int})
    return df


def build_edge_index(n_users: int, n_items: int, train_pos_idx: Dict[int, np.ndarray]) -> torch.Tensor:
    rows, cols = [], []
    for u_idx, items in train_pos_idx.items():
        for i_idx in items:
            rows.append(u_idx)                 # u -> i
            cols.append(n_users + i_idx)
            rows.append(n_users + i_idx)       # i -> u (symmetric)
            cols.append(u_idx)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index


def index_maps(maps: dict):
    u2i = {k: int(v) for k, v in maps["user_to_idx"].items()}
    i2i = {k: int(v) for k, v in maps["item_to_idx"].items()}
    return u2i, i2i


def eval_sampled(model: CustomGAT, cfg: Config, item_feats: torch.Tensor, edge_index: torch.Tensor,
                 train_pos_idx: Dict[int, np.ndarray], eval_pos: Dict[int, int], Ks=(10, 20)):
    device = next(model.parameters()).device
    with torch.no_grad():
        Z = model(item_feats, edge_index)
        U = Z[:model.n_users]
        I = Z[model.n_users:]
        user_pos_sets = {u: set(pos) for u, pos in train_pos_idx.items()}
        metrics = {f"recall@{k}": [] for k in Ks}
        metrics.update({f"ndcg@{k}": [] for k in Ks})
        for u, pos_i in eval_pos.items():
            avoid = user_pos_sets.get(u, set()) | {pos_i}
            negs = []
            while len(negs) < cfg.eval_neg_k:
                cand = np.random.randint(0, model.n_items)
                if cand not in avoid:
                    negs.append(cand)
            cand_items = np.array([pos_i] + negs, dtype=np.int64)
            u_emb = U[u]
            i_emb = I[cand_items]
            scores = (i_emb @ u_emb).cpu().numpy()
            rank = (scores > scores[0]).sum() + 1
            for k in Ks:
                hit = 1.0 if rank <= k else 0.0
                metrics[f"recall@{k}"].append(hit)
                metrics[f"ndcg@{k}"].append((1.0 / math.log2(rank + 1)) if hit else 0.0)
        return {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}


def sample_bpr_epoch(train_pos_idx: Dict[int, np.ndarray], n_items: int, samples: int):
    users = list(train_pos_idx.keys())
    out_u, out_i, out_j = [], [], []
    while len(out_u) < samples:
        u = random.choice(users)
        i = int(random.choice(train_pos_idx[u]))
        while True:
            j = random.randrange(n_items)
            if j not in train_pos_idx[u]:
                break
        out_u.append(u); out_i.append(i); out_j.append(j)
    return np.array(out_u), np.array(out_i), np.array(out_j)


def main():
    ap = argparse.ArgumentParser(description="Train Custom GAT")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--staging-prefix", default="gs://plotpointe-artifacts/staging/amazon_electronics")
    ap.add_argument("--graphs-prefix", default="gs://plotpointe-artifacts/graphs")
    ap.add_argument("--embeddings-prefix", default="gs://plotpointe-artifacts/embeddings")
    ap.add_argument("--models-prefix", default="gs://plotpointe-artifacts/models/gat/custom")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples-per-epoch", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-neg-k", type=int, default=1000)
    ap.add_argument("--item-features", choices=["fused", "txt"], default="fused")
    ap.add_argument("--loss", choices=["bpr", "bce"], default="bpr")
    args = ap.parse_args()

    cfg = Config(
        project_id=args.project_id,
        region=args.region,
        staging_prefix=args.staging_prefix,
        graphs_prefix=args.graphs_prefix,
        embeddings_prefix=args.embeddings_prefix,
        models_prefix=args.models_prefix,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        eval_neg_k=args.eval_neg_k,
        item_features=args.item_features,
        loss=args.loss,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GAT-CUSTOM] Device: {device}")

    # Download inputs
    inter_gcs = f"{cfg.staging_prefix}/interactions.parquet"
    maps_gcs = f"{cfg.graphs_prefix}/node_maps.json"
    feat_name = "fused_interacted.npy" if cfg.item_features == "fused" else "txt_interacted.npy"
    feat_gcs = f"{cfg.embeddings_prefix}/{feat_name}"
    print("[GAT-CUSTOM] Downloading inputs...")
    gcs_download(cfg.project_id, inter_gcs, "tmp/interactions.parquet")
    gcs_download(cfg.project_id, maps_gcs, "tmp/node_maps.json")
    gcs_download(cfg.project_id, feat_gcs, "tmp/features.npy")

    interactions = pd.read_parquet("tmp/interactions.parquet")
    with open("tmp/node_maps.json") as f:
        maps = json.load(f)
    user_to_idx = {k: int(v) for k, v in maps["user_to_idx"].items()}
    item_to_idx = {k: int(v) for k, v in maps["item_to_idx"].items()}

    df_idx = to_indexed(interactions, user_to_idx, item_to_idx)
    train_pos_raw, val_pos_raw, test_pos_raw = build_splits(interactions)

    # Map to index space
    train_pos_idx: Dict[int, np.ndarray] = {}
    val_pos_idx: Dict[int, int] = {}
    test_pos_idx: Dict[int, int] = {}
    for u_raw, items in train_pos_raw.items():
        u = user_to_idx.get(str(u_raw), user_to_idx.get(u_raw, None))
        if u is None: continue
        idx_items = []
        for it in items:
            it_idx = item_to_idx.get(str(it), item_to_idx.get(it, None))
            if it_idx is not None: idx_items.append(it_idx)
        if idx_items: train_pos_idx[int(u)] = np.array(idx_items, dtype=np.int64)
    for d_raw, d_idx in [(val_pos_raw, val_pos_idx), (test_pos_raw, test_pos_idx)]:
        for u_raw, it in d_raw.items():
            u = user_to_idx.get(str(u_raw), user_to_idx.get(u_raw, None))
            it_idx = item_to_idx.get(str(it), item_to_idx.get(it, None))
            if u is not None and it_idx is not None:
                d_idx[int(u)] = int(it_idx)

    n_users = int(maps["n_users"])
    n_items = int(maps["n_items"])
    print(f"[GAT-CUSTOM] n_users={n_users}, n_items={n_items}")

    # Build edge index and item features
    edge_index = build_edge_index(n_users, n_items, train_pos_idx).to(device)
    item_feats = torch.tensor(np.load("tmp/features.npy"), dtype=torch.float32).to(device)
    assert item_feats.shape[0] == n_items, "features.npy (interacted) must align to interacted items order"

    model = CustomGAT(n_users, n_items, item_feat_dim=item_feats.size(1), hidden=cfg.hidden_dim, layers=cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)

    best_ndcg20 = -1.0
    best_path = "tmp/gat_custom_best.pt"
    metrics_path = "tmp/metrics_gat_custom.json"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        # One large sampled BPR batch per epoch
        u_arr, i_arr, j_arr = sample_bpr_epoch(train_pos_idx, n_items, cfg.samples_per_epoch)
        u = torch.from_numpy(u_arr).long().to(device)
        i = torch.from_numpy(i_arr).long().to(device)
        j = torch.from_numpy(j_arr).long().to(device)

        Z = model(item_feats, edge_index)
        U = Z[:n_users]
        I = Z[n_users:]
        pos = (U[u] * I[i]).sum(dim=-1)
        neg = (U[u] * I[j]).sum(dim=-1)
        if cfg.loss == "bpr":
            loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
        else:  # bce
            logits = torch.cat([pos, neg], dim=0)
            labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[GAT-CUSTOM][Epoch {epoch}] loss={loss.item():.4f} ({cfg.loss})")

        # Validation
        model.eval()
        val_metrics = eval_sampled(model, cfg, item_feats, edge_index, train_pos_idx, val_pos_idx)
        print(f"[GAT-CUSTOM][Epoch {epoch}] val: {val_metrics}")
        if val_metrics.get("ndcg@20", 0.0) > best_ndcg20:
            best_ndcg20 = val_metrics.get("ndcg@20", 0.0)
            torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, best_path)
            print("[GAT-CUSTOM] Saved new best checkpoint")

    # Final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_metrics = eval_sampled(model, cfg, item_feats, edge_index, train_pos_idx, test_pos_idx)
    print(f"[GAT-CUSTOM] test: {test_metrics}")

    out = {
        "best_val_ndcg@20": float(best_ndcg20),
        "val": val_metrics,
        "test": test_metrics,
        "config": cfg.__dict__,
        "notes": f"One-backward-per-epoch with S sampled BPR triples; features={cfg.item_features}; loss={cfg.loss}",
    }
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)

    run_id = f"gat_custom_d{cfg.hidden_dim}_{int(time.time())}"
    chk_gcs = f"{cfg.models_prefix}/checkpoints/{run_id}.pt"
    met_gcs = f"{cfg.models_prefix}/metrics_{run_id}.json"
    gcs_upload(cfg.project_id, best_path, chk_gcs)
    gcs_upload(cfg.project_id, metrics_path, met_gcs)
    print(f"[GAT-CUSTOM] ✅ Complete. Uploaded {chk_gcs} and {met_gcs}")


if __name__ == "__main__":
    main()

