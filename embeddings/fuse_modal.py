#!/usr/bin/env python3
"""
Multimodal fusion: concat text+image embeddings → MLP → 128d fused embedding.
Trains a simple MLP fusion layer on the concatenated embeddings.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from google.cloud import aiplatform, storage
from tqdm import tqdm


class FusionMLP(nn.Module):
    """Simple MLP to fuse text and image embeddings, plus modality projection heads."""
    def __init__(self, text_dim, img_dim, output_dim=128, hidden_dim=256):
        super().__init__()
        input_dim = text_dim + img_dim
        # Fusion MLP maps concat([txt, img]) -> output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        # Projection heads map each modality into the fused space for contrastive loss
        self.txt_proj = nn.Linear(text_dim, output_dim)
        self.img_proj = nn.Linear(img_dim, output_dim)

    def forward(self, text_emb, img_emb):
        concat = torch.cat([text_emb, img_emb], dim=-1)
        return self.mlp(concat)


def contrastive_fusion_loss(fused, txt_emb, img_emb, temperature=0.07):
    """
    InfoNCE contrastive loss: fused embeddings should be similar to both
    text and image embeddings of the same item.

    This is better than simple L2 norm loss because it:
    1. Preserves semantic similarity
    2. Learns meaningful fusion
    3. Has supervision signal from both modalities
    """
    import torch.nn.functional as F

    batch_size = fused.size(0)

    # Normalize embeddings
    fused_norm = F.normalize(fused, dim=-1)
    txt_norm = F.normalize(txt_emb, dim=-1)
    img_norm = F.normalize(img_emb, dim=-1)

    # Compute similarity matrices
    sim_fused_txt = torch.matmul(fused_norm, txt_norm.T) / temperature
    sim_fused_img = torch.matmul(fused_norm, img_norm.T) / temperature

    # Labels: diagonal elements (same item)
    labels = torch.arange(batch_size, device=fused.device)

    # Cross-entropy loss (InfoNCE)
    loss_txt = F.cross_entropy(sim_fused_txt, labels)
    loss_img = F.cross_entropy(sim_fused_img, labels)

    # Combined loss
    loss = (loss_txt + loss_img) / 2

    return loss, loss_txt.item(), loss_img.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="plotpointe")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--embeddings-prefix", default="gs://plotpointe-artifacts/embeddings")
    parser.add_argument("--output-prefix", default="gs://plotpointe-artifacts/embeddings")
    parser.add_argument("--experiment-name", default="recsys-dev")
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Initialize Vertex AI
    aiplatform.init(project=args.project_id, location=args.region, experiment=args.experiment_name)

    run_id = f"fuse-modal-{int(time.time())}"
    print(f"[FUSE_MODAL] Run ID: {run_id}")

    # Start Vertex Experiment run
    aiplatform.start_run(run_id)
    aiplatform.log_params({
        "task": "fuse_modal",
        "output_dim": args.output_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
    })
    
    start_time = time.time()
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FUSE_MODAL] Device: {device}")
    
    # Download embeddings from GCS
    storage_client = storage.Client(project=args.project_id)
    bucket_name = args.embeddings_prefix.split("/")[2]
    bucket = storage_client.bucket(bucket_name)
    
    Path("tmp").mkdir(exist_ok=True)
    
    print(f"[FUSE_MODAL] Downloading text embeddings...")
    blob_txt = bucket.blob("embeddings/txt.npy")
    blob_txt.download_to_filename("tmp/txt.npy")
    txt_emb = np.load("tmp/txt.npy")
    print(f"[FUSE_MODAL] Text embeddings: {txt_emb.shape}")
    
    print(f"[FUSE_MODAL] Downloading image embeddings...")
    blob_img = bucket.blob("embeddings/img.npy")
    blob_img.download_to_filename("tmp/img.npy")
    img_emb = np.load("tmp/img.npy")
    print(f"[FUSE_MODAL] Image embeddings: {img_emb.shape}")
    
    print(f"[FUSE_MODAL] Downloading image items mapping...")
    blob_items = bucket.blob("embeddings/img_items.parquet")
    blob_items.download_to_filename("tmp/img_items.parquet")
    
    import pandas as pd
    img_items = pd.read_parquet("tmp/img_items.parquet")
    
    # Load all items to get asin index
    blob_all_items = bucket.blob("staging/amazon_electronics/items.parquet")
    blob_all_items.download_to_filename("tmp/items.parquet")
    all_items = pd.read_parquet("tmp/items.parquet")
    
    # Create asin → index mapping
    asin_to_idx = {asin: idx for idx, asin in enumerate(all_items["asin"])}
    
    # Map image embeddings to text embedding indices
    img_indices = [asin_to_idx[asin] for asin in img_items["asin"] if asin in asin_to_idx]
    print(f"[FUSE_MODAL] Matched {len(img_indices)} items with both text and image embeddings")

    # Create O(1) lookup dictionary for efficient access
    img_idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(img_indices)}
    print(f"[FUSE_MODAL] Created efficient lookup map")

    # Align embeddings
    txt_emb_aligned = txt_emb[img_indices]
    img_emb_aligned = img_emb[:len(img_indices)]

    print(f"[FUSE_MODAL] Aligned text: {txt_emb_aligned.shape}, image: {img_emb_aligned.shape}")

    # Compute mean image embedding for items without images
    mean_img_emb = img_emb_aligned.mean(axis=0)
    print(f"[FUSE_MODAL] Computed mean image embedding for fallback")
    
    # Initialize fusion model
    text_dim = txt_emb_aligned.shape[1]
    img_dim = img_emb_aligned.shape[1]
    model = FusionMLP(text_dim, img_dim, args.output_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Convert to tensors
    txt_tensor = torch.from_numpy(txt_emb_aligned).float()
    img_tensor = torch.from_numpy(img_emb_aligned).float()

    # Training loop with contrastive loss
    print(f"[FUSE_MODAL] Training fusion MLP for {args.epochs} epochs...")
    print(f"[FUSE_MODAL] Using contrastive loss (InfoNCE) for better fusion")
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_loss_txt = 0.0
        epoch_loss_img = 0.0
        n_batches = 0

        for i in range(0, len(txt_tensor), args.batch_size):
            batch_txt = txt_tensor[i:i+args.batch_size].to(device)
            batch_img = img_tensor[i:i+args.batch_size].to(device)

            optimizer.zero_grad()
            fused = model(batch_txt, batch_img)

            # Project modalities to fused space for contrastive loss
            txt_proj = model.txt_proj(batch_txt)
            img_proj = model.img_proj(batch_img)

            # Contrastive loss: fused should be similar to both text and image (same dim)
            loss, loss_txt, loss_img = contrastive_fusion_loss(fused, txt_proj, img_proj)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_txt += loss_txt
            epoch_loss_img += loss_img
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_loss_txt = epoch_loss_txt / n_batches
        avg_loss_img = epoch_loss_img / n_batches
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} (txt={avg_loss_txt:.4f}, img={avg_loss_img:.4f})")
        aiplatform.log_time_series_metrics({
            "train_loss": avg_loss,
            "train_loss_txt": avg_loss_txt,
            "train_loss_img": avg_loss_img,
        })
    
    # Generate fused embeddings for ALL items (use mean for missing images)
    print(f"[FUSE_MODAL] Generating fused embeddings for all items...")
    print(f"[FUSE_MODAL] Items with images: {len(img_indices)} ({100*len(img_indices)/len(txt_emb):.1f}%)")
    print(f"[FUSE_MODAL] Items without images: {len(txt_emb)-len(img_indices)} ({100*(len(txt_emb)-len(img_indices))/len(txt_emb):.1f}%)")
    model.eval()

    # Convert mean image embedding to tensor once
    mean_img_tensor = torch.from_numpy(mean_img_emb).float().to(device)

    fused_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(txt_emb), args.batch_size)):
            batch_txt = torch.from_numpy(txt_emb[i:i+args.batch_size]).float().to(device)

            # For items without images, use mean image embedding (better than zero)
            batch_img = mean_img_tensor.unsqueeze(0).expand(len(batch_txt), -1).clone()

            # For items with images, use actual embeddings (O(1) lookup)
            for j, global_idx in enumerate(range(i, min(i+args.batch_size, len(txt_emb)))):
                if global_idx in img_idx_map:
                    local_img_idx = img_idx_map[global_idx]
                    batch_img[j] = torch.from_numpy(img_emb_aligned[local_img_idx]).float().to(device)

            fused = model(batch_txt, batch_img)
            # Normalize
            fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)
            fused_embeddings.append(fused.cpu().numpy())
    
    fused_embeddings = np.vstack(fused_embeddings)
    print(f"[FUSE_MODAL] Fused embeddings: {fused_embeddings.shape}")
    
    # Save locally
    fused_npy_path = "tmp/fused.npy"
    np.save(fused_npy_path, fused_embeddings)
    print(f"[FUSE_MODAL] Saved fused embeddings to {fused_npy_path}")
    
    # Save fusion config
    fusion_config = {
        "text_dim": int(text_dim),
        "img_dim": int(img_dim),
        "output_dim": args.output_dim,
        "hidden_dim": args.hidden_dim,
        "count": len(fused_embeddings),
        "items_with_images": len(img_indices),
        "run_id": run_id,
    }
    config_path = "tmp/fusion_config.json"
    with open(config_path, "w") as f:
        json.dump(fusion_config, f, indent=2)
    print(f"[FUSE_MODAL] Saved fusion config to {config_path}")
    
    # Upload to GCS
    blob_fused = bucket.blob("embeddings/fused.npy")
    blob_fused.upload_from_filename(fused_npy_path)
    fused_uri = f"{args.output_prefix}/fused.npy"
    print(f"[FUSE_MODAL] Uploaded to {fused_uri}")
    
    blob_config = bucket.blob("embeddings/fusion_config.json")
    blob_config.upload_from_filename(config_path)
    config_uri = f"{args.output_prefix}/fusion_config.json"
    print(f"[FUSE_MODAL] Uploaded to {config_uri}")
    
    total_time = time.time() - start_time
    
    # Log metrics
    aiplatform.log_metrics({
        "count": len(fused_embeddings),
        "items_with_images": len(img_indices),
        "output_dim": args.output_dim,
        "total_time_sec": total_time,
    })
    
    # Log artifacts
    aiplatform.log_params({
        "fused_npy_uri": fused_uri,
        "fusion_config_uri": config_uri,
    })
    
    aiplatform.end_run()
    
    print(f"[FUSE_MODAL] ✅ Complete")
    print(f"  Run ID: {run_id}")
    print(f"  Count: {len(fused_embeddings)}")
    print(f"  Items with images: {len(img_indices)}")
    print(f"  Output dimension: {args.output_dim}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Artifacts: {fused_uri}, {config_uri}")


if __name__ == "__main__":
    main()

