#!/usr/bin/env python3
"""
Lightweight feature manifest registry/loader.
- Keeps training scripts decoupled (no behavior change unless you opt to use this helper)
- Manifests are JSON (no extra deps); registry: manifests/registry.json
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


REGISTRY_PATH = Path("manifests/registry.json")


@dataclass
class FeatureManifest:
    id: str
    dataset: str
    paths: Dict[str, str]
    defaults: Dict[str, str]

    @staticmethod
    def from_dict(d: Dict) -> "FeatureManifest":
        required_top = ["id", "dataset", "paths", "defaults"]
        for k in required_top:
            if k not in d:
                raise ValueError(f"Manifest missing required key: {k}")
        for p in ["staging_prefix", "embeddings_prefix", "graphs_prefix"]:
            if p not in d["paths"]:
                raise ValueError(f"Manifest.paths missing required key: {p}")
        return FeatureManifest(
            id=d["id"], dataset=d["dataset"], paths=d["paths"], defaults=d.get("defaults", {})
        )


def load_registry(registry_path: Path = REGISTRY_PATH) -> Dict[str, str]:
    with registry_path.open("r") as f:
        reg = json.load(f)
    if "manifests" not in reg:
        raise ValueError("Registry missing 'manifests'")
    return reg["manifests"]


def load_manifest(manifest_id: str, registry_path: Path = REGISTRY_PATH) -> FeatureManifest:
    manifests = load_registry(registry_path)
    if manifest_id not in manifests:
        raise KeyError(f"Manifest id not found in registry: {manifest_id}")
    path = Path(manifests[manifest_id])
    with path.open("r") as f:
        d = json.load(f)
    return FeatureManifest.from_dict(d)


def resolve_paths(manifest: FeatureManifest) -> Dict[str, str]:
    """Return a dict mapping training flag names to their values based on the manifest."""
    return {
        "staging_prefix": manifest.paths["staging_prefix"],
        "embeddings_prefix": manifest.paths["embeddings_prefix"],
        "graphs_prefix": manifest.paths["graphs_prefix"],
        # Provided for convenience if a caller wants to supply defaults
        "item_features": manifest.defaults.get("item_features", "fused"),
    }


__all__ = ["FeatureManifest", "load_manifest", "resolve_paths", "load_registry"]

