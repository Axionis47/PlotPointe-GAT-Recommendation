#!/usr/bin/env python3
"""
Structured logging helper for JSONL logs to stdout.
Safe to import anywhere. Only emits logs when called.
"""
from __future__ import annotations
import json
import os
import sys
import time
from typing import Any, Dict, Optional


def _now_ts() -> str:
    # ISO8601-like with seconds resolution
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_event(event: str, run_id: Optional[str] = None, **fields: Any) -> Dict[str, Any]:
    """Emit a single JSON event to stdout and return the dict.
    Example: log_event("epoch_end", epoch=5, loss=0.123, val={"ndcg@20": 0.019})
    """
    rec: Dict[str, Any] = {
        "ts": _now_ts(),
        "event": event,
    }
    if run_id:
        rec["run_id"] = run_id
    # Merge additional fields
    for k, v in fields.items():
        rec[k] = v
    try:
        sys.stdout.write(json.dumps(rec) + os.linesep)
        sys.stdout.flush()
    except Exception:
        # Never fail the training run due to logging
        pass
    return rec

__all__ = ["log_event"]

