#!/usr/bin/env python3
"""
PlotPointe Embeddings CLI (zero-behavior-change wrappers)

Usage examples:
  - python -m plotpointe.embeddings.cli --help
  - python -m plotpointe.embeddings.cli text -- --project-id=plotpointe --region=us-central1
  - python -m plotpointe.embeddings.cli image -- --project-id=plotpointe --region=us-central1
  - python -m plotpointe.embeddings.cli fusion -- --project-id=plotpointe --region=us-central1
  - python -m plotpointe.embeddings.cli smoke-gpu

Notes:
- Arguments after the subcommand are passed through to the underlying script
  unchanged. Include "--" before the original flags to avoid parsing here.
- This avoids any functional change while giving us a stable module entrypoint
  for CI/CD and Vertex job configs later.
"""

from __future__ import annotations
import argparse
import importlib
import sys
from typing import List

# Map subcommands to existing modules and the name shown in sys.argv[0]
MODULES = {
    "text": ("embeddings.embed_text", "embed_text.py"),
    "image": ("embeddings.embed_image", "embed_image.py"),
    "fusion": ("embeddings.fuse_modal", "fuse_modal.py"),
    "smoke-gpu": ("embeddings.smoke_test_gpu", "smoke_test_gpu.py"),
}


def _delegate(module_path: str, argv0: str, passthrough: List[str]) -> int:
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "main"):
        print(f"Underlying module {module_path} has no main() function", file=sys.stderr)
        return 2
    # Rebuild sys.argv to mimic direct invocation of the underlying script
    old_argv = sys.argv
    try:
        sys.argv = [argv0] + passthrough
        return int(mod.main() or 0)
    finally:
        sys.argv = old_argv


def main(argv: List[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="plotpointe-embeddings",
        description="Embeddings CLI delegating to existing scripts (no behavior change)",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=False)

    # Define lightweight subcommands; arguments are not parsed here.
    for name in MODULES.keys():
        subparsers.add_parser(name, help=f"Delegate to {name} pipeline (pass flags after --)")

    # If no args, or global help requested, print help
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0

    # Split into subcommand and passthrough args (after "--")
    cmd = argv[0]
    passthrough: List[str] = []

    if cmd not in MODULES:
        parser.print_help()
        print(f"\nUnknown command: {cmd}", file=sys.stderr)
        return 2

    # Everything after the subcommand and optional "--" is passed through
    if len(argv) > 1:
        if argv[1] == "--":
            passthrough = argv[2:]
        else:
            # Accept flags even without explicit "--" to reduce friction
            passthrough = argv[1:]

    module_path, argv0 = MODULES[cmd]
    return _delegate(module_path, argv0, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())

