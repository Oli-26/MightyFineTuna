#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "safetensors>=0.4",
#   "numpy>=1.26",
#   "packaging>=23.0",
# ]
# ///
"""
Merge LoRA adapters by weighted average. All adapters must share architecture
(same target_modules, rank, base model). Output is a fresh adapter dir
compatible with PeftModel.from_pretrained.

Use cases:
- Combine adapters trained on different data subsets
- Evolutionary mixing: sample random simplex weights, merge, blind-eval

Usage:
  # Explicit weighted merge
  ./merge_loras.py A B C --weights 0.5 0.3 0.2 --out merged_v1

  # Equal-weight average (default if --weights omitted)
  ./merge_loras.py A B C --out merged_avg

  # Random mix sampling: generate N children with random Dirichlet weights
  ./merge_loras.py A B C D --random-mix 8 --out-prefix gen2_

  # Just inspect compatibility
  ./merge_loras.py A B --check
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def load_adapter_state(adapter_dir: Path) -> dict[str, torch.Tensor]:
    sf = adapter_dir / "adapter_model.safetensors"
    if not sf.exists():
        raise FileNotFoundError(f"missing adapter_model.safetensors in {adapter_dir}")
    return load_file(str(sf))


def load_adapter_config(adapter_dir: Path) -> dict:
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"missing adapter_config.json in {adapter_dir}")
    return json.loads(cfg.read_text())


def check_compat(adapters: list[Path]) -> tuple[set[str], dict]:
    """Return (common_keys, reference_config). Errors if incompatible."""
    states = [load_adapter_state(a) for a in adapters]
    configs = [load_adapter_config(a) for a in adapters]

    keys = set(states[0])
    for i, s in enumerate(states[1:], 1):
        if set(s) != keys:
            extra = set(s) - keys
            missing = keys - set(s)
            raise ValueError(
                f"adapter {adapters[i]} keys differ from {adapters[0]}:\n"
                f"  extra in {adapters[i].name}: {sorted(extra)[:5]}{'...' if len(extra)>5 else ''}\n"
                f"  missing from {adapters[i].name}: {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}"
            )

    # Cross-check critical config fields
    ref = configs[0]
    for i, c in enumerate(configs[1:], 1):
        for field in ("base_model_name_or_path", "r", "lora_alpha", "target_modules"):
            if c.get(field) != ref.get(field):
                # target_modules might be list — compare as set
                if field == "target_modules" and set(c.get(field) or []) == set(ref.get(field) or []):
                    continue
                print(f"WARNING: {adapters[i].name} {field}={c.get(field)!r} differs from {adapters[0].name} {field}={ref.get(field)!r}", file=sys.stderr)
    return keys, ref


def merge_weighted(adapters: list[Path], weights: list[float], out_dir: Path) -> None:
    weights = np.array(weights, dtype=float)
    if (weights < 0).any():
        raise ValueError("weights must be non-negative")
    weights = weights / weights.sum()

    states = [load_adapter_state(a) for a in adapters]
    keys = set(states[0])
    for s in states[1:]:
        keys &= set(s)

    merged = {}
    for k in sorted(keys):
        # All tensors must have the same shape; weighted sum
        ref = states[0][k]
        out = torch.zeros_like(ref, dtype=torch.float32)
        for w, s in zip(weights, states):
            out = out + w * s[k].to(torch.float32)
        merged[k] = out.to(ref.dtype)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy first adapter's config + tokenizer files
    src = adapters[0]
    for name in (
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        sf = src / name
        if sf.exists():
            shutil.copy(sf, out_dir / name)

    save_file(merged, str(out_dir / "adapter_model.safetensors"))

    # Drop a metadata sidecar
    meta = {
        "merge_kind": "weighted_average",
        "ts": time.time(),
        "parents": [str(p) for p in adapters],
        "weights": weights.tolist(),
        "n_keys": len(merged),
    }
    (out_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="+", help="adapter directories to merge")
    ap.add_argument("--weights", nargs="+", type=float, help="weights per adapter (default: equal)")
    ap.add_argument("--out", help="output adapter dir (single merge)")
    ap.add_argument("--out-prefix", default="mix_", help="prefix for --random-mix outputs")
    ap.add_argument("--random-mix", type=int, default=0, help="generate N random Dirichlet-weighted merges")
    ap.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration (1.0=uniform on simplex; <1 sparser, >1 more centered)")
    ap.add_argument("--check", action="store_true", help="just verify compatibility")
    ap.add_argument("--seed", type=int, default=None, help="random seed for --random-mix")
    args = ap.parse_args()

    adapter_paths = [Path(a).resolve() for a in args.adapters]
    for p in adapter_paths:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr); return 1

    if args.check:
        keys, ref = check_compat(adapter_paths)
        print(f"OK: {len(adapter_paths)} adapters compatible.")
        print(f"  keys: {len(keys)}")
        print(f"  base: {ref.get('base_model_name_or_path')}")
        print(f"  r={ref.get('r')} alpha={ref.get('lora_alpha')} targets={ref.get('target_modules')}")
        return 0

    if args.random_mix > 0:
        rng = np.random.default_rng(args.seed)
        n = len(adapter_paths)
        for i in range(args.random_mix):
            ws = rng.dirichlet([args.alpha] * n)
            # If prefix contains a path separator, treat as a path; else nest under
            # the first adapter's parent dir.
            if "/" in args.out_prefix:
                out = Path(f"{args.out_prefix}{i:02d}")
            else:
                out = Path(adapter_paths[0]).parent / f"{args.out_prefix}{i:02d}"
            merge_weighted(adapter_paths, ws.tolist(), out)
            label = " ".join(f"{w:.2f}" for w in ws)
            print(f"  {out.name}: weights = [{label}]")
        return 0

    if not args.out:
        print("--out required (or use --random-mix N)", file=sys.stderr); return 1

    n = len(adapter_paths)
    weights = args.weights if args.weights else [1.0] * n
    if len(weights) != n:
        print(f"weights count ({len(weights)}) != adapters count ({n})", file=sys.stderr); return 1

    out_dir = Path(args.out).resolve()
    if not out_dir.is_absolute() or not str(out_dir).startswith(str(Path(adapter_paths[0]).parent)):
        # Allow relative path next to first adapter's parent
        out_dir = Path(adapter_paths[0]).parent / args.out

    merge_weighted(adapter_paths, weights, out_dir)
    norm_weights = np.array(weights, float); norm_weights /= norm_weights.sum()
    print(f"merged {n} adapters → {out_dir}")
    print(f"  weights (normalized): {[f'{w:.3f}' for w in norm_weights]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
