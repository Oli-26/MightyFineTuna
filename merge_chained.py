"""
Chain-merge two LoRA adapters sequentially into the base model and save the
result as a fresh fp16 model dir suitable for use as a `--merged-base-dir`
input to train_dpo.py.

Usage:
  python merge_chained.py \
      --base Qwen/Qwen2.5-Coder-3B-Instruct \
      --adapters checkpoints/3b-parent_b_v2 checkpoints/3b-dpo-from-3b-parent_b_v2-20260510-0139 \
      --out checkpoints/3b-dpo-merged-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="adapter dirs to apply in order (earliest first)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"[merge] base={args.base} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    for i, adapter in enumerate(args.adapters):
        print(f"[merge] step {i+1}/{len(args.adapters)}: applying {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[merge] saved → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
