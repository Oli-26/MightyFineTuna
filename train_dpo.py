#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "transformers==4.46.3",
#   "trl==0.12.2",
#   "peft==0.13.2",
#   "accelerate>=1.0",
#   "datasets>=3.0",
#   "bitsandbytes>=0.44",
#   "sentencepiece>=0.2",
#   "protobuf>=4",
# ]
# ///
"""
DPO finetune on rank pairs.

For each rank-v1 record with N candidates ranked 1..N, extract pairs:
  (chosen=better_idx, rejected=worse_idx) for each pair where better_rank < worse_rank
Filter both candidates as not flagged (errored/suspect).

Default: human-rated only (auto_rated != True). Use --include-auto for all.

Reference model = init model. To support resuming from an existing LoRA (e.g.
3b-parent_b_v2) we merge it into the base FIRST, save as a new fp16 base,
then add a fresh LoRA for DPO training. Reference == merged base (adapter off).

Usage:
  ./train_dpo.py --init-adapter checkpoints/3b-parent_b_v2 --epochs 1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PRIVATE_PREFS = ROOT / "prefs.jsonl"
CHECKPOINT_DIR = ROOT / "checkpoints"

from system_prompts import SYSTEM_PROMPT_VARIANTS, get as _get_sp


def fingerprint(rec: dict) -> str:
    cands = rec.get("candidates") or []
    h = hashlib.sha1()
    for c in cands: h.update((c or "").encode("utf-8", "replace")); h.update(b"\x00")
    return f"{rec.get('ts',0):.3f}|{rec.get('prompt','')}|{h.hexdigest()[:16]}"


def collect_pairs(include_auto: bool, include_private: bool) -> list[dict]:
    seen: dict[str, dict] = {}
    paths = sorted(glob.glob(str(RESULTS / "*-prefs.jsonl")))
    if include_private and PRIVATE_PREFS.exists():
        paths.append(str(PRIVATE_PREFS))
    for fp in paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("schema") != "rank-v1": continue
            if r.get("obsolete"): continue
            if not include_auto and r.get("auto_rated"): continue
            seen.setdefault(fingerprint(r), r)

    pairs = []
    n_recs = 0
    for rec in seen.values():
        n_recs += 1
        cands = rec.get("candidates") or []
        ranks = rec.get("rankings") or []
        errs = rec.get("errored") or [False] * len(cands)
        sus = rec.get("suspect") or [False] * len(cands)
        # collect valid (idx, rank) tuples
        valid = []
        for i, rk in enumerate(ranks):
            if rk is None: continue
            if errs[i] or sus[i]: continue
            if not cands[i]: continue
            valid.append((i, rk))
        # generate pairs across all rank-distinct combos
        for a_idx, a_rk in valid:
            for b_idx, b_rk in valid:
                if a_rk >= b_rk: continue  # only keep (better, worse) where rank_a < rank_b
                pairs.append({
                    "prompt": rec["prompt"],
                    "chosen": cands[a_idx],
                    "rejected": cands[b_idx],
                    "rank_chosen": a_rk,
                    "rank_rejected": b_rk,
                    "model": rec.get("model"),
                })
    return pairs, n_recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--init-adapter", required=True,
                    help="LoRA adapter to merge into base as starting point + reference")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-6, help="DPO needs lower lr than SFT")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature; lower=less constrained")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2500, help="max total seq length (prompt+completion)")
    ap.add_argument("--quant", choices=["none","4bit"], default="none", help="load base in 4-bit (needed for DPO on 12GB)")
    ap.add_argument("--system-prompt-id", default="F", choices=list(SYSTEM_PROMPT_VARIANTS))
    ap.add_argument("--include-auto", action="store_true",
                    help="include auto_rated records (default: human-rated only)")
    ap.add_argument("--include-prefs", action="store_true", default=True,
                    help="include local prefs.jsonl (default true)")
    ap.add_argument("--no-private", action="store_true", help="exclude prefs.jsonl")
    ap.add_argument("--output", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merged-base-dir", default=None,
                    help="path to save merged base. defaults under checkpoints/")
    args = ap.parse_args()

    pairs, total_recs = collect_pairs(args.include_auto, not args.no_private)
    print(f"loaded {total_recs} unique rank records, extracted {len(pairs)} preference pairs"
          f" (auto={args.include_auto})")
    if args.limit > 0: pairs = pairs[: args.limit]
    if not pairs:
        print("no pairs", file=sys.stderr); return 1

    # Distribution of pair difficulty (rank gap)
    from collections import Counter
    gaps = Counter((p["rank_rejected"] - p["rank_chosen"]) for p in pairs)
    print(f"rank-gap distribution: {dict(gaps)}")

    out_name = args.output or f"3b-dpo-from-{Path(args.init_adapter).name}-{time.strftime('%Y%m%d-%H%M')}"
    out_dir = CHECKPOINT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: merge init_adapter into base (so reference = base+v2 weights).
    merged_dir = Path(args.merged_base_dir) if args.merged_base_dir else (CHECKPOINT_DIR / f"{Path(args.init_adapter).name}-merged")
    if not (merged_dir / "model.safetensors").exists() and not list(merged_dir.glob("model-*.safetensors")):
        print(f"\n[merge] merging {args.init_adapter} into base → {merged_dir}", flush=True)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(args.base)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
        m = PeftModel.from_pretrained(base, args.init_adapter)
        m = m.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        m.save_pretrained(str(merged_dir))
        tok.save_pretrained(str(merged_dir))
        del m, base
        print(f"[merge] done")
    else:
        print(f"[merge] reusing existing merged base at {merged_dir}")

    # Step 2: load merged model, attach fresh LoRA, train DPO
    print(f"\n[dpo] loading torch + transformers...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig

    print(f"cuda={torch.cuda.is_available()} | dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # Load tokenizer from original base (merged_dir was saved by newer transformers
    # that stores chat_template in a separate file the older one doesn't read).
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = dict(device_map="auto" if torch.cuda.is_available() else None)
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(str(merged_dir), **load_kwargs)
    if args.quant == "4bit":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    # Use adapter-toggle ref (TRL 0.12 supports this cleanly)
    ref_model = None

    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    sp_text = _get_sp(args.system_prompt_id)
    print(f"[dpo] system prompt {args.system_prompt_id} ({len(sp_text)} chars)")

    def format_pair(ex):
        # Pre-apply chat template for prompt; chosen/rejected stay raw assistant text
        msgs = [
            {"role": "system", "content": sp_text},
            {"role": "user", "content": ex["prompt"]},
        ]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt_text,
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    ds = Dataset.from_list(pairs).map(
        format_pair, remove_columns=["rank_chosen","rank_rejected","model"]
    )
    print(f"[dpo] training set: {len(ds)} pairs")
    print(f"[dpo] example prompt prefix: {ds[0]['prompt'][:100]!r}")

    cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        log_level="info",
        logging_first_step=True,
        save_strategy="steps",
        save_steps=30,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        max_length=args.max_len,
        max_prompt_length=args.max_len // 2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if args.quant == "4bit" else "adamw_torch",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora_cfg,
    )

    # Sanity check: confirm something is actually trainable
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[dpo] trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.4f}%)", flush=True)
    if n_train == 0:
        print("ERROR: nothing trainable — bailing", file=sys.stderr); return 1

    t0 = time.time()
    print(f"\n[dpo] starting: {args.epochs} epochs × {len(ds)} pairs → {out_dir}", flush=True)
    trainer.train()
    dt = time.time() - t0
    print(f"\n=== DPO done in {dt/60:.1f} min ===")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    summary = {
        "base": args.base,
        "init_adapter": args.init_adapter,
        "merged_base": str(merged_dir),
        "pairs": len(pairs),
        "epochs": args.epochs,
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr, "beta": args.beta,
        "system_prompt_id": args.system_prompt_id,
        "include_auto": args.include_auto,
        "wall_seconds": dt,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nadapter + tokenizer at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
