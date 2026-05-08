#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.46",
#   "trl>=0.12",
#   "peft>=0.13",
#   "accelerate>=1.0",
#   "datasets>=3.0",
#   "bitsandbytes>=0.44",
# ]
# ///
"""
SFT fine-tune small student model on chosen (rank-1) canvas completions.

Reads results/*.jsonl + (optionally) prefs.jsonl, takes the rank-1 candidate
per non-obsolete record, formats as Qwen chat, trains LoRA adapter.

Default base: Qwen/Qwen2.5-Coder-0.5B-Instruct (1B-class, fits 12GB easily).
Override with --model to try Qwen3-8B etc.

Usage:
  ./train_sft.py                              # 10 epochs, default model
  ./train_sft.py --epochs 5 --rank 32         # custom hyperparams
  ./train_sft.py --include-prefs              # also include private prefs.jsonl
  ./train_sft.py --model Qwen/Qwen3-8B        # bigger student
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
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
    for c in cands: h.update((c or "").encode("utf-8", errors="replace")); h.update(b"\x00")
    return f"{rec.get('ts',0):.3f}|{rec.get('prompt','')}|{h.hexdigest()[:16]}"


def collect_examples(include_private: bool) -> list[dict]:
    seen: dict[str, dict] = {}
    paths = sorted(glob.glob(str(RESULTS / "*.jsonl")))
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
            key = fingerprint(r)
            seen.setdefault(key, r)

    examples = []
    skipped = 0
    for rec in seen.values():
        cands = rec.get("candidates") or []
        ranks = rec.get("rankings") or []
        errs = rec.get("errored") or [False] * len(cands)
        sus = rec.get("suspect") or [False] * len(cands)
        # find best non-flagged candidate (lowest rank where errored=False and suspect=False)
        best_idx = None
        best_rank = 999
        for i, rank in enumerate(ranks):
            if errs[i] or sus[i]: continue
            if rank is None: continue
            if rank < best_rank:
                best_rank = rank; best_idx = i
        if best_idx is None or not cands[best_idx]:
            skipped += 1; continue
        examples.append({
            "prompt": rec["prompt"],
            "completion": cands[best_idx],
            "rank": best_rank,
            "model": rec.get("model"),
            "host": rec.get("host"),
        })
    return examples, len(seen), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank")
    ap.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation steps")
    ap.add_argument("--max-seq", type=int, default=1500)
    ap.add_argument("--system-prompt-id", default="A", choices=list(SYSTEM_PROMPT_VARIANTS),
                    help="which system prompt variant to train against (default A)")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none", help="QLoRA: load base in 4-bit (needed for 7B+ on 12GB)")
    ap.add_argument("--include-prefs", action="store_true", help="also use private prefs.jsonl (not just results/)")
    ap.add_argument("--output", default=None, help="checkpoint dir (default: auto from model name)")
    ap.add_argument("--limit", type=int, default=0, help="cap N examples (0=all)")
    ap.add_argument("--filter-model", default=None, help="only use examples with chosen.model containing this substring (e.g. '30B')")
    ap.add_argument("--random-fraction", type=float, default=0.0, help="randomly subsample this fraction of examples (0=all, 0.5=half)")
    ap.add_argument("--seed", type=int, default=42, help="seed for --random-fraction")
    args = ap.parse_args()

    examples, total_records, skipped = collect_examples(args.include_prefs)
    if args.filter_model:
        before = len(examples)
        examples = [e for e in examples if e.get("model") and args.filter_model in e["model"]]
        print(f"filter --filter-model={args.filter_model!r}: {before} → {len(examples)}")
    if args.random_fraction and 0 < args.random_fraction < 1:
        import random as _rnd
        rng = _rnd.Random(args.seed)
        before = len(examples)
        keep = max(1, int(round(before * args.random_fraction)))
        examples = rng.sample(examples, keep)
        print(f"random subsample (frac={args.random_fraction}, seed={args.seed}): {before} → {len(examples)}")
    if args.limit > 0:
        examples = examples[: args.limit]

    if not examples:
        print("ERROR: no usable examples found", file=sys.stderr); return 1

    print(f"loaded {total_records} unique rank records, "
          f"extracted {len(examples)} chosen examples ({skipped} skipped — all flagged or empty)")
    by_model = {}
    for e in examples:
        by_model[e.get("model") or "?"] = by_model.get(e.get("model") or "?", 0) + 1
    print(f"chosen by source model: {by_model}")

    out_name = args.output or "qwen0.5b-pixelart-" + time.strftime("%Y%m%d-%H%M")
    out_dir = CHECKPOINT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports — only after CLI parse, so --help is fast.
    print("loading torch + transformers ...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"cuda available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA — training on CPU will take many hours", file=sys.stderr)

    print(f"loading base model: {args.model} (quant={args.quant})")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(device_map="auto" if torch.cuda.is_available() else None)
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.quant == "4bit":
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base)

    # LoRA on attention + MLP
    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # Format examples as Qwen chat template
    sp_text = _get_sp(args.system_prompt_id)
    print(f"using system prompt variant: {args.system_prompt_id}")
    def fmt(ex):
        msgs = [
            {"role": "system", "content": sp_text},
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = Dataset.from_list(examples).map(fmt, remove_columns=["prompt","completion","rank","model","host"])
    print(f"training set: {len(ds)} examples")
    if len(ds) >= 5:
        sample_ids = tok(ds[0]["text"]).input_ids
        print(f"sample 0 tokens: {len(sample_ids)} | first 80 chars: {ds[0]['text'][:80]!r}")

    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        max_length=args.max_seq,
        gradient_checkpointing=True,
        # Paged 8-bit AdamW halves optimizer state memory — needed for 7B+ on 12GB.
        optim="paged_adamw_8bit" if args.quant == "4bit" else "adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    t0 = time.time()
    print(f"starting training: {args.epochs} epochs × {len(ds)} examples → {out_dir}")
    trainer.train()
    dt = time.time() - t0
    print(f"\n=== training done in {dt/60:.1f} min ===")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    summary = {
        "model": args.model,
        "examples": len(examples),
        "epochs": args.epochs,
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
        "batch": args.batch, "accum": args.accum, "max_seq": args.max_seq,
        "by_source_model": by_model,
        "wall_seconds": dt,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nadapter + tokenizer saved to {out_dir}")
    print(f"summary: {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
