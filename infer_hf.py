#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.46",
#   "peft>=0.13",
#   "accelerate>=1.0",
#   "bitsandbytes>=0.44",
# ]
# ///
"""
Inference for blind eval. Uses HF transformers (+ PEFT adapter) to generate
canvas JS for each prompt and write side_{base|tuned}.jsonl — same format
that blind_eval.py --pair expects.

Usage:
  ./infer_hf.py --side base  --prompts eval_prompts.txt
  ./infer_hf.py --side tuned --adapter checkpoints/qwen0.5b-pixelart-LATEST --prompts eval_prompts.txt
  ./blind_eval.py --pair side_base.jsonl side_tuned.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

from system_prompts import SYSTEM_PROMPT_VARIANTS, get as _get_sp

_FENCE_RE = re.compile(r"```(?:javascript|js|html|jsx|typescript|ts)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_REDECL_RE = re.compile(r"^\s*(?:const|let|var)\s+(?:ctx|W|H|canvas|c)\b")


def strip_fences(s: str) -> str:
    s = _THINK_RE.sub("", s).strip()
    matches = _FENCE_RE.findall(s)
    if matches:
        s = max(matches, key=len).strip()
    elif s.startswith("```"):
        nl = s.find("\n")
        if nl != -1: s = s[nl + 1:]
        if s.endswith("```"): s = s[:-3]
        s = s.strip()
    return "\n".join(ln for ln in s.split("\n") if not _REDECL_RE.match(ln)).strip()


def load_prompts(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"): out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir")
    ap.add_argument("--side", choices=["base", "tuned"], required=True)
    ap.add_argument("--prompts", default=str(ROOT / "eval_prompts.txt"))
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--system-prompt-id", default="A", choices=list(SYSTEM_PROMPT_VARIANTS),
                    help="which system prompt variant to use (default A)")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none",
                    help="load base in 4-bit (needed for 7B+ on 12GB to avoid CPU offload)")
    ap.add_argument("--draft", default=None, help="speculative decoding: HF id of draft model (must share tokenizer)")
    ap.add_argument("--draft-adapter", default=None, help="LoRA adapter to apply to draft model")
    ap.add_argument("--out", default=None, help="output jsonl (default: side_{side}.jsonl)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    if not prompts:
        print("no prompts found", file=sys.stderr); return 1

    out_path = Path(args.out) if args.out else (ROOT / f"side_{args.side}.jsonl")

    seen = set()
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: seen.add(json.loads(line)["prompt"])
            except Exception: pass
        prompts = [p for p in prompts if p not in seen]
        print(f"[infer:{args.side}] resuming, {len(seen)} done, {len(prompts)} to go", file=sys.stderr)
    elif args.overwrite and out_path.exists():
        out_path.unlink()

    if not prompts:
        print("nothing to do"); return 0

    print(f"[infer:{args.side}] loading torch + transformers ...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if args.adapter:
        from peft import PeftModel

    print(f"  cuda: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    print(f"[infer:{args.side}] base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)

    model_id = args.base
    if args.adapter:
        print(f"[infer:{args.side}] adapter: {args.adapter} (quant={args.quant})")
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.quant == "none":
            # safe to merge when not quantized
            model = model.merge_and_unload()
        # else: keep adapter attached on top of 4-bit base
        model_id = f"{args.base}+adapter:{Path(args.adapter).name}"

    model.eval()

    draft_model = None
    draft_tok = None
    if args.draft:
        print(f"[infer:{args.side}] draft: {args.draft} adapter={args.draft_adapter}")
        draft_tok = AutoTokenizer.from_pretrained(args.draft)
        if draft_tok.pad_token is None:
            draft_tok.pad_token = draft_tok.eos_token
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if args.draft_adapter:
            draft_model = PeftModel.from_pretrained(draft_model, args.draft_adapter)
            draft_model = draft_model.merge_and_unload()
        draft_model.eval()
        model_id += f"+spec({Path(args.draft).name})"
        if args.draft_adapter:
            model_id += f"+draft_adapter:{Path(args.draft_adapter).name}"

    sp_text = _get_sp(args.system_prompt_id)
    print(f"[infer:{args.side}] system prompt variant: {args.system_prompt_id}")

    def gen(prompt: str) -> tuple[str, str]:
        msgs = [{"role": "system", "content": sp_text}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        gen_kwargs = dict(
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
        if draft_model is not None:
            gen_kwargs["assistant_model"] = draft_model
            # Universal assisted decoding — needed when target/draft tokenizers differ
            # (or when transformers' validator can't prove they match).
            gen_kwargs["tokenizer"] = tok
            gen_kwargs["assistant_tokenizer"] = draft_tok
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_ids = out[0][prompt_len:]
        raw = tok.decode(new_ids, skip_special_tokens=True)
        return raw, strip_fences(raw)

    print(f"[infer:{args.side}] {len(prompts)} prompts -> {out_path}\n", flush=True)
    started = time.time()
    for i, p in enumerate(prompts, 1):
        t0 = time.time()
        try:
            raw, code = gen(p)
        except Exception as e:
            import traceback
            print(f"[{i}/{len(prompts)}] ERROR: {type(e).__name__}: {e!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue
        rec = {
            "ts": time.time(),
            "side": args.side,
            "model": model_id,
            "prompt": p,
            "code": code,
            "raw_len": len(raw),
            "temp": args.temp,
        }
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        dt = time.time() - t0
        print(f"[{i}/{len(prompts)}] {p!r} -> {len(code)} chars ({dt:.1f}s)", flush=True)

    print(f"\n[infer:{args.side}] done in {(time.time()-started)/60:.1f} min, wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
