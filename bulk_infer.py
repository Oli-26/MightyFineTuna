#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.46",
#   "peft>=0.13",
#   "accelerate>=1.0",
# ]
# ///
"""
Bulk inference: load base ONCE, swap LoRA adapters, gen sides for each.

Usage:
  ./bulk_infer.py --base Qwen/Qwen2.5-Coder-3B-Instruct \
    --adapters checkpoints/3b-parent_a checkpoints/3b-parent_b ... \
    --prompts eval_prompts.txt \
    --out-prefix side_3b_ \
    --system-prompt-id A
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
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
        if nl != -1: s = s[nl+1:]
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
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="adapter dirs")
    ap.add_argument("--prompts", default=str(ROOT / "eval_prompts.txt"))
    ap.add_argument("--out-prefix", default="side_3b_")
    ap.add_argument("--out-dir", default=str(ROOT))
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=2500, help="bumped for eval; training data uses batch_gen.py defaults")
    ap.add_argument("--system-prompt-id", default="A", choices=list(SYSTEM_PROMPT_VARIANTS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    if not prompts:
        print("no prompts"); return 1
    print(f"[bulk] {len(prompts)} prompts × {len(args.adapters)} adapters = {len(prompts)*len(args.adapters)} gens")

    print("[bulk] loading torch + transformers...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[bulk] cuda: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    print(f"[bulk] base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
    )
    base.eval()

    sp_text = _get_sp(args.system_prompt_id)
    print(f"[bulk] system prompt: {args.system_prompt_id} ({len(sp_text)} chars)")

    # Wrap in PEFT once with first adapter, then load+set further adapters.
    first = args.adapters[0]
    print(f"[bulk] wrapping with first adapter: {Path(first).name}")
    model = PeftModel.from_pretrained(base, first, adapter_name=Path(first).name)
    for a in args.adapters[1:]:
        nm = Path(a).name
        print(f"[bulk] loading adapter: {nm}")
        model.load_adapter(a, adapter_name=nm)
    model.eval()

    out_dir = Path(args.out_dir)

    t_start = time.time()
    for adapter_path in args.adapters:
        nm = Path(adapter_path).name
        out_path = out_dir / f"{args.out_prefix}{nm}.jsonl"
        seen: set[str] = set()
        if out_path.exists() and not args.overwrite:
            for line in out_path.read_text().splitlines():
                line = line.strip()
                if not line: continue
                try: seen.add(json.loads(line)["prompt"])
                except Exception: pass
        elif args.overwrite and out_path.exists():
            out_path.unlink()
        todo = [p for p in prompts if p not in seen]
        if not todo:
            print(f"[bulk] {nm}: all done, skip")
            continue

        model.set_adapter(nm)
        print(f"\n[bulk] === {nm} ({len(todo)} prompts) ===", flush=True)

        for i, prompt in enumerate(todo, 1):
            t0 = time.time()
            msgs = [{"role":"system","content":sp_text},{"role":"user","content":prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temp,
                        do_sample=True,
                        pad_token_id=tok.eos_token_id,
                    )
                new_ids = out[0][prompt_len:]
                raw = tok.decode(new_ids, skip_special_tokens=True)
                code = strip_fences(raw)
            except Exception as e:
                print(f"  ERR {prompt!r}: {e}")
                continue
            rec = {
                "ts": time.time(),
                "side": nm,
                "model": f"{args.base}+adapter:{nm}",
                "prompt": prompt,
                "code": code,
                "raw_len": len(raw),
                "temp": args.temp,
                "system_prompt_id": args.system_prompt_id,
            }
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            dt = time.time() - t0
            print(f"  [{i}/{len(todo)}] {prompt!r} -> {len(code)} chars ({dt:.1f}s)", flush=True)

        print(f"[bulk] wrote {out_path.name}")

    elapsed = (time.time() - t_start) / 60
    print(f"\n[bulk] DONE in {elapsed:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
