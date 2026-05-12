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
    ap.add_argument("--batch-size", type=int, default=1,
                    help="prompts per generate() call. >1 enables true batching (left-padded).")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    if not prompts:
        print("no prompts"); return 1
    print(f"[bulk] {len(prompts)} prompts × {len(args.adapters)} adapters = {len(prompts)*len(args.adapters)} gens")

    # Early-skip BEFORE loading torch/model: if every adapter's output file
    # already contains all prompts, there's nothing to do. Loading torch and
    # then exiting can hang on ROCm Windows CUDA teardown.
    out_dir_pre = Path(args.out_dir)
    pending_adapters = []
    for adapter_path in args.adapters:
        nm = Path(adapter_path).name
        out_path = out_dir_pre / f"{args.out_prefix}{nm}.jsonl"
        if not out_path.exists() or args.overwrite:
            pending_adapters.append(adapter_path); continue
        seen = set()
        for line in out_path.read_text().splitlines():
            try: seen.add(json.loads(line)["prompt"])
            except Exception: pass
        if any(p not in seen for p in prompts):
            pending_adapters.append(adapter_path)
    if not pending_adapters:
        print("[bulk] all adapters complete, nothing to do")
        return 0

    print("[bulk] loading torch + transformers...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    # torch ROCm Windows wheel lacks torch.distributed.fsdp;
    # transformers.generate() imports is_fsdp_managed_module into generation.utils,
    # so patch both that module and the source module to be safe.
    _stub = lambda *a, **k: False
    import transformers.integrations.fsdp as _fsdp_mod
    import transformers.generation.utils as _gen_utils
    _fsdp_mod.is_fsdp_managed_module = _stub
    _gen_utils.is_fsdp_managed_module = _stub

    print(f"[bulk] cuda: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    print(f"[bulk] base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if args.batch_size > 1:
        tok.padding_side = "left"  # required for batched generation
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
    )
    base.eval()

    sp_text = _get_sp(args.system_prompt_id)
    print(f"[bulk] system prompt: {args.system_prompt_id} ({len(sp_text)} chars)")

    # Wrap in PEFT once with first adapter, then load+set further adapters.
    # Special sentinel: --adapters none → run the base model directly with no LoRA.
    if args.adapters == ["none"]:
        print("[bulk] no adapter mode — base model directly")
        model = base
    else:
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

        if args.adapters != ["none"]:
            model.set_adapter(nm)
        print(f"\n[bulk] === {nm} ({len(todo)} prompts) ===", flush=True)

        bs = max(1, args.batch_size)
        done = 0
        for bstart in range(0, len(todo), bs):
            batch_prompts = todo[bstart : bstart + bs]
            t0 = time.time()
            texts = []
            for p in batch_prompts:
                msgs = [{"role":"system","content":sp_text},{"role":"user","content":p}]
                texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
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
            except Exception as e:
                print(f"  ERR batch {bstart}: {e}")
                continue
            dt = time.time() - t0
            for k, p in enumerate(batch_prompts):
                new_ids = out[k][prompt_len:]
                raw = tok.decode(new_ids, skip_special_tokens=True)
                code = strip_fences(raw)
                rec = {
                    "ts": time.time(),
                    "side": nm,
                    "model": f"{args.base}+adapter:{nm}",
                    "prompt": p,
                    "code": code,
                    "raw_len": len(raw),
                    "temp": args.temp,
                    "system_prompt_id": args.system_prompt_id,
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                done += 1
                print(f"  [{done}/{len(todo)}] {p!r} -> {len(code)} chars ({dt/len(batch_prompts):.1f}s avg)", flush=True)

        print(f"[bulk] wrote {out_path.name}")

    elapsed = (time.time() - t_start) / 60
    print(f"\n[bulk] DONE in {elapsed:.1f} min")
    return 0


if __name__ == "__main__":
    rc = main()
    # PyTorch ROCm Windows can hang in CUDA context teardown. Force-exit so the
    # next pipeline step can run.
    import os
    os._exit(rc)
