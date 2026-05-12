"""Component smoke tests for .venv-fast. Writes to setup_test_report.txt."""
import sys
import traceback

results = []

def section(name):
    results.append(f"\n=== {name} ===")
    print(f"\n=== {name} ===")

def ok(msg):
    results.append(f"PASS: {msg}")
    print(f"PASS: {msg}")

def fail(msg, exc=None):
    err = f"FAIL: {msg}"
    if exc:
        err += f"\n  {type(exc).__name__}: {exc}"
    results.append(err)
    print(err)

# 1. torch
section("torch")
try:
    import torch
    ok(f"import torch -> {torch.__version__}")
    cuda = torch.cuda.is_available()
    ok(f"torch.cuda.is_available() -> {cuda}")
    if cuda:
        ok(f"device name: {torch.cuda.get_device_name(0)}")
        # bf16 capability test
        x = torch.randn(4, 4, device='cuda', dtype=torch.bfloat16)
        y = (x @ x.T).cpu()
        ok(f"bf16 matmul output shape {tuple(y.shape)}, dtype {y.dtype}")
except Exception as e:
    fail("torch", e)
    traceback.print_exc()

# 2. triton
section("triton")
try:
    import triton
    ok(f"import triton -> {triton.__version__}")
except Exception as e:
    fail("triton", e)
    traceback.print_exc()

# 3. bitsandbytes
section("bitsandbytes")
try:
    import bitsandbytes as bnb
    ok(f"import bitsandbytes -> {bnb.__version__ if hasattr(bnb,'__version__') else 'no __version__'}")
    opt = bnb.optim.AdamW8bit
    ok(f"AdamW8bit class loaded -> {opt}")
except Exception as e:
    fail("bitsandbytes", e)
    traceback.print_exc()

# 4. flash_attn
section("flash_attn")
try:
    from flash_attn import flash_attn_func
    import torch as _t
    if _t.cuda.is_available():
        q = _t.randn(1, 16, 8, 64, device='cuda', dtype=_t.bfloat16)
        k = q.clone()
        v = q.clone()
        out = flash_attn_func(q, k, v)
        ok(f"flash_attn_func output shape: {tuple(out.shape)}")
    else:
        fail("flash_attn: no cuda")
except Exception as e:
    fail("flash_attn", e)
    traceback.print_exc()

# 5. liger
section("liger")
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_qwen2()
    ok("apply_liger_kernel_to_qwen2 ran without error")
except Exception as e:
    fail("liger", e)
    traceback.print_exc()

# 6. sageattention
section("sageattention")
try:
    from sageattention import sageattn
    ok(f"sageattn callable: {sageattn}")
except Exception as e:
    fail("sageattention", e)
    traceback.print_exc()

# Write report incrementally
with open("setup_test_report_components.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print("\n--- DONE ---")
