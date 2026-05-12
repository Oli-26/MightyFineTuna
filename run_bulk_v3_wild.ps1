$env:VIRTUAL_ENV = 'C:\Users\Oli\Documents\MightyFineTuna\.venv'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'

$ROOT = 'C:\Users\Oli\Documents\MightyFineTuna'
$PY = "$env:VIRTUAL_ENV\Scripts\python.exe"

Write-Output "=== bulk_infer WILDCARD temp=0.7 max_tokens=4000 at $(Get-Date -Format 'HH:mm:ss') ==="
& $PY -u "$ROOT\bulk_infer.py" `
  --base checkpoints/3b-dpo-merged-v2 `
  --adapters checkpoints/3b-dpo-v2-2ep `
  --prompts $ROOT\eval_prompts_v2.txt `
  --out-dir $ROOT\sides_v3 `
  --out-prefix "side_v3_wild_t07_" `
  --temp 0.7 `
  --system-prompt-id F `
  --max-tokens 4000 `
  --batch-size 8
$code = $LASTEXITCODE
Write-Output "wildcard exit code: $code"
Write-Output "=== WILDCARD DONE at $(Get-Date -Format 'HH:mm:ss') ==="
