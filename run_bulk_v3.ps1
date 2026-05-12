$env:VIRTUAL_ENV = 'C:\Users\Oli\Documents\MightyFineTuna\.venv'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'

$ROOT = 'C:\Users\Oli\Documents\MightyFineTuna'
$PY = "$env:VIRTUAL_ENV\Scripts\python.exe"

# Two generation rounds (different random seeds) × 4 temps each = 8 side files.
$rounds = @('a', 'b')
$temps = @(0.5, 0.7, 0.9, 1.1)
foreach ($rnd in $rounds) {
  foreach ($t in $temps) {
    $tag = "{0:00}" -f ($t * 10)
    Write-Output "=== bulk_infer round=$rnd temp=$t starting at $(Get-Date -Format 'HH:mm:ss') ==="
    & $PY -u "$ROOT\bulk_infer.py" `
      --base checkpoints/3b-dpo-merged-v2 `
      --adapters checkpoints/3b-dpo-v2-2ep `
      --prompts $ROOT\eval_prompts_v2.txt `
      --out-dir $ROOT\sides_v3 `
      --out-prefix "side_v3${rnd}_t${tag}_" `
      --temp $t `
      --system-prompt-id F `
      --max-tokens 1200 `
      --batch-size 8
    $code = $LASTEXITCODE
    Write-Output "exit code: $code"
    if ($code -ne 0 -and $code -ne -1) {
      Write-Output "FATAL round=$rnd temp=$t with exit $code"
      exit $code
    }
  }
}
Write-Output "=== ALL 8 RUNS DONE at $(Get-Date -Format 'HH:mm:ss') ==="
