$env:VIRTUAL_ENV = 'C:\Users\Oli\Documents\MightyFineTuna\.venv'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'

$ROOT = 'C:\Users\Oli\Documents\MightyFineTuna'
$PY = "$env:VIRTUAL_ENV\Scripts\python.exe"

$temps = @(0.5, 0.7, 0.9, 1.1)
foreach ($t in $temps) {
  $tag = "{0:00}" -f ($t * 10)
  Write-Output "=== bulk_infer temp=$t (tag=t$tag) starting at $(Get-Date -Format 'HH:mm:ss') ==="
  # Run python in a child process. If it returns with exit -1 (we kill it after
  # stuck CUDA cleanup), still continue to next temp instead of aborting.
  & $PY -u "$ROOT\bulk_infer.py" `
    --base checkpoints/3b-dpo-merged-v2 `
    --adapters checkpoints/3b-dpo-v2-2ep `
    --prompts $ROOT\eval_prompts_v2.txt `
    --out-dir $ROOT\sides_v2 `
    --out-prefix "side_v2_t${tag}_" `
    --temp $t `
    --system-prompt-id F `
    --max-tokens 2500 `
    --batch-size 8
  $code = $LASTEXITCODE
  Write-Output "exit code: $code"
  if ($code -ne 0 -and $code -ne -1) {
    Write-Output "FATAL temp=$t with exit $code (non-recoverable)"
    exit $code
  }
}
Write-Output "=== ALL 4 RUNS DONE at $(Get-Date -Format 'HH:mm:ss') ==="
