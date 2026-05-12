$env:VIRTUAL_ENV = 'C:\Users\Oli\Documents\MightyFineTuna\.venv'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'

$ROOT = 'C:\Users\Oli\Documents\MightyFineTuna'
$PY = "$env:VIRTUAL_ENV\Scripts\python.exe"

New-Item -ItemType Directory -Path "$ROOT\sides_gen01" -Force | Out-Null

$temps = @(0.5, 0.7, 0.9)

# gen0: parent_b_v2 adapter on plain Qwen base
foreach ($t in $temps) {
  $tag = "{0:00}" -f ($t * 10)
  Write-Output "=== gen0 temp=$t at $(Get-Date -Format 'HH:mm:ss') ==="
  & $PY -u "$ROOT\bulk_infer.py" `
    --base Qwen/Qwen2.5-Coder-3B-Instruct `
    --adapters checkpoints/3b-parent_b_v2 `
    --prompts $ROOT\eval_prompts_v2.txt `
    --out-dir $ROOT\sides_gen01 `
    --out-prefix "side_gen0_t${tag}_" `
    --temp $t `
    --system-prompt-id F `
    --max-tokens 1200 `
    --batch-size 8
  Write-Output "gen0 t=$t exit=$LASTEXITCODE"
}

# gen1: 3b-dpo-merged-v2 base directly (no adapter on top = just first DPO round)
foreach ($t in $temps) {
  $tag = "{0:00}" -f ($t * 10)
  Write-Output "=== gen1 temp=$t at $(Get-Date -Format 'HH:mm:ss') ==="
  & $PY -u "$ROOT\bulk_infer.py" `
    --base checkpoints/3b-dpo-merged-v2 `
    --adapters none `
    --prompts $ROOT\eval_prompts_v2.txt `
    --out-dir $ROOT\sides_gen01 `
    --out-prefix "side_gen1_t${tag}_" `
    --temp $t `
    --system-prompt-id F `
    --max-tokens 1200 `
    --batch-size 8
  Write-Output "gen1 t=$t exit=$LASTEXITCODE"
}

Write-Output "=== GEN0+GEN1 DONE at $(Get-Date -Format 'HH:mm:ss') ==="
