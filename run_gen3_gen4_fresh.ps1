$env:VIRTUAL_ENV = 'C:\Users\Oli\Documents\MightyFineTuna\.venv'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'

$ROOT = 'C:\Users\Oli\Documents\MightyFineTuna'
$PY = "$env:VIRTUAL_ENV\Scripts\python.exe"

New-Item -ItemType Directory -Path "$ROOT\sides_fresh" -Force | Out-Null

# 2 temps per model — keep total round count manageable for rating
$temps = @(0.5, 0.7)

# gen3: 3b-dpo-v3-2ep adapter on 3b-dpo-v2-2ep-merged base
foreach ($t in $temps) {
  $tag = "{0:00}" -f ($t * 10)
  Write-Output "=== gen3 fresh temp=$t at $(Get-Date -Format 'HH:mm:ss') ==="
  & $PY -u "$ROOT\bulk_infer.py" `
    --base D:\dpo_models\3b-dpo-v2-2ep-merged `
    --adapters checkpoints/3b-dpo-v3-2ep `
    --prompts $ROOT\eval_prompts_v3.txt `
    --out-dir $ROOT\sides_fresh `
    --out-prefix "side_gen3_t${tag}_" `
    --temp $t `
    --system-prompt-id F `
    --max-tokens 1200 `
    --batch-size 8
  Write-Output "gen3 t=$t exit=$LASTEXITCODE"
}

# gen4: 3b-dpo-v4-2ep adapter on 3b-dpo-v3-2ep-merged base
foreach ($t in $temps) {
  $tag = "{0:00}" -f ($t * 10)
  Write-Output "=== gen4 fresh temp=$t at $(Get-Date -Format 'HH:mm:ss') ==="
  & $PY -u "$ROOT\bulk_infer.py" `
    --base D:\dpo_models\3b-dpo-v3-2ep-merged `
    --adapters checkpoints/3b-dpo-v4-2ep `
    --prompts $ROOT\eval_prompts_v3.txt `
    --out-dir $ROOT\sides_fresh `
    --out-prefix "side_gen4_t${tag}_" `
    --temp $t `
    --system-prompt-id F `
    --max-tokens 1200 `
    --batch-size 8
  Write-Output "gen4 t=$t exit=$LASTEXITCODE"
}

Write-Output "=== GEN3+GEN4 FRESH DONE at $(Get-Date -Format 'HH:mm:ss') ==="
