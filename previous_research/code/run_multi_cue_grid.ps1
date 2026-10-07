param([switch]$DryRun)

$ErrorActionPreference = "Stop"
$py = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) ".venv\Scripts\python.exe"
$rates = @("rnn=0.0003", "gru=0.01", "leaky=0.003", "fly=0.003", "rewired=0.003")
# worker k handles the conditions listed for it: cues, delay, distractors
$workers = @(
  @(@(2,8,2), @(2,4,2), @(4,8,2)),
  @(@(1,8,2), @(2,16,2), @(2,8,0)),
  @(@(3,8,2), @(2,8,6), @(1,32,6))
)
$jobs = foreach ($w in $workers) {
  Start-Job -ArgumentList $py, $rates, $w, (Split-Path $PSScriptRoot -Parent), ([bool]$DryRun) -ScriptBlock {
    param($py, $rates, $conditions, $cwd, $dryRun)
    $ErrorActionPreference = "Stop"
    Set-Location $cwd
    foreach ($c in $conditions) {
      if ($c.Count -ne 3 -or ($c | Where-Object { $_ -isnot [int] })) {
        throw "Expected three scalar integers: cues, delay, distractors."
      }
      if ($dryRun) {
        [pscustomobject]@{ cues = $c[0]; delay = $c[1]; distractors = $c[2] } | ConvertTo-Json -Compress
        continue
      }
      & $py code\multi_cue_memory.py --cues-min $c[0] --cues-max $c[0] --delay $c[1] --distractors $c[2] `
        --seeds 1101 1102 1103 --updates 12000 --eval-every 500 --rates @rates `
        --output-dir data\multi_cue\final --skip-existing
      if ($LASTEXITCODE -ne 0) {
        throw "Training failed (exit $LASTEXITCODE) for condition $($c -join ',')."
      }
    }
  }
}
try {
  $jobs | Wait-Job | Receive-Job -ErrorAction Stop
} finally {
  $jobs | Stop-Job
  $jobs | Remove-Job
}
