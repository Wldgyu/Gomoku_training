Set-Location (Resolve-Path "$PSScriptRoot\..\..\..\..")   # repository root; create gomoku\logs first
$env:PYTHONPATH = "."
$run = "gomoku/runs/renju15_20261015"
$m = "$run/models"
$extra = @("quiet_train", "selfplay_train", "onpolicy_train", "human_train", "humanr_train", "humanlost_train", "rapfi_train", "rapfi2_train",
           "rapfi3_train", "rapfi4_train", "forbid_train", "forbid2_train", "deep_train", "forbid_black_train")
function Train($kind, $name, $seed, $rw) {
    $size = if ($kind -eq "rnn") { @("--hidden", "3000") } else { @("--graph-neurons", "2950") }
    .venv\Scripts\python.exe -u -m gomoku.renju.train --kind $kind @size --rewire-seed $rw --data "$run/data" --name $name --out $m `
        --lr 1e-3 --updates 12000 --seed $seed --extra @extra *> "gomoku\logs\renju_$name.log"
}# lane 1: rewired a is already running under run_ctl_train.ps1; wait for it, then rewired b, RNN a
while (-not ((Test-Path "$m/ctl_rewired2950_a.pt") -and ((Get-Item "$m/ctl_rewired2950_a.pt").Length -gt 1000000))) { Start-Sleep 20 }
Train "rewired" "ctl_rewired2950_b" 9102 1
Train "rnn" "ctl_rnn3000_a" 9101 0
while (-not (Test-Path gomoku\logs\renju_ctl_lane2.done)) { Start-Sleep 30 }
"done" | Out-File gomoku\logs\renju_ctl_all.done
