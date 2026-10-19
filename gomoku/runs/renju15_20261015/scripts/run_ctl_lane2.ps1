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
}# lane 2 (runs next to lane 1): real Fly a, real Fly b, RNN b
Train "fly" "ctl_fly2950_a" 9101 0
Train "fly" "ctl_fly2950_b" 9102 0
Train "rnn" "ctl_rnn3000_b" 9102 0
"done" | Out-File gomoku\logs\renju_ctl_lane2.done
