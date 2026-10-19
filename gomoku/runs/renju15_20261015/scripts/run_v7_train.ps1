# Fly v7: two seeds from v6a, one GPU job at a time. Baseline eval of v6a/v5 on the new held-out suites first.
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..\..")   # repository root; create gomoku\logs first
$env:PYTHONPATH = "."
$run = "gomoku/runs/renju15_20261015"
$m = "$run/models"
$extra = @("quiet_train", "selfplay_train", "onpolicy_train", "human_train", "humanr_train", "rapfi_train", "rapfi2_train",
           "forbid_train", "forbid2_train", "deep_train", "forbid_black_train", "rapfi3_train")
.venv\Scripts\python.exe -u gomoku\logs\eval_suites.py "$run/data" "$m/fly2950_v5.pt" "$m/fly2950_v6a.pt" "$m/rnn3000_v6a.pt" *> gomoku\logs\renju_v7_baseline.log
foreach ($pair in @(@("v7a", 7401), @("v7b", 7402))) {
    $tag = $pair[0]; $seed = $pair[1]
    .venv\Scripts\python.exe -u -m gomoku.renju.train --kind fly --graph-neurons 2950 --data "$run/data" --name "fly2950_$tag" --out $m `
        --init "$m/fly2950_v6a.pt" --lr 3e-4 --updates 5000 --seed $seed --extra @extra *> "gomoku\logs\renju_fly2950_$tag.log"
}
"done" | Out-File gomoku\logs\renju_v7_train.done
