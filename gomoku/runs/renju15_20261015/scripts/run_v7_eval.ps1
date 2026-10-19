# v7 evaluation: league (v5/v6a/v7a/v7b Fly + RNN v6a), weak-Rapfi games with two eval seeds (360 games each)
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..\..")   # repository root; create gomoku\logs first
$env:PYTHONPATH = "."
$run = "gomoku/runs/renju15_20261015"
$m = "$run/models"
.venv\Scripts\python.exe -u -m gomoku.renju.league --models "$m/fly2950_v5.pt" "$m/fly2950_v6a.pt" "$m/fly2950_v7a.pt" "$m/fly2950_v7b.pt" "$m/rnn3000_v6a.pt" --out "$run/league_v7.json" *> gomoku\logs\renju_league_v7.log
foreach ($name in @("fly2950_v5", "fly2950_v6a", "fly2950_v7a", "fly2950_v7b")) {
    foreach ($seed in @(880000001, 880000777)) {
        "== $name seed $seed" | Out-File -Append gomoku\logs\renju_v7_weak.log
        .venv\Scripts\python.exe -u -m gomoku.renju.dagger eval --models "$m/$name.pt" --data "$run/data" --games 360 --workers 6 --opp-ms 60 --blunders 0.5 --seed $seed 2>&1 | Out-File -Append gomoku\logs\renju_v7_weak.log
    }
}
"done" | Out-File gomoku\logs\renju_v7_eval.done
