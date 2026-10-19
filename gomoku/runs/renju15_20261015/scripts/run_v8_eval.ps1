# v8 evaluation after training finished: weak Rapfi (2 seeds x 360 games) per model, then league
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..\..")   # repository root; create gomoku\logs first
$env:PYTHONPATH = "."
$run = "gomoku/runs/renju15_20261015"
$m = "$run/models"
while (-not (Test-Path gomoku\logs\renju_v8_train.done)) { Start-Sleep 20 }
foreach ($name in @("fly2950_v6a", "fly2950_v7", "fly2950_v8a", "fly2950_v8b")) {
    foreach ($seed in @(886000001, 886000002)) {
        "$name seed $seed" | Out-File -Append gomoku\logs\renju_v8_weak.log -Encoding utf8
        .venv\Scripts\python.exe -u -m gomoku.renju.dagger eval --models "$m/$name.pt" --data "$run/data" --games 360 --workers 6 --opp-ms 60 --blunders 0.5 --seed $seed 2>&1 | Out-File -Append gomoku\logs\renju_v8_weak.log -Encoding utf8
    }
}
.venv\Scripts\python.exe -u -m gomoku.renju.league --models "$m/fly2950_v6a.pt" "$m/fly2950_v7.pt" "$m/fly2950_v8a.pt" "$m/fly2950_v8b.pt" "$m/rnn3000_v6a.pt" --out "$run/league_v8.json" *> gomoku\logs\renju_league_v8.log
"done" | Out-File gomoku\logs\renju_v8_eval.done
