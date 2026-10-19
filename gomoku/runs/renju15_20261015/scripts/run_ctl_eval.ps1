# Evaluation of the controlled comparison, starts when all six trainings are done
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..\..")   # repository root; create gomoku\logs first
$env:PYTHONPATH = "."
$run = "gomoku/runs/renju15_20261015"
$m = "$run/models"
while (-not (Test-Path gomoku\logs\renju_ctl_all.done)) { Start-Sleep 30 }
$ctl = @("ctl_rewired2950_a", "ctl_rewired2950_b", "ctl_fly2950_a", "ctl_fly2950_b", "ctl_rnn3000_a", "ctl_rnn3000_b")
.venv\Scripts\python.exe -u gomoku\logs\eval_suites.py "$run/data" ($ctl | ForEach-Object { "$m/$_.pt" }) "$m/fly2950_v7.pt" "$m/rnn3000_v6a.pt" *> gomoku\logs\renju_ctl_static.log
foreach ($name in ($ctl + @("rnn3000_v6a"))) {
    foreach ($seed in @(886000001, 886000002)) {
        "$name seed $seed" | Out-File -Append gomoku\logs\renju_ctl_weak.log -Encoding utf8
        .venv\Scripts\python.exe -u -m gomoku.renju.dagger eval --models "$m/$name.pt" --data "$run/data" --games 360 --workers 6 --opp-ms 60 --blunders 0.5 --seed $seed 2>&1 | Out-File -Append gomoku\logs\renju_ctl_weak.log -Encoding utf8
    }
}
.venv\Scripts\python.exe -u -m gomoku.renju.league --models ($ctl | ForEach-Object { "$m/$_.pt" }) "$m/fly2950_v7.pt" "$m/rnn3000_v6a.pt" --out "$run/league_ctl.json" *> gomoku\logs\renju_league_ctl.log
"done" | Out-File gomoku\logs\renju_ctl_eval.done

