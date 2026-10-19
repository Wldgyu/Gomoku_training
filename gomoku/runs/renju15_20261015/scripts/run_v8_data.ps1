# v8 data: refreshed human positions (all games), long-search labels for the tails of games the human won, Rapfi DAgger on v7
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..\..")   # repository root; create gomoku\logs first
$env:PYTHONPATH = "."
$data = "gomoku/runs/renju15_20261015/data"
$m = "gomoku/runs/renju15_20261015/models"
New-Item -ItemType Directory -Force "$data/before_v8" | Out-Null
Copy-Item "$data/human_train.npz", "$data/humanr_train.npz" "$data/before_v8/" -Force
.venv\Scripts\python.exe -m gomoku.renju.analyze_games gomoku\기보저장 --positions "$data/human_train.npz" *> gomoku\logs\renju_v8_human.log
.venv\Scripts\python.exe -u -m gomoku.renju.dagger human --data $data --folder gomoku/기보저장 --prefix humanr --label-ms 150 --lost-last 14 *> gomoku\logs\renju_v8_humanr.log
.venv\Scripts\python.exe -u -m gomoku.renju.dagger human --data $data --folder gomoku/기보저장 --prefix humanlost --label-ms 1500 --lost-last 14 --lost-only *> gomoku\logs\renju_v8_humanlost.log
.venv\Scripts\python.exe -u -m gomoku.renju.dagger collect --models "$m/fly2950_v7.pt" --data $data --games 720 --workers 6 --opp-ms 60 100 --betas 0 0.3 0.6 --blunders 0 0.25 0.5 --hunt 0 0.5 1 --relabel-ms 1000 --prefix rapfi4 --seed 885000001 *> gomoku\logs\renju_v8_dagger.log
"done" | Out-File gomoku\logs\renju_v8_data.done
