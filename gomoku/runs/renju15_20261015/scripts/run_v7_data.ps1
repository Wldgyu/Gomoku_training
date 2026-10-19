# v7 data: deep chains, black-side forbidden-block defence, extra white-side forbidden-block chains (CPU jobs, one after another)
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..\..")   # repository root; create gomoku\logs first
$env:PYTHONPATH = "."
$data = "gomoku/runs/renju15_20261015/data"
.venv\Scripts\python.exe -u -m gomoku.renju.deep deep --data $data --train 3000 --test 300 --workers 10 --seconds 1500 *> gomoku\logs\renju_v7_deep.log
.venv\Scripts\python.exe -u -m gomoku.renju.deep forbid_black --data $data --train 1500 --test 200 --workers 10 --seconds 1500 --seed 992000001 *> gomoku\logs\renju_v7_forbid_black.log
.venv\Scripts\python.exe -u -m gomoku.renju.forbid --data $data --train 1000 --test 50 --workers 10 --seconds 1500 --seed 993000001 --prefix forbid2 *> gomoku\logs\renju_v7_forbid2.log
"done" | Out-File gomoku\logs\renju_v7_data.done
