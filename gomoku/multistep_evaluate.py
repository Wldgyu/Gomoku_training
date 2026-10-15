"""Frozen equal-budget baseline versus mined multistep pilot, with expansion gate."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .advanced_match_audit import audit
from .history_aux import HistoryTacticalData
from .models import load_policy
from .train import atomic_json

def aggregate(rows):
    games=sum(r['games'] for r in rows)
    wins=sum(r['wins'] for r in rows)
    positions=sum(r['tactical_decisions']['block']['positions'] for r in rows)
    omissions=sum(r['defense_omissions'] for r in rows)
    attacks=sum(r['tactical_decisions']['create_fork']['positions'] for r in rows)
    success=sum(r['tactical_decisions']['create_fork']['correct'] for r in rows)
    return dict(games=games,wins=wins,win_rate=wins/games,
                defense_omissions=omissions,block_opportunities=positions,
                defense_omission_rate=omissions/positions if positions else None,
                multiple_threat_game_rate=sum(r['games_reaching_multiple_threat'] for r in rows)/games,
                attack_fork_opportunities=attacks,attack_fork_successes=success,
                attack_fork_success_rate=success/attacks if attacks else None,
                attack_fork_successes_per_game=success/games,
                actual_safe_forks_created=sum(r["actual_safe_forks_created"] for r in rows),
                actual_safe_forks_per_game=sum(r["actual_safe_forks_created"] for r in rows)/games)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--seed',type=int,default=4700)
    p.add_argument('--games',type=int,default=256)
    a=p.parse_args()
    torch.set_num_threads(4)
    rows=[]
    tactical=[]
    for arm in ('baseline','multistep'):
        for kind in ('fly','gru'):
            path=a.input/arm/f'{kind}_seed{a.seed}.pt'
            model,saved=load_policy(path,torch.device('cpu'))
            if saved['completed_updates']!=saved['config']['updates']:
                raise ValueError('Last budget checkpoint required')
            for split,envseed in (('validation',140_000_470),('test',150_000_470)):
                result,replays=audit(model,a.seed,a.games,True,envseed)
                row=dict(arm=arm,split=split,**result)
                rows.append(row)
                data=HistoryTacticalData(a.input/'data'/f'{split}.npz',torch.device('cpu'))
                tactical.append(dict(arm=arm,kind=kind,split=split,
                                     cold=data.evaluate(model),history=data.evaluate(model,history=True)))
                atomic_json(a.input/f'{arm}_{kind}_{split}_replays.json',dict(games=replays))
                atomic_json(a.input/'evaluation_progress.json',dict(runs=rows,tactical=tactical))
                print(json.dumps(row),flush=True)
    summary={}
    for split in ('validation','test'):
        b=aggregate([r for r in rows if r['arm']=='baseline' and r['split']==split])
        m=aggregate([r for r in rows if r['arm']=='multistep' and r['split']==split])
        no_model_decline=all(next(r['wins'] for r in rows if r['arm']=='multistep' and r['split']==split and r['kind']==k)>=
                            next(r['wins'] for r in rows if r['arm']=='baseline' and r['split']==split and r['kind']==k) for k in ('fly','gru'))
        passed=(m['win_rate']>=b['win_rate']+.05 and no_model_decline
                and m['defense_omission_rate']<b['defense_omission_rate']
                and m['multiple_threat_game_rate']<b['multiple_threat_game_rate']
                and m['actual_safe_forks_per_game']>=b['actual_safe_forks_per_game'])
        summary[split]=dict(baseline=b,multistep=m,expansion_gate_passed=passed)
    atomic_json(a.input/'evaluation.json',dict(runs=rows,tactical=tactical,summary=summary,
        expand_four_models_three_seeds=all(s['expansion_gate_passed'] for s in summary.values()),
        gate='>=5 percentage point aggregate win improvement; neither model declines; lower defense omission rate and multiple-threat game rate; no decline in actual safe forks created per game, in both validation and test',
        seed=a.seed,interpretation='Exploratory single training seed; all actual moves are neural argmax with occupancy mask only. Evaluation game banks and all their positions were reserved before training.'))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':
    main()
