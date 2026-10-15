"""Frozen full-game transfer and exact-depth completion evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .env import BatchBoard,opponent_actions
from .history_aux import burn_in,HistoryTacticalData
from .models import load_policy
from .midgame_data import read_records
from .advanced_match_audit import audit
from .train import atomic_json

@torch.no_grad()
def middle_games(model,path,seed,device='cpu'):
    with np.load(path) as d:
        histories=d['histories'].copy()
        target=d['labels'].copy()
        horizon=d['horizon'].copy()
        family=d['family'].copy()
    count=len(histories)
    lengths=(histories>=0).sum(1)
    colors=np.where(lengths%2==0,1,-1).astype(np.int8)
    board=BatchBoard(count,12)
    for t in range(int(lengths.max())):
        active=t<lengths
        board.place(np.where(active,histories[:,t],0),1 if t%2==0 else -1,active)
    torch_device=torch.device(device)
    model=model.to(torch_device)
    state=burn_in(model,torch.from_numpy(histories.astype(np.int64)).to(torch_device))
    rng=np.random.default_rng(seed)
    wins=np.zeros(count,dtype=bool)
    decisions=np.zeros(count,dtype=np.int32)
    first_correct=np.zeros(count,dtype=bool)
    finish_turn=np.zeros(count,dtype=np.int32)
    matrix=model.matrix()
    zeros=torch.zeros(count,dtype=torch.bool,device=torch_device)
    replays=[moves[moves>=0].tolist() for moves in histories]
    for t in range(73):
        active=~board.finished
        if not active.any():
            break
        obs=torch.from_numpy(board.observe(colors)).to(torch_device)
        legal=torch.from_numpy(board.legal()).to(torch_device)
        logits,_,state=model.step(obs,state,zeros,matrix)
        action=logits.masked_fill(~legal,-1e9).argmax(1).cpu().numpy()
        if t==0:
            first_correct=target[np.arange(count),action]
        decisions[active]+=1
        for row in np.flatnonzero(active):
            replays[row].append(int(action[row]))
        board.place(action,colors,active)
        reply_active=~board.finished
        reply=opponent_actions(board,-colors,rng,'tactical')
        for row in np.flatnonzero(reply_active):
            replays[row].append(int(reply[row]))
        board.place(reply,-colors,reply_active)
        newly=board.finished & (finish_turn==0)
        finish_turn[newly]=decisions[newly]
    assert board.finished.all()
    wins=board.winner==colors
    results=[]
    for depth in (1,2,3):
        for name in ['all']+sorted(set(family[horizon==depth].tolist())):
            mask=(horizon==depth)&((family==name) if name!='all' else True)
            n=int(mask.sum())
            results.append(dict(horizon=depth,family=name,games=n,first_correct=int(first_correct[mask].sum()),
                                wins=int(wins[mask].sum()),within_horizon_wins=int((wins&(decisions<=horizon)&mask).sum()),
                                draws=int(((board.winner==0)&mask).sum()),
                                win_rate=float(wins[mask].mean()),
                                within_horizon_win_rate=float((wins&(decisions<=horizon))[mask].mean())))
    games=[dict(size=12,a=model.kind,b='tactical',a_color=int(colors[i]),winner=int(board.winner[i]),
                moves=replays[i],horizon=int(horizon[i]),family=str(family[i]),
                first_correct=bool(first_correct[i]),decisions=int(decisions[i])) for i in range(count)]
    return results,games

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arms',nargs='+',required=True)
    p.add_argument('--seed',type=int,default=4810)
    p.add_argument('--splits',nargs='+',choices=['validation','test'],default=['validation'])
    a=p.parse_args()
    torch.set_num_threads(4)
    output=a.root/'evaluation.json'
    result=json.loads(output.read_text(encoding='utf-8')) if output.exists() else dict(full_games=[],midgames=[],tactical=[])
    for arm in a.arms:
        for kind in ('fly','gru'):
            model,saved=load_policy(a.root/arm/f'{kind}_seed{a.seed}.pt',torch.device('cpu'))
            assert saved['completed_updates']==saved['config']['updates']
            for split in a.splits:
                if any(r['arm']==arm and r['kind']==kind and r['split']==split for r in result['full_games']):
                    continue
                full_seed=210_000_481 if split=='validation' else 211_000_481
                full,replays=audit(model,a.seed,256,True,full_seed)
                result['full_games'].append(dict(arm=arm,split=split,**full))
                atomic_json(a.root/f'{arm}_{kind}_{split}_full_replays.json',dict(games=replays))
                middle,games=middle_games(model,a.root/'data'/f'{split}_starts.npz',full_seed+1_000_000)
                result['midgames'].extend(dict(arm=arm,kind=kind,split=split,**r) for r in middle)
                atomic_json(a.root/f'{arm}_{kind}_{split}_mid_replays.json',dict(games=games))
                data=HistoryTacticalData(a.root/'data'/f'{split}_starts.npz',torch.device('cpu'))
                result['tactical'].append(dict(arm=arm,kind=kind,split=split,
                                              cold=data.evaluate(model),history=data.evaluate(model,True)))
                atomic_json(output,result)
                print(json.dumps(dict(arm=arm,kind=kind,split=split,full_wins=full['wins'],
                                      full_block_omissions=full['defense_omissions'],
                                      midgames=[r for r in middle if r['family']=='all'])),flush=True)
if __name__=='__main__':
    main()
