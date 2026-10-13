"""Data, budget and observed-game overlap audits for the ordered curriculum."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .env import BatchBoard
from .midgame_data import read_records,all_prefix_keys,canonical,attack
from .train import atomic_json

def load(path):
    with np.load(path) as d:
        return {k:d[k].copy() for k in d.files}

def audit_data(root,phase):
    holdout=read_records(root/'data/validation_starts.npz')+read_records(root/'data/test_starts.npz')
    reserved=all_prefix_keys(holdout)
    rows=[]
    names=['baseline_aux','step1_starts','validation_starts','test_starts']
    if phase=='step2':
        names += ['step2_aux','step2_starts']
    for name in names:
        d=load(root/'data'/f'{name}.npz')
        histories=d['histories']
        lengths=(histories>=0).sum(1)
        colors=np.where(lengths%2==0,1,-1).astype(np.int8)
        board=BatchBoard(len(histories),12)
        for t in range(int(lengths.max())):
            active=t<lengths
            board.place(np.where(active,histories[:,t],0),1 if t%2==0 else -1,active)
        np.testing.assert_array_equal(board.observe(colors),d['obs'])
        assert not board.finished.any(),name
        assert d['labels'].any(1).all(),name
        assert not (d['labels'] & ~board.legal()).any(),name
        assert len(set(d['keys']))==len(d['keys']),name
        keys={canonical(c,int(cor)) for c,cor in zip(board.board,colors)}
        if name not in ('validation_starts','test_starts'):
            assert not keys & reserved,name
        if 'horizon' in d:
            certified=np.array([attack(c,int(cor))[0] for c,cor in zip(board.board,colors)])
            np.testing.assert_array_equal(certified,d['horizon'])
            for depth in (1,2,3):
                assert ((d['horizon']==depth)&(colors==1)).any()
                assert ((d['horizon']==depth)&(colors==-1)).any()
        rows.append(dict(name=name,positions=len(histories),legal_nonempty_targets=True,
                         exact_history_reconstruction=True,nonterminal=True,
                         symmetry_normalized_holdout_overlap=0 if name not in ('validation_starts','test_starts') else None,
                         categories={str(c):int((d['category']==c).sum()) for c in sorted(set(d['category'].tolist()))},
                         sha256=hashlib.sha256((root/'data'/f'{name}.npz').read_bytes()).hexdigest()))
    atomic_json(root/f'data_audit_{phase}.json',dict(datasets=rows,all_holdout_prefixes_reserved=len(reserved)))

def audit_runs(root):
    rows=[]
    for arm in ('baseline','step1','step2'):
        for kind in ('fly','gru'):
            path=root/arm/f'{kind}_seed4810.json'
            if not path.exists():
                continue
            d=json.loads(path.read_text(encoding='utf-8'))
            c=d['config']
            assert d['completed_updates']==c['updates']==300
            assert c['warmup_updates']==600 and c['epochs']==2 and c['auxiliary_batch']==128
            assert c['envs']==32 and c['rollout']==16
            samples=d['auxiliary_samples_by_category']
            assert sum(samples.values())==153600
            if arm=='step2':
                assert c['balanced_auxiliary']
                assert all(.15<int(n)/153600<.19 for n in samples.values())
            rows.append(dict(arm=arm,kind=kind,auxiliary_examples=sum(samples.values()),
                             learner_decisions=d['learner_decisions'],samples_by_category=samples,
                             midgame_episode_outcomes=d.get('midgame_outcomes',{}),
                             episode_starts=d.get('episode_starts',{}),
                             prefix_frames=d.get('rollout_prefix_frames',0)))
    overlaps=[]
    trainsets={}
    for arm,name in (('baseline','baseline_aux'),('step1','baseline_aux'),('step2','step2_aux')):
        path=root/'data'/f'{name}.npz'
        if not path.exists():
            continue
        d=load(path)
        trainsets[arm]={canonical(((o[0]-o[1])*(1 if o[3,0,0] else -1)).reshape(-1).astype(np.int8),
                                 1 if o[3,0,0] else -1) for o in d['obs']}
        pool=root/'data'/f'{arm}_starts.npz'
        if pool.exists():
            d=load(pool)
            trainsets[arm] |= {canonical(((o[0]-o[1])*(1 if o[3,0,0] else -1)).reshape(-1).astype(np.int8),
                                 1 if o[3,0,0] else -1) for o in d['obs']}
    for file in root.glob('*_*_*_*_replays.json'):
        games=json.loads(file.read_text(encoding='utf-8'))['games']
        keys=set()
        for game in games:
            board=BatchBoard(1,12)
            for t,action in enumerate(game['moves']):
                color=1 if t%2==0 else -1
                keys.add(canonical(board.board[0],color))
                board.place(np.array([action]),color)
            assert board.finished[0] and int(board.winner[0])==game['winner']
        row=dict(file=file.name,games=len(games),overlap={arm:len(keys&ks) for arm,ks in trainsets.items()})
        overlaps.append(row)
    atomic_json(root/'run_audit.json',dict(runs=rows,evaluation_game_overlap=overlaps,
                                         any_observed_overlap=any(any(r['overlap'].values()) for r in overlaps)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['step1','step2','runs'],required=True)
    a=p.parse_args()
    if a.phase=='runs':
        audit_runs(a.root)
    else:
        audit_data(a.root,a.phase)
    print('audit complete',a.phase,flush=True)
if __name__=='__main__':
    main()
