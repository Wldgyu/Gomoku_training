"""Independent reachable chain motifs, with translated/rotated random contexts."""
import argparse, json
from pathlib import Path
import numpy as np
from .env import BatchBoard
from .multistep_tactics import multi_targets
from .mine_multistep import save_records, game_keys
from .tactics import position_key
from .train import atomic_json

def collect(per_category,seed,excluded):
    rng=np.random.default_rng(seed)
    seen=set(excluded)
    records=[]
    counts={5:0,6:0}
    for attempt in range(100000):
        cat=5 if counts[5]<per_category else 6
        color=1 if rng.random()<.5 else -1
        attacker=color if cat==5 else -color
        a=np.array([[5,1],[5,2],[5,3],[3,4],[4,4]])
        b=np.array([[5,0],[1,4],[8,4],[9,2],[10,0]])
        swap=rng.integers(2)
        if swap:
            a,b=a[:,::-1],b[:,::-1]
        for axis in (0,1):
            if rng.integers(2):
                a[:,axis]=11-a[:,axis]
                b[:,axis]=11-b[:,axis]
        coords=np.concatenate((a,b))
        lower=-coords.min(0)
        upper=11-coords.max(0)
        delta=np.array([rng.integers(lower[i],upper[i]+1) for i in (0,1)])
        a,b=a+delta,b+delta
        black=list((a if attacker==1 else b)@np.array([12,1]))
        white=list((b if attacker==1 else a)@np.array([12,1]))
        empty=np.array([i for i in range(144) if i not in black+white])
        filler=rng.choice(empty,2*int(rng.integers(0,9))+(color==-1),replace=False)
        black.extend(filler[::2].tolist())
        white.extend(filler[1::2].tolist())
        rng.shuffle(black)
        rng.shuffle(white)
        history=np.full(144,-1,dtype=np.int16)
        moves=[x for pair in zip(black,white) for x in pair]
        if color==-1:
            moves.append(black[-1])
        board=BatchBoard(1,12)
        valid=True
        for ply,action in enumerate(moves):
            board.place(np.array([action]),1 if ply%2==0 else -1)
            if board.finished[0]:
                valid=False
                break
        if not valid:
            continue
        key=position_key(board.board[0],color)
        if key in seen:
            continue
        labels,categories=multi_targets(board,np.array([color]))
        if categories[0]!=cat:
            continue
        history[:len(moves)]=moves
        seen.add(key)
        counts[cat]+=1
        records.append(dict(obs=board.observe(color)[0].astype(np.int8),labels=labels[0],
                            category=cat,keys=key,histories=history,
                            origin=dict(generator='forcing_cross_chain',seed=seed,attempt=attempt)))
        if min(counts.values())>=per_category:
            print(f'chain motifs {counts} attempts={attempt}',flush=True)
            return records
    raise RuntimeError(counts)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    a=p.parse_args()
    reserved=set()
    for split in ('validation','test'):
        games=json.loads((a.data/f'{split}_replays.json').read_text(encoding='utf-8'))['games']
        reserved |= game_keys(games)
    keys=set(reserved)
    for name in ('baseline','train','mined'):
        with np.load(a.data/f'{name}.npz') as d:
            keys.update(d['keys'].tolist())
    records=collect(512,170_000_470,keys)
    save_records(a.data/'chain_train.npz',records)
    with np.load(a.data/'train.npz') as d:
        arrays={k:d[k].copy() for k in ('obs','labels','category','keys','histories')}
    # The mined examples occupy the tail and are all retained.
    np.savez_compressed(a.data/'train.npz',**{
        k:np.concatenate((np.array([r[k] for r in records]),v[len(records):]))
        for k,v in arrays.items()},size=12)
    import hashlib
    manifest=json.loads((a.data/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(chain_motif_positions=len(records),chain_motif_scope='single crossing motif with D4/translation and random legal context',
                    train_counts={str(c):int((np.load(a.data/'train.npz')['category']==c).sum()) for c in range(1,7)})
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.data.glob('*.npz')}
    atomic_json(a.data/'manifest.json',manifest)
if __name__=='__main__':
    main()
