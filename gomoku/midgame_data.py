"""Exact-depth attack starts with multiple structural families and legal contexts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .env import BatchBoard
from .multistep_tactics import winning_cells,safe_forks,forcing_chains,multi_targets
from .tactics import position_key
from .train import atomic_json

FIELDS=('obs','labels','category','keys','histories','horizon','family')

def canonical(cells,color):
    grid=cells.reshape(12,12)
    boards=[np.rot90(x,k).tobytes() for x in (grid,np.fliplr(grid)) for k in range(4)]
    return hashlib.sha256(min(boards)+bytes([int(color)+1])).hexdigest()

def attack(cells,color):
    wins=winning_cells(cells,color)
    if wins.any():
        return 1,wins
    forks=safe_forks(cells,color)
    if forks.any():
        return 2,forks
    chains=forcing_chains(cells,color)
    labels=np.zeros(144,dtype=bool)
    if chains:
        labels[list(chains)]=True
        return 3,labels
    return 0,labels

def template(depth,family,rng):
    anchor=np.array([5,5])
    horizontal=np.array([0,1])
    own,enemy=[],[]
    if depth==1:
        offsets=[-2,-1,0,1] if family=='four' else [-2,-1,1,2]
        own=[anchor+i*horizontal for i in offsets]
    elif depth==2 and family in ('open_three','broken_three'):
        offsets=[-1,0,1] if family=='open_three' else [-2,0,1]
        own=[anchor+i*horizontal for i in offsets]
    elif depth==2:
        own=[anchor-i*horizontal for i in (1,2,3)]
        own += [anchor-np.array([i,0]) for i in (1,2,3)]
        enemy=[anchor-4*horizontal,anchor-np.array([4,0])]
    else:
        missing=int(rng.choice((-2,-1))) if family=='broken_chain' else 0
        own=[anchor+i*horizontal for i in (-3,-2,-1,0) if i!=missing]
        enemy=[anchor-4*horizontal]
        joint=anchor+missing*horizontal
        direction=np.array([1,int(rng.choice((-1,1)))]) if family=='diagonal_chain' else np.array([-1,0])
        own += [joint+i*direction for i in (1,2)]
        anchor=joint
    if family in ('counterattack','counterattack_chain'):
        # The attack move simultaneously blocks an enemy four.
        own.append(anchor-np.array([5,5]))
        enemy += [anchor-np.array([i,i]) for i in (1,2,3,4)]
    coords=np.array(own+enemy)
    if np.any((coords<0)|(coords>11)) or len(np.unique(coords,axis=0))!=len(coords):
        return None
    if rng.integers(2):
        coords=coords[:,::-1]
    for axis in (0,1):
        if rng.integers(2):
            coords[:,axis]=11-coords[:,axis]
    lo=-coords.min(0)
    hi=11-coords.max(0)
    coords += np.array([rng.integers(lo[i],hi[i]+1) for i in (0,1)])
    return coords[:len(own)]@np.array([12,1]),coords[len(own):]@np.array([12,1])

def collect(per_depth,seed,excluded=()):
    rng=np.random.default_rng(seed)
    seen=set(excluded)
    families={1:['four','broken_four'],2:['open_three','broken_three','cross','counterattack'],
              3:['straight_chain','broken_chain','diagonal_chain','counterattack_chain']}
    records=[]
    for depth in (1,2,3):
        count=0
        for attempt in range(200000):
            family=families[depth][count%len(families[depth])]
            color=1 if (count//len(families[depth]))%2==0 else -1
            pattern=template(depth,family,rng)
            if pattern is None:
                continue
            own,enemy=pattern
            black=list(own if color==1 else enemy)
            white=list(enemy if color==1 else own)
            empty=[x for x in range(144) if x not in black+white]
            rng.shuffle(empty)
            while len(black)-len(white)!=(color==-1):
                if len(black)-len(white)>(color==-1):
                    white.append(empty.pop())
                else:
                    black.append(empty.pop())
            for _ in range(int(rng.integers(1,9))):
                black.append(empty.pop())
                white.append(empty.pop())
            rng.shuffle(black)
            rng.shuffle(white)
            moves=[x for pair in zip(black,white) for x in pair]
            if color==-1:
                moves.append(black[-1])
            board=BatchBoard(1,12)
            for ply,action in enumerate(moves):
                if board.finished[0]:
                    break
                board.place(np.array([action]),1 if ply%2==0 else -1)
            if board.finished[0]:
                continue
            h,labels=attack(board.board[0],color)
            if h!=depth:
                continue
            key=canonical(board.board[0],color)
            if key in seen:
                continue
            seen.add(key)
            history=np.full(144,-1,dtype=np.int16)
            history[:len(moves)]=moves
            records.append(dict(obs=board.observe(color)[0].astype(np.int8),labels=labels,
                                category={1:1,2:3,3:5}[depth],
                                keys=position_key(board.board[0],color),histories=history,
                                horizon=depth,family=family))
            count+=1
            if count>=per_depth:
                print(f'{seed} depth={depth} records={count} attempts={attempt}',flush=True)
                break
        else:
            raise RuntimeError((seed,depth,family,count))
    return records

def save(path,records):
    np.savez_compressed(path,**{k:np.array([r[k] for r in records]) for k in FIELDS},size=12)

def all_prefix_keys(records):
    keys=set()
    for record in records:
        board=BatchBoard(1,12)
        moves=record['histories']
        for ply,action in enumerate(moves[moves>=0]):
            color=1 if ply%2==0 else -1
            keys.add(canonical(board.board[0],color))
            board.place(np.array([action]),color)
        color=1 if int(board.moves[0])%2==0 else -1
        keys.add(canonical(board.board[0],color))
    return keys

def read_records(path):
    with np.load(path) as d:
        arrays={k:d[k].copy() for k in FIELDS}
    return [{k:arrays[k][i].copy() if isinstance(arrays[k][i],np.ndarray) else arrays[k][i].item() for k in FIELDS}
            for i in range(len(arrays['keys']))]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','diversify'],required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.phase=='prepare':
        test=collect(128,202_000_481)
        validation=collect(128,201_000_481,all_prefix_keys(test))
        save(a.output/'test_starts.npz',test)
        save(a.output/'validation_starts.npz',validation)
        excluded=all_prefix_keys(test+validation)
        with np.load(a.base) as saved:
            d={k:saved[k].copy() for k in ('obs','labels','category','keys','histories')}
            keep=[]
            starts=[]
            for i in range(len(d['keys'])):
                obs=d['obs'][i]
                color=1 if obs[3,0,0] else -1
                cells=((obs[0]-obs[1])*color).reshape(-1).astype(np.int8)
                if canonical(cells,color) in excluded:
                    continue
                keep.append(i)
                if d['category'][i] not in (1,3,5):
                    continue
                h,labels=attack(cells,color)
                if h:
                    starts.append(dict(obs=obs,labels=labels,category={1:1,2:3,3:5}[h],
                                       keys=d['keys'][i],histories=d['histories'][i],horizon=h,family='previous_pool'))
            arrays={k:d[k][keep] for k in ('obs','labels','category','keys','histories')}
        np.savez_compressed(a.output/'baseline_aux.npz',**arrays,size=12)
        save(a.output/'step1_starts.npz',starts)
        atomic_json(a.output/'data_prepare.json',dict(seed=4810,base_examples=len(keep),starts=len(starts),
                      validation_starts=len(validation),test_starts=len(test),
                      canonical_D4_reserved=len(excluded),families={str(h):sorted({r['family'] for r in test if r['horizon']==h}) for h in (1,2,3)}))
    else:
        holdout=read_records(a.output/'test_starts.npz')+read_records(a.output/'validation_starts.npz')
        excluded=all_prefix_keys(holdout)
        with np.load(a.output/'baseline_aux.npz') as d:
            for obs in d['obs']:
                c=1 if obs[3,0,0] else -1
                excluded.add(canonical(((obs[0]-obs[1])*c).reshape(-1).astype(np.int8),c))
        records=collect(512,200_000_481,excluded)
        save(a.output/'step2_starts.npz',records)
        # Add both direct attacks and their continuation targets under real
        # tactical replies. Prefixes contain only moves that already occurred.
        from .env import opponent_actions
        extra=list(records)
        rng=np.random.default_rng(203_000_481)
        seen={r['keys'] for r in records}
        for record in records:
            color=1 if record['obs'][3,0,0] else -1
            board=BatchBoard(1,12)
            moves=record['histories'][record['histories']>=0].tolist()
            for t,x in enumerate(moves):
                board.place(np.array([x]),1 if t%2==0 else -1)
            for _ in range(record['horizon']-1):
                h,label=attack(board.board[0],color)
                if not h:
                    break
                action=int(rng.choice(np.flatnonzero(label)))
                board.place(np.array([action]),color)
                moves.append(action)
                if board.finished[0]:
                    break
                reply=int(opponent_actions(board,-color,rng,'tactical')[0])
                board.place(np.array([reply]),-color)
                moves.append(reply)
                if board.finished[0]:
                    raise AssertionError('Certified attack lost to opponent')
                h,label=attack(board.board[0],color)
                history=np.full(144,-1,dtype=np.int16)
                history[:len(moves)]=moves
                key=position_key(board.board[0],color)
                canon=canonical(board.board[0],color)
                if h and key not in seen and canon not in excluded:
                    seen.add(key)
                    extra.append(dict(obs=board.observe(color)[0].astype(np.int8),labels=label,
                                      category={1:1,2:3,3:5}[h],keys=key,histories=history,
                                      horizon=h,family=record['family']+'_continuation'))
        # Defensive counterparts remain reachable: the attacker plays a
        # non-winning contextual move, then the defender gets the real turn.
        defensive_examples=0
        for record in records:
            if record['horizon']==1:
                continue
            color=1 if record['obs'][3,0,0] else -1
            moves=record['histories'][record['histories']>=0].tolist()
            base=BatchBoard(1,12)
            for t,x in enumerate(moves):
                base.place(np.array([x]),1 if t%2==0 else -1)
            candidates=rng.permutation(np.flatnonzero(base.legal()[0]))[:8]
            for action in candidates:
                board=BatchBoard(1,12)
                for t,x in enumerate(moves):
                    board.place(np.array([x]),1 if t%2==0 else -1)
                board.place(np.array([action]),color)
                if board.finished[0]:
                    continue
                remaining,_=attack(board.board[0],color)
                if remaining!=record['horizon']:
                    continue
                label,cat=multi_targets(board,np.array([-color]))
                if cat[0] not in (4,6) or not label[0].any():
                    continue
                key=position_key(board.board[0],-color)
                if key in seen or canonical(board.board[0],-color) in excluded:
                    continue
                seen.add(key)
                history=np.full(144,-1,dtype=np.int16)
                history[:len(moves)+1]=moves+[int(action)]
                extra.append(dict(obs=board.observe(-color)[0].astype(np.int8),labels=label[0],
                                  category=int(cat[0]),keys=key,histories=history,
                                  horizon=0,family=record['family']+'_prevention'))
                defensive_examples+=1
                break
        with np.load(a.output/'baseline_aux.npz') as d:
            fields=('obs','labels','category','keys','histories')
            arrays={k:d[k].copy() for k in fields}
        n=len(extra)
        np.savez_compressed(a.output/'step2_aux.npz',
                            **{k:np.concatenate((np.array([r[k] for r in extra]),v[n:])) for k,v in arrays.items()},size=12)
        # Keep enough defensive categories for equal category sampling.
        with np.load(a.output/'step2_aux.npz') as d:
            counts={str(c):int((d['category']==c).sum()) for c in range(1,7)}
            assert min(counts.values())>0
            assert len(set(d['keys']))==len(d['keys'])
        atomic_json(a.output/'data_diversify.json',dict(attack_starts=len(records),
                     attack_and_continuation_examples=n,defensive_counterparts=defensive_examples,counts=counts,
                     source='multiple exact-depth structural families with randomized legal context',
                     files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.glob('*.npz')}))
if __name__=='__main__':
    main()
