"""Mine losing games, reserve complete new game banks, and build equal-size data."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from .env import BatchBoard
from .tactics import position_key
from .multistep_tactics import winning_cells, multi_targets, forcing_chains
from .advanced_match_audit import audit
from .models import load_policy
from .train import atomic_json

def snapshots(game):
    if game['size'] != 12:
        raise ValueError('Only 12x12 games are supported')
    board = BatchBoard(1, 12)
    history = np.full(144, -1, dtype=np.int16)
    result = []
    for ply, action in enumerate(game['moves']):
        color = 1 if ply % 2 == 0 else -1
        if color == game['a_color']:
            result.append((board.board[0].copy(), history.copy(), ply))
        board.place(np.array([action]), color)
        history[ply] = action
    if not board.finished[0] or int(board.winner[0]) != game['winner']:
        raise ValueError('Invalid replay winner')
    return result

def restore(cells, history):
    board = BatchBoard(1,12)
    for ply, action in enumerate(history[history >= 0]):
        board.place(np.array([action]), 1 if ply % 2 == 0 else -1)
    np.testing.assert_array_equal(board.board[0], cells)
    return board

def mine(games, excluded=()):
    seen = set(excluded)
    records, unresolved = [], []
    counts = {'unique_block_loss_games':0, 'multiple_loss_games':0, 'other_loss_games':0}
    for index, game in enumerate(games):
        if game['winner'] != -game['a_color']:
            continue
        cause = game['loss_cause']
        key = {'missed_unique_block':'unique_block_loss_games',
               'already_multiple_winning_cells':'multiple_loss_games'}.get(cause,'other_loss_games')
        counts[key] += 1
        positions = snapshots(game)
        if cause == 'missed_unique_block':
            candidates = [(len(positions)-1, 0)]
        elif cause == 'already_multiple_winning_cells':
            first = next(i for i,(cells,_,_) in enumerate(positions)
                         if winning_cells(cells,-game['a_color']).sum() >= 2)
            candidates = [(first-lag,lag) for lag in (1,2) if first >= lag]
        else:
            candidates = []
        for turn, lag in candidates:
            cells, history, ply = positions[turn]
            poskey = position_key(cells, game['a_color'])
            if poskey in seen:
                continue
            board = restore(cells,history)
            labels, cat = multi_targets(board,np.array([game['a_color']]))
            origin = dict(game=index, kind=game['a'], seed=game['seed'],
                          environment_seed=game['environment_seed'], opening_index=game['opening_index'],
                          ply=ply, learner_turns_before_multiple=lag, cause=cause, key=poskey)
            if not cat[0] or not labels[0].any():
                origin["reason"] = "no_supported_nonempty_label; not a proof of general unavoidability"
                unresolved.append(origin)
                continue
            seen.add(poskey)
            records.append(dict(obs=board.observe(game['a_color'])[0].astype(np.int8),
                                labels=labels[0], category=int(cat[0]), keys=poskey,
                                histories=history, origin=origin))
        if (index+1) % 256 == 0:
            print(f'mine games={index+1} labels={len(records)} unresolved={len(unresolved)}',flush=True)
    return records, unresolved, counts

def save_records(path, records):
    if not records:
        raise ValueError(f'No labelled positions for {path}')
    np.savez_compressed(path,**{k:np.array([r[k] for r in records])
                               for k in ('obs','labels','category','keys','histories')},size=12)
    atomic_json(path.with_suffix('.origins.json'),dict(records=[r['origin'] for r in records]))

def game_keys(games):
    # Reserve every ply, including the opponent's turns, of each held-out game.
    keys = set()
    for game in games:
        board = BatchBoard(1,12)
        for ply, action in enumerate(game['moves']):
            color = 1 if ply % 2 == 0 else -1
            keys.add(position_key(board.board[0],color))
            board.place(np.array([action]),color)
    return keys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eval-games',type=int,default=256)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    banks={}
    for split, envseed in (('validation',140_000_470),('test',150_000_470)):
        path=a.output/f'{split}_replays.json'
        if path.exists():
            banks[split]=json.loads(path.read_text(encoding='utf-8'))['games']
        else:
            games=[]
            for kind in ('fly','gru'):
                model,_=load_policy(a.source/f'{kind}_seed4601.pt',torch.device('cpu'))
                _,replays=audit(model,4601,a.eval_games,True,envseed)
                games.extend(replays)
            atomic_json(path,dict(games=games,environment_seed=envseed,retired_source_policy=True))
            banks[split]=games
    reserved=game_keys(banks['validation']) | game_keys(banks['test'])
    for split in ('validation','test'):
        # Deduplicate across held-out banks, too.
        other=game_keys(banks['test']) if split=='validation' else game_keys(banks['validation'])
        records,unresolved,counts=mine(banks[split],other)
        save_records(a.output/f'{split}.npz',records)
        atomic_json(a.output/f'{split}_mining.json',dict(counts=counts,unresolved=unresolved))
    source=json.loads((a.source/'mining_replays.json').read_text(encoding='utf-8'))['games']
    records,unresolved,counts=mine(source,reserved)
    save_records(a.output/'mined.npz',records)
    with np.load(a.base) as saved:
        arrays={k:saved[k].copy() for k in ('obs','labels','category','keys','histories')}
    mined_keys={r['keys'] for r in records}
    eligible=np.array([i for i,k in enumerate(arrays['keys']) if k not in reserved and k not in mined_keys])
    rng=np.random.default_rng(4700)
    budget=min(len(arrays['keys']),len(eligible))
    if budget < 10000:
        raise ValueError(f'Only {len(eligible)} eligible baseline positions for budget {budget}; supply a larger base pool')
    order=rng.permutation(eligible)
    np.savez_compressed(a.output/'baseline.npz',**{k:v[order[:budget]] for k,v in arrays.items()},size=12)
    n=min(len(records),budget//2)
    selected=rng.permutation(len(records))[:n]
    mined={k:np.array([records[i][k] for i in selected]) for k in arrays}
    np.savez_compressed(a.output/'train.npz',
                        **{k:np.concatenate((v[order[:budget-n]],mined[k])) for k,v in arrays.items()},size=12)
    manifest=dict(size=12,budget_positions=budget,mined_positions=n,counts=counts,
                  unresolved_positions=len(unresolved),all_eval_game_positions_reserved=len(reserved),
                  source_evaluation_retired=True,source_games=len(source),
                  categories={str(c):sum(r['category']==c for r in records) for c in range(1,7)},
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.glob('*.npz')})
    atomic_json(a.output/'manifest.json',manifest)
    atomic_json(a.output/'unresolved.json',dict(positions=unresolved))
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':
    main()
