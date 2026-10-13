"""Bounded AND/OR teacher: robust over sampled replies, not all legal replies.

Own choices are OR nodes; every selected opponent reply is an AND branch.
Budget exhaustion is UNKNOWN and never generates a positive target.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .env import geometry, opponent_actions
from .long_plan import has_win
from .midgame_data import attack, canonical
from .multistep_tactics import winning_cells
from .offense_dagger import board_from_cells, quiet_candidates


def opponent_replies(cells, color, top_k=3, tie_seeds=4):
    """color is the learner color; cells already contains its latest move."""
    legal = cells == 0
    if not legal.any():
        return []
    lines, incidence = geometry(12)
    values = cells[lines]
    own = (values == -color).sum(1)
    other = (values == color).sum(1)
    powers = np.array([0, 1, 6, 30, 2000, 0], dtype=np.float32)
    score = (powers[own] * (other == 0) + .8 * powers[other] * (own == 0)) @ incidence.T
    wins = winning_cells(cells, -color)
    blocks = winning_cells(cells, color)
    score += wins * 100000 + blocks * 10000
    score[~legal] = -np.inf
    ranked = np.flatnonzero(legal)
    ranked = ranked[np.argsort(-score[ranked], kind='stable')][:top_k]
    # Never prune away an immediate winning response or a required block.
    replies = list(map(int, np.flatnonzero((wins | blocks) & legal)))
    replies.extend(map(int, ranked))
    board = board_from_cells(cells.copy())
    for seed_index in range(tie_seeds):
        payload = cells.tobytes() + bytes([color + 1]) + seed_index.to_bytes(4, 'little')
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
        replies.append(int(opponent_actions(board, -color, np.random.default_rng(seed), 'tactical')[0]))
    return list(dict.fromkeys(replies))


def nested_quiet_candidates(cells, color, limit):
    """Retain the exact 3-move prefix, then the 6-move prefix across score ties."""
    widths=[min(3,limit)]
    if limit>3:widths.append(min(6,limit))
    if limit>6:widths.append(limit)
    moves=[]
    for width in widths:moves.extend(map(int,quiet_candidates(cells,color,width)))
    return list(dict.fromkeys(moves))[:limit]


class RobustTeacher:
    def __init__(self, candidate_limit=3, top_k=3, tie_seeds=4, node_budget=400, root_candidate_limit=None):
        if min(candidate_limit, top_k, tie_seeds, node_budget) < 1:
            raise ValueError('Search limits must be positive')
        if root_candidate_limit is not None and root_candidate_limit < 1:
            raise ValueError('Root candidate limit must be positive')
        self.candidate_limit = candidate_limit
        self.root_candidate_limit = root_candidate_limit
        self._root_depth = None
        self.top_k, self.tie_seeds = top_k, tie_seeds
        self.node_budget = node_budget
        self.nodes = 0
        self.cache = {}

    def solve(self, cells, color, preparations=6):
        """Return win/no_plan/unknown; no_plan is not a proof of loss."""
        if self._root_depth is None:
            self._root_depth = preparations
        key = (cells.tobytes(), int(color), preparations)
        if key in self.cache:
            return self.cache[key]
        if self.nodes >= self.node_budget:
            return {'status': 'unknown', 'reason': 'node_budget'}
        self.nodes += 1
        if has_win(cells, -color):
            return {'status': 'no_plan', 'reason': 'opponent_won'}
        if has_win(cells, color):
            return {'status': 'win', 'kind': 'terminal', 'own_moves': 0}
        horizon, labels = attack(cells, color)
        if horizon:
            result = {'status': 'win', 'kind': 'exact_tactic', 'own_moves': int(horizon),
                      'labels': np.flatnonzero(labels).tolist()}
            self.cache[key] = result
            return result
        if preparations <= 0 or not (cells == 0).any():
            return {'status': 'no_plan', 'reason': 'depth_or_draw'}
        candidates = list(map(int, np.flatnonzero(winning_cells(cells, -color))))
        if self.root_candidate_limit is not None and preparations == self._root_depth:
            candidates += nested_quiet_candidates(cells,color,self.root_candidate_limit)
        else:
            candidates += list(map(int,quiet_candidates(cells,color,self.candidate_limit)))
        candidates = list(dict.fromkeys(candidates))
        found, unknown = {}, False
        for action in candidates:
            after = cells.copy()
            after[action] = color
            if has_win(after, color):
                found[action] = {'status': 'win', 'kind': 'terminal_move', 'own_moves': 1}
                continue
            replies = opponent_replies(after, color, self.top_k, self.tie_seeds)
            branches, action_unknown, failed = [], False, not bool(replies)
            for reply in replies:
                replied = after.copy()
                replied[reply] = -color
                child = self.solve(replied, color, preparations - 1)
                if child['status'] == 'no_plan':
                    failed = True
                    break
                if child['status'] == 'unknown':
                    action_unknown = True
                branches.append({'reply': reply, 'plan': child})
            if not failed and not action_unknown:
                found[action] = {'status': 'win', 'kind': 'sampled_and',
                                 'own_moves': 1 + max(b['plan']['own_moves'] for b in branches),
                                 'branches': branches}
            unknown |= action_unknown and not failed
        if found:
            result = {'status': 'win', 'kind': 'sampled_or', 'labels': list(found),
                      'own_moves': min(p['own_moves'] for p in found.values()),
                      'actions': {str(a): p for a, p in found.items()}}
        else:
            result = {'status': 'unknown' if unknown else 'no_plan',
                      'reason': 'node_budget' if unknown else 'no_sampled_plan'}
        # Unknown results depend on budget remaining and are not reusable.
        if result['status'] != 'unknown':
            self.cache[key] = result
        return result


def verify_plan(cells, color, plan, preparations=6, top_k=3, tie_seeds=4):
    """Replay each distinct board/proof pair; proof hashes include every child."""
    import hashlib
    signatures={};checked={}
    def signature(node):
        key=id(node)
        if key in signatures:return signatures[key]
        actions=[]
        for action,choice in sorted(node.get('actions',{}).items(),key=lambda x:str(x[0])):
            branches=tuple((b['reply'],signature(b['plan'])) for b in choice.get('branches',[]))
            actions.append((str(action),choice.get('status'),choice.get('kind'),choice.get('own_moves'),branches))
        value=(node.get('status'),node.get('kind'),tuple(node.get('labels',[])),node.get('own_moves'),tuple(actions))
        digest=hashlib.sha256(repr(value).encode()).digest();signatures[key]=digest
        return digest
    def check(state,node,depth):
        if not isinstance(node,dict) or node.get('status')!='win':return False
        key=(state.tobytes(),depth,signature(node))
        if key in checked:return checked[key]
        value=validate(state,node,depth);checked[key]=value
        return value
    def validate(state,node,depth):
        if has_win(state,-color):return False
        kind=node.get('kind')
        if kind=='terminal':return has_win(state,color)
        if kind=='exact_tactic':
            horizon,valid=attack(state,color);labels=node.get('labels',[])
            return bool(horizon and labels and node['own_moves']==horizon and all(0<=a<144 and state[a]==0 and valid[a] for a in labels))
        if kind!='sampled_or' or depth<=0:return False
        actions=node.get('actions',{})
        if not actions or set(map(int,actions))!=set(node.get('labels',[])):return False
        for text_action,choice in actions.items():
            action=int(text_action)
            if not 0<=action<144 or state[action]!=0 or choice.get('status')!='win':return False
            after=state.copy();after[action]=color
            if has_win(after,color):
                if choice.get('kind')!='terminal_move':return False
                continue
            expected=opponent_replies(after,color,top_k,tie_seeds);branches=choice.get('branches',[])
            if choice.get('kind')!='sampled_and' or not expected or sorted(b['reply'] for b in branches)!=sorted(expected):return False
            for branch in branches:
                replied=after.copy();replied[branch['reply']]=-color
                if not check(replied,branch['plan'],depth-1):return False
        return True
    try:
        cells=np.asarray(cells)
        if cells.shape!=(144,) or color not in (-1,1) or not np.isin(cells,[-1,0,1]).all():return False
        return check(cells.astype(np.int8),plan,preparations)
    except (KeyError,ValueError,TypeError,AttributeError,IndexError):return False


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--preparations', type=int, default=6)
    p.add_argument('--candidate-limit', type=int, default=3)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--tie-seeds', type=int, default=4)
    p.add_argument('--node-budget', type=int, default=400)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.limit < 1 or a.preparations < 0:
        p.error('limit must be positive and preparations nonnegative')
    config = dict(input=str(a.input.resolve()), input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                  preparations=a.preparations, candidate_limit=a.candidate_limit, top_k=a.top_k,
                  tie_seeds=a.tie_seeds, node_budget=a.node_budget)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(config=config, scope='All selected replies at every preparation step; not all legal moves', rows=[])
    if a.resume and a.output.exists():
        report = json.loads(a.output.read_text(encoding='utf-8'))
        if report['config'] != config:
            raise ValueError('Resume configuration or input hash changed')
    elif a.output.exists():
        raise FileExistsError('Use --resume or a new output path')
    with np.load(a.input, allow_pickle=False) as data:
        for index in range(len(report['rows']), min(a.limit, len(data['obs']))):
            obs = data['obs'][index]
            color = 1 if obs[3, 0, 0] else -1
            cells = ((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8)
            teacher = RobustTeacher(a.candidate_limit, a.top_k, a.tie_seeds, a.node_budget)
            plan = teacher.solve(cells, color, a.preparations)
            if plan['status'] == 'win' and not verify_plan(cells, color, plan, a.preparations, a.top_k, a.tie_seeds):
                raise RuntimeError('Independent witness verification failed')
            old = np.flatnonzero(data['labels'][index]).tolist()
            row = dict(index=index, canonical=canonical(cells, color), old_labels=old,
                       accepted_old_labels=sorted(set(old) & set(plan.get('labels', []))),
                       nodes=teacher.nodes, plan=plan)
            report['rows'].append(row)
            report['counts'] = {status: sum(r['plan']['status'] == status for r in report['rows'])
                                for status in ('win', 'no_plan', 'unknown')}
            atomic_json(a.output, report)
            print(json.dumps(dict(index=index, nodes=teacher.nodes, status=plan['status'],
                                  kind=plan.get('kind'), accepted_old_labels=row['accepted_old_labels'])), flush=True)
    atomic_json(a.output, report)


if __name__ == '__main__':
    main()
