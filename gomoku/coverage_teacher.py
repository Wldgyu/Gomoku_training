"""Coverage mining: a separate 2,000-state allowance for each root action."""
import time
import numpy as np
from .robust_long_plan import RobustTeacher,has_win,nested_quiet_candidates,opponent_replies
from .multistep_tactics import winning_cells,features
from .midgame_data import attack

class RootCoverageTeacher:
    def __init__(self,root_width=12,per_root_budget=2000):
        self.root_width=root_width;self.per_root_budget=per_root_budget;self.nodes=1

    def solve(self,cells,color,preparations=6,model_action=None):
        horizon,labels=attack(cells,color)
        if horizon:return dict(status='win',kind='exact_tactic',own_moves=int(horizon),labels=np.flatnonzero(labels).tolist())
        actions=list(map(int,np.flatnonzero(winning_cells(cells,-color))))+nested_quiet_candidates(cells,color,self.root_width)
        if model_action is not None and cells[model_action]==0:actions.append(int(model_action))
        actions=list(dict.fromkeys(actions));found={};unknown=False
        solver=RobustTeacher(candidate_limit=3,node_budget=self.per_root_budget)
        for action in actions:
            after=cells.copy();after[action]=color
            if has_win(after,color):found[str(action)]=dict(status='win',kind='terminal_move',own_moves=1);continue
            solver.nodes=0  # Exact child proofs are reusable; allowance resets per root.
            branches=[];failed=False;unresolved=False
            replies=opponent_replies(after,color,3,4)
            if not replies:failed=True
            for reply in replies:
                replied=after.copy();replied[reply]=-color
                if has_win(replied,-color):failed=True;break
                child=solver.solve(replied,color,preparations-1)
                if child['status']=='no_plan':failed=True;break
                if child['status']=='unknown':unresolved=True
                branches.append(dict(reply=reply,plan=child))
            self.nodes+=solver.nodes
            if not failed and not unresolved:
                found[str(action)]=dict(status='win',kind='sampled_and',own_moves=1+max(b['plan']['own_moves'] for b in branches),branches=branches)
            unknown|=unresolved and not failed
        if found:return dict(status='win',kind='sampled_or',labels=sorted(map(int,found)),actions=found,own_moves=min(v['own_moves'] for v in found.values()))
        return dict(status='unknown' if unknown else 'no_plan',reason='per_root_budget' if unknown else 'no_sampled_plan')


def coverage_search(payload):
    arm,index,color,chosen,cells=payload
    features.cache_clear();start=time.perf_counter()
    teacher=RootCoverageTeacher(12,2000)
    plan=teacher.solve(np.asarray(cells,dtype=np.int8),color,6,chosen)
    return dict(arm=arm,index=index,plan=plan,nodes=teacher.nodes,seconds=time.perf_counter()-start)


def coverage_game(payload):
    arm,game,states=payload
    features.cache_clear();start=time.perf_counter();rows=[];skipped=0
    for state in states:
        index,color,chosen,cells,known_status,known_labels=state
        if chosen in known_labels or (known_status=='no_plan' and not known_labels):
            skipped+=1;continue
        if known_status=='certified_miss':
            return dict(arm=arm,game=game,rows=rows,skipped=skipped,selected_existing_index=index,seconds=time.perf_counter()-start)
        cells=np.asarray(cells,dtype=np.int8)
        if (cells==color).sum()<4:
            horizon,labels=attack(cells,color)
            if not horizon:skipped+=1;continue
            plan=dict(status='win',kind='exact_tactic',own_moves=int(horizon),labels=np.flatnonzero(labels).tolist())
            nodes=1
        else:
            teacher=RootCoverageTeacher(12,2000)
            plan=teacher.solve(cells,color,6,chosen);nodes=teacher.nodes
        row=dict(arm=arm,game=game,index=index,status=plan['status'],nodes=nodes,seconds=None)
        if plan['status']=='win':
            row.update(plan=plan,labels=plan['labels'],kind=plan['kind'])
        rows.append(row)
        if plan['status']=='win' and chosen not in plan['labels']:break
    return dict(arm=arm,game=game,rows=rows,skipped=skipped,seconds=time.perf_counter()-start)
