"""Verified branched supervision from current-policy sampled visited states."""
import numpy as np
from .env import BatchBoard
from .midgame_data import attack
from .offense_dagger import record
from .robust_long_plan import verify_plan


def records_from_tree(history, color, plan, preparations=6, round_index=1):
    board=BatchBoard(1,12)
    moves=history[history>=0].tolist()
    for ply,action in enumerate(moves):
        board.place(np.array([action]),1 if ply%2==0 else -1)
    if not verify_plan(board.board[0],color,plan,preparations):
        raise ValueError('Unverified or incomplete teacher tree')
    result=[];row_by_state={};expanded_actions={}
    def walk(current, prefix, node):
        if current.finished[0] or node['kind']=='terminal':
            return
        labels=np.zeros(144,dtype=bool)
        labels[node['labels']]=True
        padded=np.full(144,-1,dtype=np.int16)
        padded[:len(prefix)]=prefix
        horizon,_=attack(current.board[0],color)
        key=current.board[0].tobytes()
        if key in row_by_state:
            result[row_by_state[key]]['labels']|=labels
        else:
            row_by_state[key]=len(result)
            result.append(record(current,color,padded,labels,{1:1,2:3,3:5}.get(horizon,5),round_index))
        done=expanded_actions.setdefault(key,set())
        if node['kind']=='exact_tactic':
            return
        for text_action,choice in node['actions'].items():
            if text_action in done:continue
            done.add(text_action)
            if choice['kind']=='terminal_move':
                continue
            action=int(text_action)
            for branch in choice['branches']:
                child=BatchBoard(1,12)
                for field in ('board','line_counts','moves','last','finished','winner'):
                    getattr(child,field)[:]=getattr(current,field)
                child.place(np.array([action]),color)
                child.place(np.array([branch['reply']]),-color)
                walk(child,prefix+[action,branch['reply']],branch['plan'])
    walk(board,moves,plan)
    return result
