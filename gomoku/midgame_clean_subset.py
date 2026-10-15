"""Supplementary common full-game subset excluding observed training overlap."""
import copy,json
from pathlib import Path
import numpy as np
from .env import BatchBoard
from .expanded_tactics import expanded_targets
from .multistep_tactics import winning_cells
from .train import atomic_json
RUN=Path(__file__).resolve().parent/'runs/midgame_20261007'

def game_counters(game):
    board=BatchBoard(1,12)
    color=game['a_color']
    counters={name:dict(positions=0,correct=0) for name in ('win','block','create_fork','prevent_fork')}
    names={1:'win',2:'block',3:'create_fork',4:'prevent_fork'}
    reached=False
    forks=0
    for ply,action in enumerate(game['moves']):
        turn=1 if ply%2==0 else -1
        if turn==color:
            labels,category=expanded_targets(board,np.array([color]))
            if category[0]:
                name=names[int(category[0])]
                counters[name]['positions']+=1
                counters[name]['correct']+=int(labels[0,action])
            reached |= winning_cells(board.board[0],-color).sum()>=2
            before=winning_cells(board.board[0],color).sum()
        board.place(np.array([action]),turn)
        if turn==color and not board.finished[0]:
            after=winning_cells(board.board[0],color).sum()
            enemy=winning_cells(board.board[0],-color).sum()
            forks+=int(before<2 and after>=2 and enemy==0)
    assert board.finished[0] and int(board.winner[0])==game['winner']
    result=game['winner']*color
    return dict(counters=counters,reached=int(reached),forks=forks,result=result)

def main():
    overlaps=json.loads((RUN/'overlap_details.json').read_text(encoding='utf-8'))['hits']
    exclude={s:sorted({h['opening_index'] for h in overlaps if f'_{s}_' in h['file']}) for s in ('validation','test')}
    original=json.loads((RUN/'evaluation.json').read_text(encoding='utf-8'))['full_games']
    rows=[]
    for raw in original:
        row=copy.deepcopy(raw)
        games=json.loads((RUN/f"{raw['arm']}_{raw['kind']}_{raw['split']}_full_replays.json").read_text(encoding='utf-8'))['games']
        removed=[g for g in games if g['opening_index'] in exclude[raw['split']]]
        for game in removed:
            stats=game_counters(game)
            row['games']-=1
            row[{1:'wins',-1:'losses',0:'draws'}[stats['result']]]-=1
            row['games_reaching_multiple_threat']-=stats['reached']
            row['actual_safe_forks_created']-=stats['forks']
            for name,c in stats['counters'].items():
                for key in ('positions','correct'):
                    row['tactical_decisions'][name][key]-=c[key]
        for c in row['tactical_decisions'].values():
            c['accuracy']=c['correct']/c['positions'] if c['positions'] else None
        block=row['tactical_decisions']['block']
        row['defense_omissions']=block['positions']-block['correct']
        row['multiple_threat_game_rate']=row['games_reaching_multiple_threat']/row['games']
        row['actual_safe_forks_per_game']=row['actual_safe_forks_created']/row['games']
        row['removed_common_indices']=exclude[raw['split']]
        # Terminal cause totals are not recomputed for this supplementary view.
        row.pop('terminal_loss_causes',None)
        row.pop('preceding_turn_of_multiple_threat_losses',None)
        rows.append(row)
    summaries={}
    for split in ('validation','test'):
        summaries[split]={}
        for arm in ('baseline','step1','step2'):
            rr=[r for r in rows if r['split']==split and r['arm']==arm]
            n=sum(r['games'] for r in rr)
            omissions=sum(r['defense_omissions'] for r in rr)
            opportunities=sum(r['tactical_decisions']['block']['positions'] for r in rr)
            summaries[split][arm]=dict(games=n,wins=sum(r['wins'] for r in rr),
                                      defense_omissions=omissions,defense_opportunities=opportunities,
                                      defense_omission_rate=omissions/opportunities,
                                      multiple_threat_game_rate=sum(r['games_reaching_multiple_threat'] for r in rr)/n,
                                      safe_forks=sum(r['actual_safe_forks_created'] for r in rr))
    atomic_json(RUN/'clean_full_game_subset.json',dict(runs=rows,summary=summaries,
        common_removed_indices=exclude,posthoc_supplementary=True,zero_remaining_observed_training_overlap=True,
        interpretation='Common opening indices with an exact D4 position match in any arm are excluded from all arms and both models. Original preregistered full-game results remain reported; no setting or model was changed.'))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':
    main()
