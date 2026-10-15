"""Collect bounded 12x12 search-teacher suggestions on novel quiet history prefixes."""
from pathlib import Path
import json
import numpy as np

from .dagger import records_from_file, save_records, heldout_canonical_keys
from .env import BatchBoard
from .expanded_tactics import expanded_targets
from .external_engine import PiskvorkEngine
from .multistep_tactics import winning_cells
from .opening_followup import replay_keys, key, verify_records
from .tactics import position_key

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/balanced_curriculum_20261014'
ENGINE=ROOT/'external/rapfi_250615/engine'


def main():
    evaluation=[]
    for directory in (RUN,ROOT/'runs/opening_followup_diverse_20261014'):
        for path in directory.glob('*.json'):
            if not path.stat().st_size: continue
            data=json.loads(path.read_text())
            if isinstance(data,dict) and 'replays' in data: evaluation.append(data)
    reserved=replay_keys(evaluation)
    reserved |= heldout_canonical_keys([ROOT/'runs/offense_e_20261008/data/fresh_test_starts.npz'])
    known=set(reserved)
    records=[]; provenance=[]; counts={}
    for kind in ('fly','gru','rnn','rewired'):
        pools={1:[],-1:[]}
        rows=records_from_file(RUN/f'{kind}_data.npz')
        sources=json.loads((RUN/f'{kind}_sources.json').read_text())
        for record,source in zip(rows,sources):
            if source!='fresh': continue
            actual=record['histories'][record['histories']>=0]
            board=BatchBoard(1,12)
            for ply,action in enumerate(actual):
                color=1 if ply%2==0 else -1
                if 4<=ply<=25:
                    labels,category=expanded_targets(board,np.array([color]))
                    if (not int(category[0]) and not winning_cells(board.board[0],-color).any()
                            and len(pools[color])<16):
                        history=np.full(144,-1,dtype=np.int16); history[:ply]=actual[:ply]
                        candidate=dict(obs=board.observe(color)[0].astype(np.int8),
                                       labels=np.zeros(144,dtype=bool),category=7,
                                       histories=history,keys=position_key(board.board[0],color),round=1)
                        canonical=key(candidate)
                        if canonical not in known:
                            known.add(canonical)
                            pools[color].append(candidate)
                board.place(np.array([action]),color)
        counts[kind]={'black':len(pools[1]),'white':len(pools[-1])}
        with PiskvorkEngine([ENGINE/'pbrain-rapfi-windows-sse.exe'],size=12,turn_ms=200,cwd=ENGINE) as engine:
            for color in (1,-1):
                for record in pools[color]:
                    cells=((record['obs'][0]-record['obs'][1])*color).reshape(-1).astype(np.int8)
                    response=engine.recommend(cells,color,history=record['histories'])
                    record['labels'][response['action']]=True
                    records.append(record)
                    provenance.append(dict(kind=kind,color=color,key=record['keys'],
                                           action=response['action'],evaluation=response['evaluation'],
                                           seconds=response['seconds'],evaluator_disabled=response['evaluator_disabled']))
        print(json.dumps(dict(event='quiet_teacher_collection',kind=kind,**counts[kind])),flush=True)
    verify_records(records)
    assert not any(key(r) in reserved for r in records)
    save_records(RUN/'rapfi_quiet12_bank.npz',records)
    report=dict(status='complete',counts=counts,records=len(records),provenance=provenance,
                turn_ms=200,rule='freestyle',size=12,illegal_labels=0,bad_history_observations=0,
                reserved_evaluation_overlap=0,teacher='Rapfi 250615 classical evaluation plus search',
                scope='suggestions on quiet visited states, not forced-win certificates',training_performed=False)
    (RUN/'rapfi_quiet12_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(event='complete',records=len(records))),flush=True)


if __name__=='__main__': main()
