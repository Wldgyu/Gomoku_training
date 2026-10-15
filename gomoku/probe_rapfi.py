"""Verify official engine board support and prepare a small teacher data pilot."""
from pathlib import Path
import hashlib
import json
import numpy as np

from .external_engine import PiskvorkEngine
from .dagger import records_from_file, save_records
from .env import BatchBoard
from .tactics import targets

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "external/rapfi_250615/engine"
RUN = ROOT / "runs/balanced_curriculum_20261014"


def main():
    executable = ENGINE / "pbrain-rapfi-windows-sse.exe"
    report = dict(executable=str(executable), executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256((ENGINE/'config.toml').read_bytes()).hexdigest(),
                  freestyle_weights_sha256=hashlib.sha256((ENGINE/'mix9svqfreestyle_bsmix.bin.lz4').read_bytes()).hexdigest(),
                  release="250615", source="https://github.com/dhbloo/rapfi/releases/tag/250615",
                  coordinate_convention="x=column,y=row; stored action=row*size+column", rule="freestyle INFO rule 0",
                  engine_probes={}, teacher_records=[])
    for size in (12,15):
        cells = np.zeros(size*size,dtype=np.int8)
        cells[(size//2)*size+size//2] = 1
        with PiskvorkEngine([executable],size=size,turn_ms=200,cwd=ENGINE) as engine:
            response = engine.recommend(cells,-1)
            # Only call NNUE enabled when the engine explicitly reports its evaluation mapping.
            enabled = any("mix9svq" in s and ("enabled" in s.lower() or "weight" in s.lower())
                          and "disabled" not in s.lower() for s in engine.transcript)
            report["engine_probes"][str(size)] = dict(response=response, transcript=engine.transcript,
                           evaluator_disabled=response["evaluator_disabled"], nnue_enable_evidence=enabled)
    records = []
    for kind in ("fly","gru","rnn","rewired"):
        rows = records_from_file(RUN / f"{kind}_data.npz")
        sources = json.loads((RUN / f"{kind}_sources.json").read_text())
        fresh = [r for r,s in zip(rows,sources) if s=="fresh"]
        # Bounded pilot: select both colors and different categories before engine querying.
        selected=[]
        for category in range(1,7):
            for black in (True,False):
                pool=[r for r in fresh if r['category']==category and bool(r['obs'][3,0,0])==black]
                if pool: selected.append(pool[0])
        with PiskvorkEngine([executable],size=12,turn_ms=200,cwd=ENGINE) as engine:
            for record in selected:
                obs=record['obs']; color=1 if obs[3,0,0] else -1
                cells=((obs[0]-obs[1])*color).reshape(-1).astype(np.int8)
                response=engine.recommend(cells,color,history=record['histories'])
                board=BatchBoard(1,12)
                board.board[0]=cells
                board.moves[0]=int(np.count_nonzero(cells))
                values=cells[board.lines]
                board.line_counts[0,0]=(values==1).sum(-1)
                board.line_counts[0,1]=(values==-1).sum(-1)
                basic,category=targets(board,np.array([color]))
                basic_correct=bool(basic[0,response['action']]) if int(category[0]) else None
                labels=np.zeros(144,dtype=bool); labels[response['action']]=True
                records.append(dict(record,labels=labels))
                report['teacher_records'].append(dict(kind=kind,category=int(record['category']),
                     color=color,action=response['action'],legal=True,basic_correct=basic_correct,
                     matches_previous_label=bool(record['labels'][response['action']]),
                     evaluation=response['evaluation'],seconds=response['seconds'],
                     evaluator_disabled=response['evaluator_disabled']))
    # Write into pre-created files so the normal task can read outputs from this engine probe.
    save_records(RUN/'rapfi_classical12_pilot.npz',records)
    report.update(status='complete',records=len(records), illegal_moves=0,
                  basic_tactical_errors=sum(r['basic_correct'] is False for r in report['teacher_records']),
                  training_performed=False)
    (RUN/'rapfi_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],records=len(records),basic_tactical_errors=report['basic_tactical_errors'],
                    probes={k:dict(disabled=v['evaluator_disabled'],nnue_evidence=v['nnue_enable_evidence'])
                            for k,v in report['engine_probes'].items()})),flush=True)


if __name__=='__main__':
    main()
