from pathlib import Path
import itertools, json
import numpy as np
import torch
from gomoku.models import load_policy
from gomoku.evaluate import opening_bank, match
from gomoku.train import atomic_json

ROOT=Path("gomoku/runs/offense_e_20261008")
paths={
 "C25-Fly":Path("gomoku/runs/iterative_20261008/c_dagger25/fly_seed4810.pt"),
 "C25-GRU":Path("gomoku/runs/iterative_20261008/c_dagger25/gru_seed4810.pt"),
 "E-Fly":ROOT/"e/fly_seed4810.pt",
 "E-GRU":ROOT/"e/gru_seed4810.pt",
}
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
device=torch.device("cuda")
models={name:load_policy(path,device)[0].eval() for name,path in paths.items()}
seed=249_000_481
openings=opening_bank(128,seed,12)
atomic_json(ROOT/"league_openings.json",{"size":12,"seed":seed,"two_ply_openings":openings.tolist()})
pairs=[]; replays=[]
standings={name:{"games":0,"wins":0,"draws":0,"losses":0,"points":0.0} for name in models}
for a,b in itertools.combinations(models,2):
    colors=[]
    for color in (1,-1):
        result,examples=match(models[a],models[b],openings,color)
        colors.append(result)
        replays.extend({"a":a,"b":b,"opening_seed":seed,**x} for x in examples)
    total={k:sum(x[k] for x in colors) for k in ("games","wins","draws","losses")}
    total.update(a=a,b=b,colors=colors,score=(total["wins"]+0.5*total["draws"])/total["games"],mean_plies=float(np.mean([x["mean_plies"] for x in colors])))
    pairs.append(total)
    sa,sb=standings[a],standings[b]
    sa["games"]+=total["games"]; sa["wins"]+=total["wins"]; sa["draws"]+=total["draws"]; sa["losses"]+=total["losses"]; sa["points"]+=total["wins"]+0.5*total["draws"]
    sb["games"]+=total["games"]; sb["wins"]+=total["losses"]; sb["draws"]+=total["draws"]; sb["losses"]+=total["wins"]; sb["points"]+=total["losses"]+0.5*total["draws"]
    print(json.dumps({"a":a,"b":b,**total}),flush=True)
for name,row in standings.items():
    row["score_rate"]=row["points"]/row["games"]
ranked=sorted((dict(model=k,**v) for k,v in standings.items()),key=lambda x:(x["score_rate"],x["wins"]),reverse=True)
result={"rule":"12x12 freestyle; five or more wins; no forbidden moves","policy":"frozen deterministic argmax; no search","opening_seed":seed,"paired_two_ply_openings":128,"games_per_pair":256,"total_games":sum(x["games"] for x in pairs),"pairs":pairs,"standings":ranked}
atomic_json(ROOT/"league.json",result)
atomic_json(ROOT/"league_replay_examples.json",{"games":replays})
print(json.dumps({"standings":ranked,"total_games":result["total_games"]}),flush=True)
