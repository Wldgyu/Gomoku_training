from pathlib import Path
import json, torch
from gomoku.models import load_policy
from gomoku.advanced_match_audit import audit
from gomoku.midgame_evaluate import middle_games
from gomoku.history_aux import HistoryTacticalData
from gomoku.train import atomic_json

ROOT=Path("gomoku/runs/offense_e_20261008")
TEST=ROOT/"data/fresh_test_starts.npz"
models={
 "c25_fly":Path("gomoku/runs/iterative_20261008/c_dagger25/fly_seed4810.pt"),
 "c25_gru":Path("gomoku/runs/iterative_20261008/c_dagger25/gru_seed4810.pt"),
 "e_fly":ROOT/"e/fly_seed4810.pt",
 "e_gru":ROOT/"e/gru_seed4810.pt",
}
torch.set_num_threads(4)
data=HistoryTacticalData(TEST,torch.device("cpu"),"history")
rows=[]
examples=[]
for name,path in models.items():
    model,saved=load_policy(path,torch.device("cpu"))
    full,replays=audit(model,4810,games=256,record_all=False,environment_seed=247_000_481)
    middle,_=middle_games(model,TEST,248_000_481,"cpu")
    tactical={"cold":data.evaluate(model,False),"history":data.evaluate(model,True)}
    row={"model":name,"checkpoint":str(path),"checkpoint_config":saved["config"],"full":full,
         "midgames":middle,"tactical":tactical}
    rows.append(row)
    examples.extend({"model":name,**x} for x in replays)
    alls=[x for x in middle if x["family"]=="all"]
    print(json.dumps({"model":name,"wins":full["wins"],"defense_omissions":full["defense_omissions"],
      "multiple_threat_rate":full["multiple_threat_game_rate"],"safe_forks":full["actual_safe_forks_created"],
      "midgames":alls}),flush=True)
atomic_json(ROOT/"evaluation.json",{"full_environment_seed":247000481,"midgame_reply_seed":248000481,
 "fresh_test":str(TEST),"runs":rows,"frozen_policy":"neural argmax; no teacher/search at runtime"})
atomic_json(ROOT/"evaluation_replay_examples.json",{"games":examples})
