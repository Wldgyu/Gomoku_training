import sys, json, torch
from pathlib import Path
from gomoku.renju import train
from gomoku.models import load_policy
d = Path(sys.argv[1]); out = {}
suites = train.suites(d)
for p in sys.argv[2:]:
    model = load_policy(Path(p), torch.device(train.DEVICE))[0]
    model.eval()
    out[Path(p).stem] = {k: {c: round(v["accuracy"], 3) for c, v in train.accuracy(model, s).items()} for k, s in suites.items() if k in ("rapfi_test", "selfplay_test", "quiet_test", "forbid_test", "onpolicy_test", "rapfi2_test", "deep_test", "forbid_black_test", "rapfi3_test", "rapfi4_test")}
    print(Path(p).stem, json.dumps(out[Path(p).stem]), flush=True)




