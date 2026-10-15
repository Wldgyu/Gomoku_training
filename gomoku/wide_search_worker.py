"""Isolate root candidate breadth; all later own turns retain the same breadth."""
import time
import numpy as np
from .robust_long_plan import RobustTeacher
from .multistep_tactics import features


def wide_search(payload):
    arm,index,color,chosen,cells,width=payload
    features.cache_clear()
    start=time.perf_counter()
    teacher=RobustTeacher(candidate_limit=3,node_budget=2000,root_candidate_limit=width)
    plan=teacher.solve(np.asarray(cells,dtype=np.int8),color,6)
    return dict(arm=arm,index=index,width=width,color=color,chosen=chosen,
                plan=plan,nodes=teacher.nodes,seconds=time.perf_counter()-start)
