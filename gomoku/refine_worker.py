"""Read-only search worker. Dataset writes are owned by the parent process."""
import time
import numpy as np
from .robust_long_plan import RobustTeacher


def refine_state(payload):
    arm,index,color,chosen,cells=payload
    start=time.perf_counter()
    teacher=RobustTeacher(node_budget=2000)
    plan=teacher.solve(np.asarray(cells,dtype=np.int8),color,6)
    return dict(arm=arm,index=index,color=color,chosen=chosen,plan=plan,
                nodes=teacher.nodes,seconds=time.perf_counter()-start)
