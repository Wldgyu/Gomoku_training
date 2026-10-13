"""Reachable midgame episode resets and detached recurrent prefix restoration."""
from pathlib import Path
import numpy as np
import torch
from .env import LearnerEnv, opponent_actions
from .history_aux import burn_in

class MidgameEnv(LearnerEnv):
    def __init__(self,count,seed,size,opponent,start_data,empty_probability=.5):
        if size != 12:
            raise ValueError("Midgame curriculum currently supports 12x12")
        with np.load(start_data) as d:
            self.pool_histories=d['histories'].copy()
            self.pool_obs=d['obs'].copy()
            self.pool_depth=d['horizon'].copy()
        self.pool_colors=np.where(self.pool_obs[:,3,0,0]>0,1,-1).astype(np.int8)
        self.groups={(depth,color):np.flatnonzero((self.pool_depth==depth)&(self.pool_colors==color))
                     for depth in (1,2,3) for color in (1,-1)}
        if any(not len(g) for g in self.groups.values()):
            raise ValueError("Every depth requires both colors")
        self.empty_probability=empty_probability
        self.level=1
        self.histories=np.full((count,size*size),-1,dtype=np.int16)
        self.start_horizon=np.zeros(count,dtype=np.int8)
        self.start_plies=np.zeros(count,dtype=np.int32)
        self.start_ids=np.full(count,-1,dtype=np.int32)
        self.start_counts={str(i):0 for i in range(4)}
        self.prefix_frames=0
        super().__init__(count,seed,size,opponent)

    def reset(self,rows,flip=True):
        rows=np.asarray(rows,dtype=np.int64)
        super().reset(rows,flip)
        self.histories[rows]=-1
        self.start_horizon[rows]=0
        self.start_ids[rows]=-1
        white=rows[self.colors[rows]==-1]
        self.histories[white,0]=self.games.last[white]
        selected=rows[self.rng.random(len(rows))>=self.empty_probability]
        for color in (1,-1):
            rr=selected[self.colors[selected]==color]
            if not len(rr):
                continue
            ids=self.rng.choice(self.groups[self.level,color],len(rr))
            histories=self.pool_histories[ids]
            obs=self.pool_obs[ids]
            cells=((obs[:,0]-obs[:,1])*color).reshape(len(rr),-1).astype(np.int8)
            self.games.board[rr]=cells
            for slot,c in enumerate((1,-1)):
                self.games.line_counts[rr,slot]=(cells[:,self.games.lines]==c).sum(-1)
            lengths=(histories>=0).sum(1)
            self.games.moves[rr]=lengths
            self.games.last[rr]=histories[np.arange(len(rr)),lengths-1]
            self.histories[rr]=histories
            self.start_horizon[rr]=self.level
            self.start_ids[rr]=ids
        self.start_plies[rows]=self.games.moves[rows]
        for depth in range(4):
            self.start_counts[str(depth)]+=int((self.start_horizon[rows]==depth).sum())

    def step(self,actions):
        rr=np.arange(self.games.count)
        self.histories[rr,self.games.moves]=actions
        self.games.place(actions,self.colors)
        reply=~self.games.finished
        opp=opponent_actions(self.games,-self.colors,self.rng,self.opponent)
        active=np.flatnonzero(reply)
        self.histories[active,self.games.moves[active]]=opp[active]
        self.games.place(opp,-self.colors,reply)
        done=self.games.finished.copy()
        rewards=(self.games.winner*self.colors).astype(np.float32)
        rows=np.flatnonzero(done)
        info=dict(rows=rows,results=rewards[rows].copy(),plies=self.games.moves[rows].copy(),
                  colors=self.colors[rows].copy(),start_horizon=self.start_horizon[rows].copy(),
                  episode_decisions=((self.games.moves[rows]-self.start_plies[rows]+1)//2).copy())
        if len(rows):
            self.reset(rows)
        return rewards,done,info

def restore_prefix_state(model,state,resets,histories):
    rows=torch.nonzero(resets,as_tuple=True)[0]
    if not len(rows):
        return state
    restored=burn_in(model,histories[rows])
    return state.index_copy(0,rows,restored)

def prefix_sequence(model,obs,state,resets,histories):
    """Restore reset prefixes in BOTH actor collection and PPO recomputation."""
    steps,count=obs.shape[:2]
    maps=model.encoder.trunk(obs.flatten(0,1)).reshape(steps,count,16,model.size,model.size)
    features=model.encoder.project(maps.mean((-2,-1)))
    matrix=model.matrix()
    logits,values=[],[]
    zeros=torch.zeros(count,dtype=torch.bool,device=obs.device)
    for t in range(steps):
        state=restore_prefix_state(model,state,resets[t],histories[t])
        modulation,value,state=model.from_features(features[t],state,zeros,matrix)
        logits.append(model.cell_logits(maps[t],modulation))
        values.append(value)
    return torch.stack(logits),torch.stack(values)
