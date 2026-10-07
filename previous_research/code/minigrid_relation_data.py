"""MiniGrid Memory의 단서·목표 물체를 일관되게 바꾸는 짝 에피소드."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from minigrid.core.world_object import Ball, Key
from minigrid.envs.memory import MemoryEnv

from minigrid_memory_pilot import create_env, masked_cue_observation, tensor_observation

VARIANTS = ((False, False), (True, False), (False, True), (True, True))
EXCLUDED_JUNCTION_X = (7, 13)  # S9와 S15에 해당하는 복도 끝 위치


def make_env(size: int, *, random_length: bool = False):
    if random_length:
        if size != 17:
            raise ValueError("가변 복도는 S17Random에서만 사용합니다")
        return MemoryEnv(size=17, random_length=True, max_steps=100)
    if size in (7, 9, 11, 13):
        return create_env(size, 100)
    return MemoryEnv(size=size, max_steps=100)


def reset_variant(env, seed: int, *, cue_flip: bool = False,
                  target_flip: bool = False) -> tuple[dict, dict]:
    env.reset(seed=seed)
    base = env.unwrapped
    center = base.height // 2
    junction_x = int(base.success_pos[0])
    cue_pos = (1, center - 1)
    top_pos = (junction_x, center - 2)
    bottom_pos = (junction_x, center + 2)
    cue = base.grid.get(*cue_pos)
    top = base.grid.get(*top_pos)
    bottom = base.grid.get(*bottom_pos)
    if not isinstance(cue, (Ball, Key)) or not isinstance(top, (Ball, Key)) or not isinstance(bottom, (Ball, Key)):
        raise TypeError("예상 밖의 단서 또는 갈림길 물체")
    if type(top) is type(bottom):
        raise RuntimeError("갈림길의 두 물체가 같습니다")
    if cue_flip:
        base.grid.set(*cue_pos, Key(cue.color) if isinstance(cue, Ball) else Ball(cue.color))
    if target_flip:
        base.grid.set(*top_pos, bottom)
        base.grid.set(*bottom_pos, top)
    if cue_flip != target_flip:
        base.success_pos, base.failure_pos = base.failure_pos, base.success_pos
    updated_cue = base.grid.get(*cue_pos)
    updated_top = base.grid.get(*top_pos)
    correct_turn = int(base.success_pos[1] > center)
    if (type(updated_cue) is type(updated_top)) != (correct_turn == 0):
        raise RuntimeError("짝 에피소드의 정답과 물체 관계가 일치하지 않습니다")
    return base.gen_obs(), {"junction_x": junction_x, "start_x": int(base.agent_pos[0]),
                            "correct_turn": correct_turn, "cue_type": type(updated_cue).__name__,
                            "cue_flip": cue_flip, "target_flip": target_flip}


def teacher_episode(seed: int, size: int, *, cue_flip: bool = False,
                    target_flip: bool = False, random_length: bool = False
                    ) -> tuple[list[dict], list[int], dict]:
    env = make_env(size, random_length=random_length)
    try:
        observation, meta = reset_variant(env, seed, cue_flip=cue_flip, target_flip=target_flip)
        start_x, junction_x, turn = meta["start_x"], meta["junction_x"], meta["correct_turn"]
        if start_x == 1:
            actions = [2] * (junction_x - 1) + [turn, 2]
        else:
            actions = [1, 1] + [2] * (start_x - 1) + [1, 1]
            actions += [2] * (junction_x - 1) + [turn, 2]
        observations, cue_seen = [], False
        for index, action in enumerate(actions):
            observations.append(observation)
            cue_seen |= not np.array_equal(observation["image"], masked_cue_observation(env)["image"])
            observation, reward, terminated, truncated, _ = env.step(action)
            if truncated or (terminated != (index == len(actions) - 1)):
                raise RuntimeError("짝 에피소드 교사 경로가 예상과 다르게 종료됐습니다")
        if not cue_seen or reward <= 0:
            raise RuntimeError("짝 에피소드에서 단서 관측 또는 성공 보상이 없습니다")
        return observations, actions, meta
    finally:
        env.close()


def build_dataset(seeds: Iterable[int], size: int, *, variants=VARIANTS,
                  random_length: bool = False, excluded_junction_x=()) -> list[dict]:
    rows = []
    for seed in seeds:
        for cue_flip, target_flip in variants:
            observations, actions, meta = teacher_episode(
                seed, size, cue_flip=cue_flip, target_flip=target_flip,
                random_length=random_length
            )
            if meta["junction_x"] in excluded_junction_x:
                break
            images, directions = tensor_observation(observations, torch.device("cpu"))
            rows.append({"images": images, "directions": directions,
                         "actions": torch.tensor(actions), "meta": meta, "seed": seed})
    return rows
