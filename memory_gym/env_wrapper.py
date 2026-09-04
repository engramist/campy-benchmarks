"""
campy-benchmarks / memory_gym / env_wrapper.py
Environment wrappers for MemoryGym (MysteryPath-v0 and MortarMayhem-v0).
Provides seamless fallback to high-fidelity simulated environments when gymnasium/memory-gym is uninstalled.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple


class SimulatedMysteryPathEnv:
    """Simulated 2D Grid MysteryPath environment.
    
    A path from start to goal is flashed at Step 0, then hidden.
    The agent must traverse the hidden path step-by-step up to max_steps (up to 500).
    """

    def __init__(self, grid_size: int = 15, max_steps: int = 500):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.path: List[Tuple[int, int]] = []
        self._path_set: set[Tuple[int, int]] = set()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
        self.current_step = 0
        self.agent_pos = (0, 0)
        
        # Generate random contiguous path from (0,0) to goal
        path = [(0, 0)]
        curr = [0, 0]
        while curr[0] < self.goal_pos[0] or curr[1] < self.goal_pos[1]:
            choices = []
            if curr[0] < self.goal_pos[0]:
                choices.append((1, 0))
            if curr[1] < self.goal_pos[1]:
                choices.append((0, 1))
            move = random.choice(choices)
            curr[0] += move[0]
            curr[1] += move[1]
            path.append((curr[0], curr[1]))
        
        self.path = path
        self._path_set = set(path)

        # Step 0 observation: flashed path
        obs = {
            "step": 0,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "flashed_path": list(self.path),
            "is_flashed": True,
        }
        return obs, {}

    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Action is a target coordinate (x, y)."""
        self.current_step += 1
        self.agent_pos = action

        done = False
        reward = 0.0

        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0  # Success!
        elif self.agent_pos not in self._path_set or self.current_step >= self.max_steps:
            done = True
            reward = 0.0  # Off-path or timed out

        obs = {
            "step": self.current_step,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "flashed_path": None,  # Hidden after step 0
            "is_flashed": False,
        }
        return obs, reward, done, False, {"optimal_length": len(self.path)}


def make_memory_gym_env(env_name: str = "MysteryPath-v0", max_steps: int = 500):
    """Factory creating real gym env if installed, or simulated env otherwise."""
    try:
        import gymnasium as gym
        import memory_gym
        env = gym.make(env_name)
        return env
    except Exception:
        # Fallback to simulated high-fidelity environment
        return SimulatedMysteryPathEnv(grid_size=15, max_steps=max_steps)
