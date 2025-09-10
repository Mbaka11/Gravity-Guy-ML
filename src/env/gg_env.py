# src/env/gg_env.py
"""
Purpose
-------
Headless, Gym-like wrapper around the Gravity-Guide game so agents can train
without opening a Pygame window. Exposes the minimal reset/step API:

    obs = env.reset(level_seed=None)
    obs, reward, done, info = env.step(action)

Observation uses the helper in `observations.py` (6-D vector).
Actions are discrete: 0=do nothing, 1=flip gravity.

Summary
-------
- Fixed small timestep (dt) for determinism and speed.
- Reward = horizontal progress (scroll distance per step) minus an optional flip penalty.
- Episode ends when the player goes off-screen or when a time limit is reached.
- Seeds:
    * level_seed controls procedural layout (None = random each reset)
    * rng_seed reserved for future stochasticity in the env/agent

Args (constructor)
------------------
level_seed : Optional[int]
    Seed for level generation. None -> random per reset.
rng_seed : Optional[int]
    Seed for env RNG (reserved for future noise/augmentations).
max_time_s : float
    Time limit per episode in seconds (prevents infinite episodes).
flip_penalty : float
    Reward penalty applied when action==1 (flip).
dt : float
    Fixed simulation step (seconds). Smaller -> more accurate, slower.

Returns (API)
-------------
reset(...) -> obs : List[float]
step(action:int) -> (obs: List[float], reward: float, done: bool, info: dict)
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import pygame

from ..game.config import (
    WIDTH, HEIGHT, SCROLL_PX_PER_S, PLAYER_X, PLAYER_H
)
from ..game.level import LevelGen
from ..game.player import Player
from .observations import build_observation

class GGEnv:
    """
    Minimal headless environment for Gravity-Guide.

    Action space: 2 (0=NOOP, 1=FLIP)
    Observation: 6 floats (see observations.build_observation)
    """
    
    def __init__(
        self,
        level_seed: Optional[int] = None,
        rng_seed: Optional[int] = None,
        max_time_s: float = 60.0,
        flip_penalty: float = 0.01,
        dt: float = 1.0 / 120.0,
    ):
        self.level_seed = level_seed
        self.rng_seed = rng_seed
        self.max_time_s = max_time_s
        self.flip_penalty = flip_penalty
        self.dt = dt
        
        # Runtime state set in reset()
        self.level: Optional[LevelGen] = None
        self.player: Optional[Player] = None
        self.time_s: float = 0.0
        self.distance_px: float = 0.0
        self.current_level_seed: Optional[int] = None  # resolved seed used by LevelGen
        
    def reset(self, level_seed: Optional[int] = None) -> List[float]:
        """
        Summary
        -------
        Start a new episode and return the first observation.

        Args
        ----
        level_seed : Optional[int]
            If provided, overrides the environment's default seed for this reset.
            Use None to let the LevelGen randomize the layout.

        Returns
        -------
        List[float]
            The initial observation (6-D vector).
        """
        # Resolve which seed to use for this episode
        seed_spec = self.level_seed if level_seed is None else level_seed

        # Create level (support both LevelGen(seed=...) and LevelGen(...))
        try:
            level = LevelGen(seed=seed_spec)
        except TypeError:
            level = LevelGen(seed_spec)

        # Spawn player roughly mid-height
        player = Player(
            x=float(PLAYER_X),
            y=HEIGHT / 2 - PLAYER_H / 2,
            vy=0.0,
            grav_dir=1,
            grounded=False,
        )

        # Reset episode bookkeeping
        self.level = level
        self.player = player
        self.time_s = 0.0
        self.distance_px = 0.0
        self.current_level_seed = getattr(level, "seed", seed_spec)

        # One collision settle so 'grounded' is consistent if we spawn on a platform
        plat_rects = [p.rect for p in self.level.platforms]
        prev_y = self.player.y
        # Keep the same order we found stable in-game:
        self.player.resolve_collisions_swept(prev_y, plat_rects)

        obs = build_observation(self.player, plat_rects)
        return obs
    
    def step(self, action: int) -> Tuple[List[float], float, bool, Dict]:
        """
        Advance the simulation by one fixed step given an action.
        """
        assert self.level is not None and self.player is not None, "Call reset() first."

        # --- apply action and detect if a flip actually occurred
        prev_grav = self.player.grav_dir
        if action == 1:
            self.player.try_flip()
        did_flip = (self.player.grav_dir != prev_grav)

        # --- update world headlessly
        self.level.update_and_generate(self.dt)

        # --- update player physics
        prev_y = self.player.y
        self.player.update_physics(self.dt)

        # --- resolve collisions with Platform objects (not just rects)
        grounded, collision_occurred = self.player.resolve_collisions_with_platforms(self.level.platforms)

        # --- reward: progress minus penalty only if the flip actually happened
        step_progress = self.dt * SCROLL_PX_PER_S
        reward = step_progress - (self.flip_penalty if did_flip else 0.0)

        # --- episode bookkeeping
        self.time_s += self.dt
        self.distance_px += step_progress
        out_of_bounds = (self.player.y < -80) or (self.player.y > HEIGHT + 80)
        if out_of_bounds:
            reward -= 20.0  # extra penalty for going off-screen
        time_up = self.time_s >= self.max_time_s
        done = bool(out_of_bounds or time_up)

        # --- observation (use platform rects for observation system)
        plat_rects = [p.rect for p in self.level.platforms]
        obs = build_observation(self.player, plat_rects)

        info = {
            "time_s": self.time_s,
            "distance_px": self.distance_px,
            "level_seed": self.current_level_seed,
            "out_of_bounds": out_of_bounds,
            "time_up": time_up,
            "grounded": bool(self.player.grounded),
            "cooldown": max(0.0, float(getattr(self.player, "_flip_cooldown", 0.0))),
            "grav_dir": int(self.player.grav_dir),
            "did_flip": bool(did_flip),
            "probes": obs[3:6],
            "collision_occurred": bool(collision_occurred),  # Added for debugging
            "player_x": float(self.player.x),  # Track horizontal position
            "off_screen_left": bool(off_screen_left),
            "off_screen_right": bool(off_screen_right),
        }
        return obs, float(reward), done, info

    # Small helpers (optional)
    @property
    def action_space_n(self) -> int:
        """Number of discrete actions (2)."""
        return 2

    @property
    def observation_size(self) -> int:
        """Length of the observation vector (6)."""
        return 6