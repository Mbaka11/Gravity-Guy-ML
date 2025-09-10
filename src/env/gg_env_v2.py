# src/env/gg_env_v2.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
import pygame

from src.game.config import (
    WIDTH, HEIGHT, FPS,                     # FPS used as a reference; sim_fps is set below
    PLAYER_X, PLAYER_W, PLAYER_H,
    SCROLL_PX_PER_S,
    COLOR_BG, COLOR_PLAT, COLOR_ACCENT, COLOR_DANGER
)
from src.game.level import LevelGen, rect_intersects_triangle_strict
from src.game.player import Player
from src.env.observations_v2 import build_observation_v2


class GGEnv(gym.Env):
    """
    Gravity Guy Gymnasium environment (vector observations).
    - Simulation at 60 Hz (internal).
    - Agent acts every `frame_skip` frames (default 4) -> 15 decisions/sec.
    - Observation v2: shape (15,), float32.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 frame_skip: int = 4,
                 time_limit_seconds: Optional[float] = 30.0):
        super().__init__()
        assert frame_skip >= 1, "frame_skip must be >= 1"
        self.render_mode = render_mode
        self.frame_skip = int(frame_skip)

        # Internal sim timing
        self.sim_fps = 60
        self.dt = 1.0 / self.sim_fps

        # Optional built-in truncation (you can also use a TimeLimit wrapper)
        self.time_limit_decisions = None
        if time_limit_seconds is not None:
            # decisions per second = sim_fps / frame_skip
            self.time_limit_decisions = int(self.sim_fps * time_limit_seconds / self.frame_skip)

        # --- Gym spaces ---
        # Actions: 0 = NOOP, 1 = FLIP
        self.action_space = gym.spaces.Discrete(2)

        # Observations: (15,) float32
        # [y_top_norm, vy_norm, grav,
        #  ceil@120, floor@120, spikeTop@120, spikeBot@120,
        #  ceil@240, floor@240, spikeTop@240, spikeBot@240,
        #  ceil@360, floor@360, spikeTop@360, spikeBot@360]
        low = np.array([0.0, -1.0, -1.0] + [0.0, 0.0, 0.0, 0.0] * 3, dtype=np.float32)
        high = np.array([1.0,  1.0,  1.0] + [1.0, 1.0, 1.0, 1.0] * 3, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Runtime state ---
        self.level: Optional[LevelGen] = None
        self.player: Optional[Player] = None
        self.alive: bool = True
        self.distance_px: float = 0.0
        self.timestep: int = 0                   # number of *decision* steps elapsed
        self.current_seed: Optional[int] = None  # LevelGen's effective seed for this episode
        self.last_grounded: bool = False
        self.death_cause: Optional[str] = None   # "spike" | "oob" | None

        # Rendering
        self.screen = None
        self.clock = None

        # Will be set by Gymnasium when calling reset(seed=...)
        self.np_random = None  # type: ignore[attr-defined]

    # -------------------- Core API --------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)  # initializes self.np_random

        # Seeding policy:
        # - If a seed is provided, use it directly for LevelGen for strict reproducibility.
        # - If not, let LevelGen randomize internally (None).
        level_seed = int(seed) if seed is not None else None

        # Fresh world
        self.level = LevelGen(level_seed)
        self.player = Player(
            x=float(PLAYER_X),
            y=HEIGHT / 2 - PLAYER_H / 2,   # TOP-based convention
            vy=0.0,
            grav_dir=1,
            grounded=False
        )
        # If Player maintains a rect, sync it
        if hasattr(self.player, "rect"):
            self.player.rect.topleft = (PLAYER_X - PLAYER_W // 2, int(self.player.y))

        # Episode bookkeeping
        self.alive = True
        self.death_cause = None
        self.distance_px = 0.0
        self.timestep = 0
        self.current_seed = self.level.seed
        self.last_grounded = False

        # Observation
        obs = self._get_obs()
        info = {"seed": self.current_seed, "distance_px": self.distance_px}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        assert self.level is not None and self.player is not None

        # Apply action once at the start of the decision step
        if action == 1 and self.alive:
            self.player.try_flip()

        died_this_step = False
        grounded = False

        # Simulate frame_skip sub-steps (early exit on death)
        for _ in range(self.frame_skip):
            # Update world
            self.level.update_and_generate(self.dt)
            self.player.update_physics(self.dt)
            grounded, _ = self.player.resolve_collisions_with_platforms(self.level.platforms)

            # Spike (strict) + OOB deaths
            if self._check_spike_death():
                self.alive = False
                self.death_cause = "spike"
                died_this_step = True
            elif self._out_of_bounds():
                self.alive = False
                self.death_cause = "oob"
                died_this_step = True

            # Progress distance every sub-step (regardless of alive/dead)
            self.distance_px += SCROLL_PX_PER_S * self.dt

            if died_this_step:
                break

        # Reward: +1 if alive after this decision; -1 on death (once)
        reward = 1.0 if self.alive else -1.0

        # Termination / truncation
        self.timestep += 1
        terminated = not self.alive
        truncated = False
        if (self.time_limit_decisions is not None) and (self.timestep >= self.time_limit_decisions):
            truncated = True

        obs = self._get_obs()
        info = {
            "distance_px": self.distance_px,
            "timestep": self.timestep,
            "seed": self.current_seed,
            "grounded": grounded,
            "death_cause": self.death_cause
        }

        # Optional on-screen render
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # -------------------- Helpers --------------------

    def _get_obs(self) -> np.ndarray:
        assert self.level is not None and self.player is not None
        plat_rects: List[pygame.Rect] = [p.rect for p in self.level.platforms]
        return build_observation_v2(self.player, plat_rects, self.level.spikes)

    def _player_rect(self) -> pygame.Rect:
        assert self.player is not None
        return pygame.Rect(
            PLAYER_X - PLAYER_W // 2,
            int(self.player.y),   # TOP-based player y
            PLAYER_W,
            PLAYER_H
        )

    def _check_spike_death(self) -> bool:
        """Strict triangle vs rect death test, aligned with the game."""
        assert self.level is not None
        pr = self._player_rect()
        for sp in self.level.spikes:
            tri = sp.world_points()
            if sp.aabb().colliderect(pr) and rect_intersects_triangle_strict(pr, tri):
                return True
        return False

    def _out_of_bounds(self) -> bool:
        assert self.player is not None
        return (self.player.y < -80) or (self.player.y > HEIGHT + 80)

    # -------------------- Rendering --------------------

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Gravity Guy — Gym Env v2")
            self.clock = pygame.time.Clock()

        # Pump minimal event queue so the OS doesn't think we're hung
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # In training, you generally won't be here; ignore quit closes.
                pass

        # Draw current state
        self.screen.fill(COLOR_BG)
        if self.level is not None:
            self.level.draw(self.screen, COLOR_PLAT)

        if self.player is not None:
            color = COLOR_ACCENT if self.alive else COLOR_DANGER
            pygame.draw.rect(self.screen, color, self._player_rect())

        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(self.metadata.get("render_fps", 60))

        if self.render_mode == "rgb_array":
            # Return an (H, W, 3) uint8 array
            arr = pygame.surfarray.array3d(self.screen)  # (W, H, 3)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
