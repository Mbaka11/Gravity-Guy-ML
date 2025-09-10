# src/env/obs_v2_unit.py
import numpy as np, pygame
from typing import List
from src.env.observations_v2 import build_observation_v2
from src.game.config import HEIGHT, WIDTH, PLAYER_X, PLAYER_H

class DummyPlayer:
    def __init__(self, y, vy=0.0, grav_dir=1):
        self.y = y        # TOP-based
        self.vy = vy
        self.grav_dir = grav_dir

class DummyPlat:  # minimal wrapper to emulate Spike.platform
    def __init__(self, rect: pygame.Rect): self.rect = rect

def make_rects() -> List[pygame.Rect]:
    # One top platform near y=120, thickness 20; one bottom near y=HEIGHT-140
    top = pygame.Rect(0, 120, WIDTH, 20)
    bot = pygame.Rect(0, HEIGHT-140, WIDTH, 20)
    return [top, bot]

def make_spikes(platform_rects):
    # One spike on top lane around x=PLAYER_X+120; one on bottom around x=PLAYER_X+360
    plat_top = DummyPlat(platform_rects[0])
    plat_bot = DummyPlat(platform_rects[1])
    return [
        type("SpikeStub", (), {"platform": plat_top, "lane": "top", "local_x": (PLAYER_X+120 - plat_top.rect.left)})(),
        type("SpikeStub", (), {"platform": plat_bot, "lane": "bot", "local_x": (PLAYER_X+360 - plat_bot.rect.left)})(),
    ]

def main():
    pygame.init()  # safe even if we only use Rects
    plats = make_rects()
    spikes = make_spikes(plats)
    player = DummyPlayer(y=HEIGHT/2 - PLAYER_H/2, vy=0.0, grav_dir=1)

    obs = build_observation_v2(player, plats, spikes)
    print("obs:", obs)
    assert isinstance(obs, np.ndarray) and obs.dtype == np.float32 and obs.shape == (15,), "Shape/dtype mismatch"

    # range checks
    assert 0.0 <= obs[0] <= 1.0, "y_top_norm out of range"
    assert -1.0 <= obs[1] <= 1.0, "vy_norm out of range"
    assert obs[2] in (-1.0, 1.0), "grav_dir invalid"

    # probe blocks: [ceil, floor, spikeTop, spikeBot] × 3
    for i in range(3):
        b = 3 + 4*i
        ceil, floor, st, sb = obs[b:b+4]
        assert 0.0 <= ceil <= 1.0 and 0.0 <= floor <= 1.0, "ceil/floor out of range"
        assert st in (0.0, 1.0) and sb in (0.0, 1.0), "spike flags not 0/1"

    # sanity: we placed spikes at 120 (top) and 360 (bot)
    assert obs[3+2] == 1.0, "Expected top spike at +120"
    assert obs[3+8+3] == 1.0, "Expected bot spike at +360"

    print("✓ obs_v2 unit sanity passed")

if __name__ == "__main__":
    main()
