# src/game/game.py
import sys, argparse, random
import pygame
from pygame import K_SPACE, K_ESCAPE, K_r, K_n
from .config import (
    WIDTH, HEIGHT, FPS,
    COLOR_BG, COLOR_FG, COLOR_ACCENT, COLOR_PLAT, COLOR_DANGER,
    PLAYER_X, PLAYER_W, PLAYER_H, SCROLL_PX_PER_S, SEED_DEFAULT,
    DEBUG_OBS_OVERLAY, OBS_PROBE_OFFSETS
)
from .level import LevelGen
from .player import Player
from src.env.observations import build_observation
from .level import rect_intersects_triangle_strict
from src.env.observations_v2 import build_observation_v2

TEST_OBSERVATIONS_LOGS = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None,
                   help="Level seed. Omit for SEED_DEFAULT, use -1 for random each launch.")
    return p.parse_args()

def run():
    args = parse_args()

    # Resolve seed: None -> use SEED_DEFAULT; -1 -> random
    if args.seed is None:
        launch_seed = SEED_DEFAULT
    elif args.seed == -1:
        launch_seed = None  # signals LevelGen to randomize
    else:
        launch_seed = args.seed

    # Test observations
    _print_timer = 0.0 if TEST_OBSERVATIONS_LOGS else None

    pygame.init()
    pygame.display.set_caption("Gravity Guy — playable baseline")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    def reset_world(seed_spec):
        level = LevelGen(seed_spec)      # seed_spec: int or None
        player = Player(x=float(PLAYER_X), y=HEIGHT/2 - PLAYER_H/2, vy=0.0, grav_dir=1, grounded=False)
        distance_px = 0.0
        alive = True
        # Freeze the actual seed we ended up with (if None → LevelGen randomized it)
        return level, player, distance_px, alive, level.seed

    level, player, distance_px, alive, current_seed = reset_world(launch_seed)
    font = pygame.font.SysFont("jetbrainsmono", 18)

    btn_w, btn_h = 180, 70  # Increased height to fit both messages
    restart_rect = pygame.Rect((WIDTH - btn_w)//2, (HEIGHT - btn_h)//2, btn_w, btn_h)

    while True:
        dt = clock.tick(FPS) / 1000.0
        if dt > 1.0 / 30.0:  # clamp stalls
            dt = 1.0 / 30.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == K_SPACE and alive:
                    player.try_flip()
                if event.key == K_r and not alive:
                    # Restart SAME seed
                    level, player, distance_px, alive, current_seed = reset_world(current_seed)
                if event.key == K_n and not alive:
                    # Restart with NEW RANDOM seed (even if still alive)
                    level, player, distance_px, alive, current_seed = reset_world(None)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and (not alive):
                if restart_rect.collidepoint(event.pos):
                    level, player, distance_px, alive, current_seed = reset_world(current_seed)

        # player_rect = pygame.Rect(PLAYER_X - PLAYER_W//2, int(player.y) - PLAYER_H//2, PLAYER_W, PLAYER_H)

        if alive:
            level.update_and_generate(dt)
            prev_y = player.y
            player.update_physics(dt)
            grounded, collision_occurred = player.resolve_collisions_with_platforms(level.platforms)

            player_rect = pygame.Rect(
                PLAYER_X - PLAYER_W // 2,
                int(player.y),
                PLAYER_W,
                PLAYER_H
            )

            distance_px += dt * SCROLL_PX_PER_S

            # Test observations (if enabled)
            if _print_timer is not None:
                _print_timer -= dt
                if _print_timer <= 0.0:
                    _print_timer = 0.5  # print twice per second
                    plat_rects = [p.rect for p in level.platforms]
                    obs = build_observation(player, plat_rects)
                    # Add collision debug info
                    moving_platforms = sum(1 for p in level.platforms if p.platform_type == "moving")
                    print(f"OBS y={obs[0]:.2f} vy={obs[1]:.2f} g={obs[2]:+.0f} p120={obs[3]:.2f} p240={obs[4]:.2f} p360={obs[5]:.2f} | grounded={player.grounded} | moving_plats={moving_platforms}")
            
            
            for sp in level.spikes:
                tri = sp.world_points()     # <-- sans paramètre
                aabb = sp.aabb()            # <-- sans paramètre
                if aabb.colliderect(player_rect):
                    if rect_intersects_triangle_strict(player_rect, tri):
                        alive = False
                        break
            
            # Check for out-of-bounds death
            out_of_bounds = (player.y < -80) or (player.y > HEIGHT + 80)
            if out_of_bounds:
                alive = False

        # --- Render ---
        screen.fill(COLOR_BG)
        level.draw(screen, COLOR_PLAT)
        color_player = COLOR_ACCENT if alive else COLOR_DANGER
        pygame.draw.rect(screen, color_player, player.rect)

        # HUD shows seed so you can reproduce runs
        g_txt = "↓" if player.grav_dir > 0 else "↑"
        hud = f"Seed: {current_seed}   Dist: {int(distance_px)} px   Grav: {g_txt}   {'ALIVE' if alive else 'DEAD'}"
        screen.blit(font.render(hud, True, COLOR_FG), (12, 10))
        screen.blit(font.render("SPACE flip | ESC quit", True, (160, 180, 210)), (12, 32))

        # ---- Debug Observation v2 Overlay (top-left) ----
        if DEBUG_OBS_OVERLAY:
            # Build the observation using rects + spikes
            plat_rects = [p.rect for p in level.platforms]
            obs = build_observation_v2(player, plat_rects, level.spikes)

            # Draw vertical probe lines at the probe x-positions
            for dx in OBS_PROBE_OFFSETS:
                x = PLAYER_X + dx
                pygame.draw.line(screen, (90, 180, 255), (x, 0), (x, HEIGHT), 1)

            # Compose text lines
            y_norm, vy_norm, grav = float(obs[0]), float(obs[1]), int(obs[2])
            lines = [
                f"OBS v2  y={y_norm:.2f}  vy={vy_norm:.2f}  g={grav:+d}",
            ]
            for i, dx in enumerate(OBS_PROBE_OFFSETS):
                b = 3 + 4*i
                ceil_n, floor_n, spike_top, spike_bot = obs[b:b+4]
                lines.append(
                    f"+{dx:>3}: ceil={ceil_n:.2f}  floor={floor_n:.2f}  T={int(spike_top)}  B={int(spike_bot)}"
                )

            # Optional: a small translucent panel for readability
            panel_w = 320
            panel_h = 18 * (len(lines) + 1)
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel.fill((10, 20, 35, 150))  # RGBA with alpha
            screen.blit(panel, (12, 54))

            # Render the lines
            y0 = 60
            for li, msg in enumerate(lines):
                screen.blit(font.render(msg, True, (200, 220, 255)), (20, y0 + li*18))

        if not alive:
            pygame.draw.rect(screen, (40, 60, 90), restart_rect, border_radius=10)
            pygame.draw.rect(screen, (90, 130, 180), restart_rect, width=2, border_radius=10)
            
            # First button text (Restart)
            btn_txt = font.render("Restart (R)", True, (220, 235, 255))
            screen.blit(btn_txt, (restart_rect.centerx - btn_txt.get_width()//2,
                                  restart_rect.centery - btn_txt.get_height() - 5))  # Moved up
            
            # Second button text (New Random)
            btn_txt2 = font.render("New Random (N)", True, (220, 235, 255))
            screen.blit(btn_txt2, (restart_rect.centerx - btn_txt2.get_width()//2,
                                   restart_rect.centery + 5))  # Moved down

        pygame.display.flip()

if __name__ == "__main__":
    run()
