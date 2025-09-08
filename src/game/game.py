# src/game/game.py
import sys, argparse, random
import pygame
from pygame import K_SPACE, K_ESCAPE, K_r, K_n
from .config import (
    WIDTH, HEIGHT, FPS,
    COLOR_BG, COLOR_FG, COLOR_ACCENT, COLOR_PLAT, COLOR_DANGER,
    PLAYER_X, PLAYER_W, PLAYER_H, SCROLL_PX_PER_S, SEED_DEFAULT
)
from .level import LevelGen
from .player import Player
from src.env.observations import build_observation


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

    btn_w, btn_h = 220, 80  # wider and taller to fit both options
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

        if alive:
            level.update_and_generate(dt)
            prev_y = player.y
            player.update_physics(dt)
            plat_rects = [p.rect for p in level.platforms]
            # your collision calls here (vertical swept, then side resolve)...
            grounded, _ = player.resolve_collisions_swept(prev_y, plat_rects)
            _ = player.resolve_side_collisions(plat_rects)

            distance_px += dt * SCROLL_PX_PER_S

            # Out of bounds check (x or y)
            out_of_bounds = (
                player.y < -80 or player.y > HEIGHT + 80 or
                player.x < -PLAYER_W or player.x > WIDTH + PLAYER_W
            )
            if out_of_bounds:
                alive = False

            # TEST observations
            if _print_timer is not None:
                _print_timer -= dt
                if _print_timer <= 0.0:
                    _print_timer = 0.5  # print twice per second
                    plat_rects = [p.rect for p in level.platforms]
                    obs = build_observation(player, plat_rects)
                    # pretty debug: y, vy, g, probes
                    print(f"OBS y={obs[0]:.2f} vy={obs[1]:.2f} g={obs[2]:+.0f} p120={obs[3]:.2f} p240={obs[4]:.2f} p360={obs[5]:.2f} | grounded={player.grounded}")

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

        if not alive:
            pygame.draw.rect(screen, (40, 60, 90), restart_rect, border_radius=10)
            pygame.draw.rect(screen, (90, 130, 180), restart_rect, width=2, border_radius=10)
            btn_txt = font.render("Restart (R)", True, (220, 235, 255))
            btn_txt2 = font.render("New Random (N)", True, (220, 235, 255))
            # Center both options vertically in the larger box
            total_height = btn_txt.get_height() + btn_txt2.get_height() + 12
            y_start = restart_rect.centery - total_height // 2
            screen.blit(btn_txt, (restart_rect.centerx - btn_txt.get_width()//2, y_start))
            screen.blit(btn_txt2, (restart_rect.centerx - btn_txt2.get_width()//2, y_start + btn_txt.get_height() + 12))

        pygame.display.flip()

if __name__ == "__main__":
    run()
