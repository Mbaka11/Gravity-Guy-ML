"""
Replay system for visualizing agent gameplay from recorded episodes.
Takes a level seed and sequence of actions, then plays them back visually.
"""

import pygame
import json
import time
from pathlib import Path
from typing import List, Optional, Dict

from src.game.config import (
    WIDTH, HEIGHT, PLAYER_X, PLAYER_H,
    COLOR_BG, COLOR_FG, COLOR_ACCENT, COLOR_PLAT, COLOR_DANGER,
    SCROLL_PX_PER_S
)
from src.game.level import LevelGen
from src.game.player import Player
from src.env.gg_env import GGEnv

class GameReplay:
    def __init__(
        self, 
        level_seed: Optional[int] = None,
        actions: List[int] = None,
        fps: int = 60
    ):
        """
        Initialize the replay system.
        
        Args:
            level_seed: Seed used for level generation (None = random)
            actions: List of actions (0=NOOP, 1=FLIP) to replay
            fps: Frame rate for visualization
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Gravity Guy - Agent Replay")
        
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont("jetbrainsmono", 18)
        
        # Create level and player instances
        self.level = LevelGen(seed=level_seed)
        self.player = Player(
            x=float(PLAYER_X),
            y=HEIGHT / 2 - PLAYER_H / 2,
            vy=0.0,
            grav_dir=1,
            grounded=False,
        )
        
        self.actions = actions or []
        self.current_step = 0
        self.distance_px = 0.0
        
    def run_replay(self):
        """Run the visual replay until all actions are exhausted or user quits."""
        running = True
        paused = False
        dt = 1.0 / self.fps
        
        while running and self.current_step < len(self.actions):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            if not paused:
                # Apply the recorded action
                action = self.actions[self.current_step]
                if action == 1:
                    self.player.try_flip()
                
                # Update game state (similar to game.py)
                self.level.update_and_generate(dt)
                prev_y = self.player.y
                self.player.update_physics(dt)
                plat_rects = [p.rect for p in self.level.platforms]
                self.player.resolve_collisions_swept(prev_y, plat_rects)
                
                self.distance_px += dt * SCROLL_PX_PER_S
                
                # Check for game over conditions (similar to GGEnv)
                if self.player.y < -80 or self.player.y > HEIGHT + 80:
                    print("Game Over: Player out of bounds")
                    break
                
                self.current_step += 1
            
            # Draw everything (similar to game.py)
            self.screen.fill(COLOR_BG)
            self.level.draw(self.screen, COLOR_PLAT)
            
            # Draw player
            color_player = COLOR_ACCENT
            pygame.draw.rect(self.screen, color_player, self.player.rect)
            
            # Draw HUD
            g_txt = "↓" if self.player.grav_dir > 0 else "↑"
            hud = f"Step: {self.current_step}   Dist: {int(self.distance_px)} px   Grav: {g_txt}"
            hud_surface = self.font.render(hud, True, COLOR_FG)
            self.screen.blit(hud_surface, (12, 10))
            
            if paused:
                pause_txt = self.font.render("PAUSED", True, COLOR_DANGER)
                self.screen.blit(pause_txt, (WIDTH - pause_txt.get_width() - 12, 10))
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        # Keep the final frame visible for a moment
        time.sleep(1)
        pygame.quit()

def replay_from_run(run_file: str, episode_idx: int = 0):
    """
    Replay a specific episode from a run file.
    
    Args:
        run_file: Path to the .jsonl file containing run data
        episode_idx: Index of episode to replay (default=0)
    """
    # Read the run file
    run_path = Path(run_file)
    episodes = []
    
    with open(run_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    
    if not episodes:
        print(f"No episodes found in {run_file}")
        return
    
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found. File contains {len(episodes)} episodes.")
        return
    
    episode = episodes[episode_idx]
    
    # Get the recorded actions directly from the episode data
    if 'actions' not in episode:
        print("This recording doesn't contain action history. Please re-run the policy test to generate new recordings.")
        return
        
    actions = episode['actions']
    
    print(f"Replaying episode {episode_idx} with seed {episode['level_seed']}")
    print(f"Policy: {episode['policy']}")
    print(f"Original steps: {episode['steps']}")
    print(f"Original distance: {episode['distance_px']}px")
    print(f"Original flips: {episode['flips']}")
    print(f"Total actions: {len(actions)}")
    print(f"\nPress SPACE to pause/unpause")
    print("Press ESC to quit")
    
    # Create and run the visual replay
    replay = GameReplay(
        level_seed=episode['level_seed'],
        actions=actions,
        fps=60  # Can adjust this for slower/faster replay
    )
    replay.run_replay()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_replay.py <run_file> [episode_index]")
        print("Example: python visualize_replay.py experiments/runs/1234567890_heuristic.jsonl 0")
        sys.exit(1)
    
    run_file = sys.argv[1]
    episode_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    replay_from_run(run_file, episode_idx)
