from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class RainGridEnv(gym.Env):
    """5x5 grid world with higher costs in lower rows (rain accumulation)."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 8,
    }

    def __init__(
        self,
        grid_size: int = 5,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        cell_size: int = 80,
        random_start: bool = False,
        random_goal: bool = False,
    ) -> None:
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.grid_size = int(grid_size)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.cell_size = int(cell_size)
        self.random_start = bool(random_start)
        self.random_goal = bool(random_goal)

        self.action_space = spaces.Discrete(4)  # 0=up, 1=right, 2=down, 3=left
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(2,),
            dtype=np.int64,
        )

        rows = np.arange(self.grid_size, dtype=np.float32)
        row_costs = 1.0 + rows  # higher cost as we move down
        self.cost_grid = np.repeat(row_costs[:, None], self.grid_size, axis=1)

        self.agent_pos = np.array([0, 0], dtype=np.int64)
        self.start_pos = np.array([0, 0], dtype=np.int64)
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int64)
        self.step_count = 0

        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None

    def _get_obs(self) -> np.ndarray:
        return self.agent_pos.copy()

    def _get_info(self) -> dict:
        r, c = int(self.agent_pos[0]), int(self.agent_pos[1])
        return {"cell_cost": float(self.cost_grid[r, c])}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self.random_goal:
            goal_col = int(self.np_random.integers(0, self.grid_size))
            self.goal_pos = np.array([self.grid_size - 1, goal_col], dtype=np.int64)
        else:
            self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int64)

        if self.random_start:
            start_col = int(self.np_random.integers(0, self.grid_size))
            self.start_pos = np.array([0, start_col], dtype=np.int64)
        else:
            self.start_pos = np.array([0, 0], dtype=np.int64)
        self.agent_pos = self.start_pos.copy()
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        dr = dc = 0
        if action == 0:  # up
            dr = -1
        elif action == 1:  # right
            dc = 1
        elif action == 2:  # down
            dr = 1
        elif action == 3:  # left
            dc = -1

        new_r = int(np.clip(self.agent_pos[0] + dr, 0, self.grid_size - 1))
        new_c = int(np.clip(self.agent_pos[1] + dc, 0, self.grid_size - 1))
        self.agent_pos = np.array([new_r, new_c], dtype=np.int64)

        self.step_count += 1
        cell_cost = float(self.cost_grid[new_r, new_c])
        reward = -cell_cost

        terminated = (new_r == int(self.goal_pos[0])) and (new_c == int(self.goal_pos[1]))
        truncated = False
        if self.max_steps is not None and self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode is None:
            return None

        if self._pygame is None:
            import pygame

            self._pygame = pygame
            pygame.init()
            if self.render_mode == "human":
                width = self.grid_size * self.cell_size
                height = self.grid_size * self.cell_size
                self._screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Rain Grid World")
                self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont(None, int(self.cell_size * 0.35))

        pygame = self._pygame
        width = self.grid_size * self.cell_size
        height = self.grid_size * self.cell_size

        if self.render_mode == "human":
            surface = self._screen
        else:
            surface = pygame.Surface((width, height))

        self._draw_grid(surface)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])
            return None

        rgb_array = pygame.surfarray.array3d(surface)
        return np.transpose(rgb_array, (1, 0, 2))

    def _draw_grid(self, surface) -> None:
        pygame = self._pygame
        surface.fill((30, 30, 30))

        top_color = np.array([190, 220, 255], dtype=np.float32)
        bottom_color = np.array([40, 90, 160], dtype=np.float32)

        for r in range(self.grid_size):
            mix = r / max(1, self.grid_size - 1)
            color = top_color * (1 - mix) + bottom_color * mix
            color = tuple(int(c) for c in color)
            for c in range(self.grid_size):
                x = c * self.cell_size
                y = r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (20, 20, 20), rect, 2)

                if self._font is not None:
                    cost_text = self._font.render(
                        str(int(self.cost_grid[r, c])), True, (15, 15, 15)
                    )
                    text_rect = cost_text.get_rect(center=rect.center)
                    surface.blit(cost_text, text_rect)

        goal_rect = pygame.Rect(
            int(self.goal_pos[1]) * self.cell_size,
            int(self.goal_pos[0]) * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(surface, (80, 160, 80), goal_rect, 4)

        start_rect = pygame.Rect(
            int(self.start_pos[1]) * self.cell_size,
            int(self.start_pos[0]) * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(surface, (200, 60, 60), start_rect, 4)

        agent_center = (
            int(self.agent_pos[1] * self.cell_size + self.cell_size / 2),
            int(self.agent_pos[0] * self.cell_size + self.cell_size / 2),
        )
        pygame.draw.circle(surface, (240, 120, 40), agent_center, int(self.cell_size * 0.3))
        pygame.draw.circle(surface, (25, 25, 25), agent_center, int(self.cell_size * 0.3), 2)

    def close(self) -> None:
        if self._pygame is not None:
            if self._screen is not None:
                self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
