import random
import numpy as np
import pygame
import collections

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
PADDLE_H = 50
BALL_R = 5

class PongEnv:
    def __init__(self, render=False):
        self.render_mode = render
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_vx = 5 * random.choice([-1, 1])
        self.ball_vy = 5 * random.choice([-1, 1])
        self.agent_y = SCREEN_HEIGHT // 2 - PADDLE_H // 2
        self.bot_y = SCREEN_HEIGHT // 2 - PADDLE_H // 2
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.render_mode:
            pygame.event.pump()  # keep window responsive

        # Actions: 0=stay,1=up,2=down
        if action == 1:
            self.agent_y -= 6
        elif action == 2:
            self.agent_y += 6
        self.agent_y = int(np.clip(self.agent_y, 0, SCREEN_HEIGHT - PADDLE_H))

        # perfect bot
        self.bot_y = int(np.clip(self.ball_y - PADDLE_H // 2, 0, SCREEN_HEIGHT - PADDLE_H))

        # move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # bounce top/bottom
        if self.ball_y <= BALL_R:
            self.ball_y = BALL_R
            self.ball_vy *= -1
        elif self.ball_y >= SCREEN_HEIGHT - BALL_R:
            self.ball_y = SCREEN_HEIGHT - BALL_R
            self.ball_vy *= -1

        reward = 0.0

        # agent paddle collision
        if (self.ball_x - BALL_R) <= (25 + 15) and self.agent_y < self.ball_y < self.agent_y + PADDLE_H:
            self.ball_x = 25 + 15 + BALL_R
            self.ball_vx = abs(self.ball_vx)
            reward = 1.0

        # bot paddle collision
        if (self.ball_x + BALL_R) >= (SCREEN_WIDTH - 40) and self.bot_y < self.ball_y < self.bot_y + PADDLE_H:
            self.ball_x = SCREEN_WIDTH - 40 - BALL_R
            self.ball_vx = -abs(self.ball_vx)

        # out of bounds
        if self.ball_x < 0:
            self.done = True
            reward = -1.0
        elif self.ball_x > SCREEN_WIDTH:
            self.done = True
            reward = 0.5

        return self._get_state(), reward, self.done, {}

    def render(self):
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), BALL_R)
        pygame.draw.rect(self.screen, (255, 255, 255), (25, self.agent_y, 15, PADDLE_H))
        pygame.draw.rect(self.screen, (255, 255, 255), (SCREEN_WIDTH - 40, self.bot_y, 15, PADDLE_H))
        pygame.display.flip()
        self.clock.tick(60)

    def _get_state(self):
        return np.array([
            (self.ball_x - SCREEN_WIDTH / 2) / (SCREEN_WIDTH / 2),
            (self.ball_y - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2),
            self.ball_vx / 10.0,
            self.ball_vy / 10.0,
            (self.agent_y - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2)
        ], dtype=np.float32)

    def close(self):
        if self.render_mode:
            try:
                pygame.display.quit()
            except Exception:
                pass
            pygame.quit()
            self.render_mode = False


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)