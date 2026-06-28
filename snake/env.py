import random
import collections
import numpy as np
import pygame

from random import randint

# Initial vars
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 600
apple_r = 8
CELL_SIZE = SCREEN_WIDTH // 15

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Actions
# 0 = do nothing
# 1 = turn left
# 2 = turn right
class SnakeEnv:
    def __init__(self, render=False):
        self.render_mode = render
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.grid = [[0 for _ in range(15)] for _ in range(15)]
        self.grid[8][2] = 5
        self.grid[8][1] = 2
        self.grid[8][0] = 2


        self.dir = 2

        # snake
        self.head_x = 3
        self.head_y = 8
        self.previous_head_x = 2
        self.previous_head_y = 8
        self.grid[self.head_y][self.head_x] = 3
        self.snake_length = 4
        self.snake_body = collections.deque(maxlen=self.snake_length)
        #initial snake
        self.snake_body.appendleft([8, 0])
        self.snake_body.appendleft([8, 1])
        self.snake_body.appendleft([8, 2])
        self.snake_body.appendleft([self.head_y, self.head_x])


        # apple
        for i in range(3):
            apple_x = randint(0, 14)
            apple_y = randint(0, 14)
            while self.grid[apple_y][apple_x] != 0:
                apple_x = randint(0, 14)
                apple_y = randint(0, 14)
            self.grid[apple_y][apple_x] = 1

        self.done = False
        return self._get_state()

    def step(self, action):
        if self.render_mode:
            pygame.event.pump()


        if action == 1:
            if self.dir == 1:
                self.dir = 4
            else:
                self.dir -= 1

        if action == 2:
            if self.dir == 4:
                self.dir = 1
            else:
                self.dir += 1

        reward = 0

        self.grid[self.previous_head_y][self.previous_head_x] = 2

        self.grid[self.head_y][self.head_x] = 5
        self.previous_head_x = self.head_x
        self.previous_head_y = self.head_y
        if self.dir == 1:
            self.head_x += 1
        elif self.dir == 2:
            self.head_y += 1
        elif self.dir == 3:
            self.head_x -= 1
        else:
            self.head_y -= 1

        if 0 <= self.head_x <= 14 and 0 <= self.head_y <= 14 and self.grid[self.head_y][self.head_x] != 2:
            if self.grid[self.head_y][self.head_x] == 1:
                reward = 1
                self.snake_length += 1
                self.snake_body = collections.deque(self.snake_body, maxlen=self.snake_length)
                apple_x = randint(0, 14)
                apple_y = randint(0, 14)
                while self.grid[apple_y][apple_x] != 0:
                    apple_x = randint(0, 14)
                    apple_y = randint(0, 14)
                self.grid[apple_y][apple_x] = 1
            reward -= 0.01
            # Remove tail if no apple
            tail_y, tail_x = self.snake_body[-1]
            if self.grid[self.head_y][self.head_x] == 0:
                self.grid[tail_y][tail_x] = 0

            self.grid[self.head_y][self.head_x] = 3
            self.snake_body.appendleft([self.head_y, self.head_x])
        else:
            self.done = True
            reward = -1

        return self._get_state(), reward, self.done, {}

    def render(self):
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        self.screen.fill(WHITE)

        for y in range(15):
            for x in range(15):
                cell = self.grid[y][x]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                if cell == 0:
                    pygame.draw.rect(self.screen, WHITE, rect)
                elif cell == 1:
                    pygame.draw.rect(self.screen, RED, rect)
                elif cell == 2:
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif cell == 3:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif cell == 5:
                    pygame.draw.rect(self.screen, (80, 255, 0), rect)

        pygame.display.flip()
        self.clock.tick(15)

    def _get_state(self):
        state = np.array(self.grid).flatten()
        state = np.append(state, self.dir)
        return state

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