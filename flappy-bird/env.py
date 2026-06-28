import pygame
import random
import torch
from collections import namedtuple, deque

class FlappyBirdEnv:
    def __init__(self, render_mode=False):
        pygame.init()
        self.WIDTH, self.HEIGHT = 400, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT)) if render_mode else None
        self.clock = pygame.time.Clock()

        self.gravity = 0.5
        self.flap_power = -9
        self.pipe_gap = 200
        self.pipe_width = 60
        self.pipe_speed = 3
        self.pipe_freq = 90

        self.bird_x = 80
        self.bird_radius = 15

        self.render_mode = render_mode
        self.current_episode = 0
        self.top_score = 0

        self.reset()

    def reset(self):
        self.bird_y = self.HEIGHT // 2
        self.bird_velocity = 0
        self.frame = 0
        self.score = 0
        self.pipes = [self._create_pipe(safe_top=200)]
        self.done = False
        return self._get_state()

    def _create_pipe(self, safe_top=None):
        top = safe_top if safe_top else random.randint(120, self.HEIGHT - self.pipe_gap - 120)
        return {'x': self.WIDTH + self.pipe_width // 2, 'top': top, 'bottom': top + self.pipe_gap, 'passed': False}

    def _move_pipes(self):
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed

    def _check_collision(self):
        for pipe in self.pipes:
            if pipe['x'] < self.bird_x + self.bird_radius < pipe['x'] + self.pipe_width:
                if self.bird_y - self.bird_radius < pipe['top'] or self.bird_y + self.bird_radius > pipe['bottom']:
                    return True
        return self.bird_y - self.bird_radius < 0 or self.bird_y + self.bird_radius > self.HEIGHT

    def _get_state(self):
        pipe = next((p for p in self.pipes if p['x'] + self.pipe_width > self.bird_x), None)
        if pipe is None:
            pipe = {'x': self.WIDTH, 'top': self.HEIGHT // 2, 'bottom': (self.HEIGHT // 2 + self.pipe_gap)}
        dx = pipe['x'] - self.bird_x
        dy = (pipe['top'] + self.pipe_gap // 2) - self.bird_y
        return torch.tensor([dx / self.WIDTH, dy / self.HEIGHT, self.bird_velocity / 10.0], dtype=torch.float32).unsqueeze(0)

    def step(self, action):
        if action == 1:
            self.bird_velocity = self.flap_power

        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity

        if self.frame % self.pipe_freq == 0:
            self.pipes.append(self._create_pipe())
        self._move_pipes()
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + self.pipe_width > 0]

        if self._check_collision():
            self.done = True
            reward = -1.0
        else:
            reward = 0.1
            pipe = next((p for p in self.pipes if p['x'] + self.pipe_width > self.bird_x), None)
            if pipe:
                distance_to_pipe = pipe['x'] - self.bird_x
                proximity_reward = max(0, (self.WIDTH - distance_to_pipe) / self.WIDTH * 0.05)
                reward += proximity_reward

            for pipe in self.pipes:
                if pipe['x'] + self.pipe_width < self.bird_x and not pipe['passed']:
                    pipe['passed'] = True
                    self.score += 1
                    reward = 1.0

        self.frame += 1
        state = self._get_state()
        return state, reward, self.done, {}

    def render(self):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        self.screen.fill((135, 206, 250))  # Sky

        pygame.draw.circle(self.screen, (255, 255, 255), (self.bird_x, int(self.bird_y)), self.bird_radius)
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 200, 0), (pipe['x'], 0, self.pipe_width, pipe['top']))
            pygame.draw.rect(self.screen, (0, 200, 0), (pipe['x'], pipe['bottom'], self.pipe_width, self.HEIGHT))

        if self.current_episode:
            self.draw_text(f"Episode: {self.current_episode}", 10, 10, align="right")
            self.draw_text(f"Score: {self.score}", 10, 40, align="right")
            self.draw_text(f"Top Score: {self.top_score}", 10, 70, align="right")

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def draw_text(self, text, x, y, size=28, color=(0, 0, 0), align="left"):
        font = pygame.font.SysFont(None, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "right":
            x = self.WIDTH - x - text_rect.width
        elif align == "center":
            x = x - text_rect.width // 2
        self.screen.blit(text_surface, (x, y))


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
