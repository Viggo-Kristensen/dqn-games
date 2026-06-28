import pygame
import torch

from model import QNetwork
from env import SnakeEnv

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 600
CELL_SIZE = SCREEN_WIDTH // 15

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


MODEL_PATH = "DQN_Snake_Final"  # change if using a checkpoint like DQN_Snake.pt.ep200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

env = SnakeEnv(render=True)
state = env.reset()

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        action = torch.argmax(model(state_tensor), dim=1).item()

    state, reward, done, _ = env.step(action)
    env.render()

print("Game over.")
pygame.quit()