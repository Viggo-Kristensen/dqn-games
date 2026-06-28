import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from itertools import count

from env import FlappyBirdEnv, ReplayBuffer
from model import QNetwork

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 5000
TAU = 0.005
LR = 0.001

env = FlappyBirdEnv(render_mode=True)
n_actions = 2
state = env.reset()
n_observations = state.shape[1]

device = torch.device("cpu")
policy_net = QNetwork(n_observations, n_actions).to(device)
target_net = QNetwork(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayBuffer(10000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 1000

for i_episode in range(num_episodes):
    state = env.reset()
    env.current_episode = i_episode + 1
    state = state.to(device)

    for t in count():
        action = select_action(state)
        observation, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], dtype=torch.float32, device=device)

        next_state = None if done else observation.to(device)

        memory.push(state, action, next_state, reward)

        state = next_state
        optimize_model()

        target_state_dict = target_net.state_dict()
        policy_state_dict = policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key]*TAU + target_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_state_dict)

        if env.render_mode:
            env.render()

        if done:
            if env.score > env.top_score:
                env.top_score = env.score
            print(f"Episode {i_episode + 1} finished after {t + 1} steps, score: {env.score}")
            break

env.close()
print("Training complete")
