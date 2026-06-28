import random
import collections
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from env import ReplayBuffer, SnakeEnv

def select_action(net, state, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    qvals = net(state_v)
    return int(torch.argmax(qvals, dim=1).item())

def compute_td_loss(batch, policy_net, target_net, device, gamma):
    states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
    dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    expected_q = rewards + next_q_values * gamma * (1.0 - dones)
    return F.mse_loss(q_values, expected_q)

def train(
    num_episodes=5000,
    max_steps_per_episode=2000,
    buffer_capacity=50000,
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    hidden_dim=128,
    target_update_freq=1000,
    start_training_after=1000,
    train_freq=4,
    eps_start=1.0,
    eps_final=0.02,
    eps_decay=20000,
    save_path="DQN_Snake.pt",
    render_every_n=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv(render=False)
    n_actions = 3
    state_dim = 15*15+1
    policy_net = QNetwork(state_dim, hidden_dim, n_actions).to(device)
    target_net = QNetwork(state_dim, hidden_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    replay = ReplayBuffer(buffer_capacity)
    total_steps = 0
    episode_rewards = []

    if render_every_n > 0:
        env_render = SnakeEnv(render=True)
    else:
        env_render = None

    for eps in range(1, num_episodes+1):
        state = env.reset()
        ep_reward = 0.0
        for step in range(max_steps_per_episode):
            total_steps += 1
            epsilon = eps_final + (eps_start - eps_final) * math.exp(-1.0 * total_steps / eps_decay)
            action = select_action(policy_net, state, epsilon, n_actions, device)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            replay.push(state, action, reward, next_state, float(done))
            state = next_state

            if len(replay) > batch_size and total_steps > start_training_after and total_steps % train_freq == 0:
                batch = replay.sample(batch_size)
                loss = compute_td_loss(batch, policy_net, target_net, device, gamma)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if env_render is not None and eps % render_every_n == 0:
                env_render.grid = [row[:] for row in env.grid]
                env_render.head_x = env.head_x
                env_render.head_y = env.head_y
                env_render.snake_body = collections.deque(env.snake_body)
                env_render.dir = env.dir
                env_render.render()

            if done:
                break

        episode_rewards.append(ep_reward)

        if eps % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Episode {eps:4d} | TotalSteps {total_steps:7d} | Epsilon {epsilon:.3f} | AvgReward(last100) {avg_reward:.3f}")

        if eps % 1000 == 0:
            torch.save(policy_net.state_dict(), f"{save_path}.ep{eps}")

    if env_render is not None:
        env_render.close()
    torch.save(policy_net.state_dict(), save_path)
    env.close()
    print("Training complete. Model saved to", save_path)

if __name__ == "__main__":
    train(
        num_episodes=100000,
        batch_size=64,
        lr=1e-4,
        gamma=0.99,
        eps_decay=30000,
        start_training_after=2000,
        render_every_n=1000
    )