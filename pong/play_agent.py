import torch
from env import PongEnv
from model import QNetwork

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
env = PongEnv(render=True)

# Model
state_dim = 5
hidden_dim = 128
n_actions = 3

net = QNetwork(state_dim, hidden_dim, n_actions).to(device)
net.load_state_dict(torch.load("dqn_pong.pt.ep6600", map_location=device))
net.eval()

state = env.reset()

done = False

while True:

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = net(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

    state, reward, done, _ = env.step(action)

    env.render()

    if done:
        print("Episode finished with reward:", reward)

        state = env.reset()
        done = False

env.close()