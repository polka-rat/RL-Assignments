import torch
import torch.nn as nn
import numpy as np
import random
import wandb

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )

    def forward(self, x):
        return self.layers(x)
    
class Trainer():
    def __init__(self, env, buffer_capacity=125000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.998, lr=75e-5, seed=None, use_wandb=False, wandb_project="rl-pa2", wandb_run_name=None, trunc_length=2000):
        self.env = env
        self.seed = seed
        if self.seed is not None:
            self._set_seed(self.seed)

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = epsilon
        self.lr = lr
        self.trunc_length = trunc_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self._init_q_network(self.policy_net, self.seed)
        self._init_q_network(self.target_net, self.seed)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.update_target_net()
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_run = None
        seed_label = self.seed if self.seed is not None else "none"
        self.episode_returns_title = f"episode_returns_seed_{seed_label}"
        self.episode_returns = []

        if self.seed is not None and hasattr(self.action_space, "seed"):
            self.action_space.seed(self.seed)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _init_q_network(self, network, seed):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def _init_wandb(self, num_episodes, max_steps, target_update_steps, replay_factor):
        if not self.use_wandb:
            return

        env_name = getattr(getattr(self.env, "spec", None), "id", type(self.env).__name__)
        wandb.init(
            entity="randomteamfr",
            project=self.wandb_project,
            name=self.wandb_run_name,
            tags=[
                f"env:{env_name}",
                f"seed:{self.seed}",
                "algo:dqn",
            ],
            config={
                "env_name": env_name,
                "seed": self.seed,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "buffer_capacity": self.buffer.capacity,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "lr": self.lr,
                "epsilon_start": self.initial_epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "num_episodes": num_episodes,
                "max_steps": max_steps,
                "target_update_steps": target_update_steps,
                "replay_factor": replay_factor,
                "trunc_length": self.trunc_length,
                "optimizer": type(self.optimizer).__name__,
                "network_hidden_layers": [32, 32],
                "device": str(self.device),
            },
        )

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(dim=1)[0]
            target_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, num_episodes=1000, target_update_steps=20, replay_factor=1):
        self._init_wandb(num_episodes, self.trunc_length, target_update_steps, replay_factor)
        episode_rewards = []
        self.episode_returns = []
        total_env_steps = 0
        target_updates = 0

        for episode in range(num_episodes):
            reset_seed = None if self.seed is None else self.seed + episode
            state, _ = self.env.reset(seed=reset_seed)
            total_reward = 0.0
            episode_losses = []
            episode_steps = 0

            for _ in range(self.trunc_length):
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_env_steps += 1
                episode_steps += 1
                step_losses = []

                self.buffer.push(state, action, reward, next_state, done)
                for _ in range(replay_factor):
                    loss = self.train_step()
                    if loss is not None:
                        episode_losses.append(loss)
                        step_losses.append(loss)

                if total_env_steps % target_update_steps == 0:
                    self.update_target_net()
                    target_updates += 1

                state = next_state
                total_reward += reward

                if self.use_wandb:
                    step_log_data = {
                        "episode": episode + 1,
                        "episode_steps": episode_steps,
                        "buffer_size": len(self.buffer),
                        "total_env_steps": total_env_steps,
                        "target_updates": target_updates,
                        "epsilon": self.epsilon,
                    }
                    if step_losses:
                        step_log_data["loss"] = float(np.mean(step_losses))
                    wandb.log(step_log_data)

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            episode_rewards.append(total_reward)
            self.episode_returns.append(total_reward)

            if self.use_wandb:
                mean_loss = float(np.mean(episode_losses)) if episode_losses else None
                log_data = {
                    "episode": episode + 1,
                    "episode_return": total_reward,
                    "episode_steps": episode_steps,
                    "episode_length": episode_steps,
                    "epsilon": self.epsilon,
                    "buffer_size": len(self.buffer),
                    "total_env_steps": total_env_steps,
                    "target_updates": target_updates,
                }
                if mean_loss is not None:
                    log_data["episode_mean_loss"] = mean_loss
                wandb.log(log_data)

        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.summary[self.episode_returns_title] = list(self.episode_returns)

        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

        return episode_rewards
