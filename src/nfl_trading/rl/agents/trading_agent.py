"""
RL Trading Agents

Implements PPO and SAC agents for NFL trading decisions with proper neural network
architectures and experience replay.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import deque, namedtuple
from abc import ABC, abstractmethod
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])


@dataclass
class AgentConfig:
    """Configuration for RL agents"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    update_frequency: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.1
    gradient_clip: float = 0.5


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic neural network for trading decisions.
    Combines game features, market data, and portfolio state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        action_type: str = "continuous"
    ):
        super().__init__()

        self.action_type = action_type
        self.action_dim = action_dim

        # Shared feature extraction layers
        layers = []
        input_dim = state_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Actor network (policy)
        if action_type == "continuous":
            # For continuous actions, output mean and log_std
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # For discrete actions, output action probabilities
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic network (value function)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor-critic network"""
        features = self.feature_extractor(state)

        # Get value estimate
        value = self.critic(features)

        # Get action distribution
        if self.action_type == "continuous":
            # Continuous actions: Gaussian distribution
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
            action_dist = Normal(action_mean, action_std)
        else:
            # Discrete actions: Categorical distribution
            action_logits = self.actor(features)
            action_dist = Categorical(logits=action_logits)

        return action_dist, value.squeeze(-1)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from current policy"""
        action_dist, value = self.forward(state)

        if deterministic and self.action_type == "continuous":
            action = action_dist.mean
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)

        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given state-action pairs"""
        action_dist, value = self.forward(state)

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return log_prob, value, entropy


class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Pre-allocate memory for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        if self.size < batch_size:
            batch_size = self.size

        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.BoolTensor(self.dones[indices])
        )

    def __len__(self):
        return self.size


class TradingAgent(ABC):
    """Base class for RL trading agents"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.training_step = 0
        self.episode_rewards = []
        self.episode_losses = []

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action given current state"""
        pass

    @abstractmethod
    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """Update agent parameters"""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save agent parameters"""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load agent parameters"""
        pass

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'mean_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'mean_episode_loss': np.mean(self.episode_losses[-100:]) if self.episode_losses else 0,
            'total_episodes': len(self.episode_rewards)
        }


class PPOAgent(TradingAgent):
    """
    Proximal Policy Optimization (PPO) agent for trading decisions.
    Uses clipped surrogate objective and generalized advantage estimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: AgentConfig,
        action_type: str = "continuous"
    ):
        super().__init__(config)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_type = action_type

        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.gae_lambda = 0.95

        # Networks
        self.policy = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            action_type=action_type
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

        # Experience storage for on-policy learning
        self.experiences = []

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.policy.get_action(state_tensor, deterministic=not training)
            _, value = self.policy.forward(state_tensor)

        action_np = action.cpu().numpy().flatten()

        info = {
            'log_prob': log_prob.item() if log_prob.dim() == 0 else log_prob.cpu().numpy(),
            'value': value.item(),
            'action_raw': action_np
        }

        return action_np, info

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """Store experience for PPO update"""
        experience = Experience(state, action, reward, next_state, done, log_prob, value)
        self.experiences.append(experience)

    def update(self, experiences: Optional[List[Experience]] = None) -> Dict[str, float]:
        """Update PPO agent using collected experiences"""
        if experiences is None:
            experiences = self.experiences

        if len(experiences) < self.config.batch_size:
            return {}

        # Convert experiences to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([e.log_prob for e in experiences]).to(self.device)
        old_values = torch.FloatTensor([e.value for e in experiences]).to(self.device)

        # Compute returns and advantages using GAE
        returns, advantages = self._compute_gae(rewards, old_values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for _ in range(self.ppo_epochs):
            # Forward pass
            log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

            # Policy loss with clipping
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

        # Clear experiences
        self.experiences.clear()
        self.training_step += 1

        # Track losses
        avg_policy_loss = total_policy_loss / self.ppo_epochs
        avg_value_loss = total_value_loss / self.ppo_epochs
        avg_entropy_loss = total_entropy_loss / self.ppo_epochs

        self.episode_losses.append(avg_policy_loss + avg_value_loss)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + avg_value_loss + avg_entropy_loss
        }

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        next_value = 0  # Bootstrap value for terminal state
        next_advantage = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                next_advantage = 0

            # TD error
            delta = rewards[t] + self.config.gamma * next_value - values[t]

            # GAE advantage
            advantages[t] = delta + self.config.gamma * self.gae_lambda * next_advantage

            # Return
            returns[t] = advantages[t] + values[t]

            next_value = values[t]
            next_advantage = advantages[t]

        return returns, advantages

    def save(self, filepath: str):
        """Save PPO agent"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }, filepath)

    def load(self, filepath: str):
        """Load PPO agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']


class SACAgent(TradingAgent):
    """
    Soft Actor-Critic (SAC) agent for trading decisions.
    Uses maximum entropy reinforcement learning for better exploration.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: AgentConfig
    ):
        super().__init__(config)

        if config.device == "cuda" and not torch.cuda.is_available():
            config.device = "cpu"
            self.device = torch.device("cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # SAC hyperparameters
        self.tau = 0.005  # Soft update coefficient
        self.alpha = 0.2  # Entropy regularization coefficient
        self.target_update_interval = 1

        # Networks
        self.actor = self._build_actor().to(self.device)
        self.critic1 = self._build_critic().to(self.device)
        self.critic2 = self._build_critic().to(self.device)
        self.target_critic1 = self._build_critic().to(self.device)
        self.target_critic2 = self._build_critic().to(self.device)

        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.learning_rate)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, state_dim, action_dim)

    def _build_actor(self) -> nn.Module:
        """Build actor network for continuous actions"""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.action_dim * 2)  # mean and log_std
        )

    def _build_critic(self) -> nn.Module:
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self._get_action_and_log_prob(state_tensor, deterministic=not training)

        action_np = action.cpu().numpy().flatten()

        info = {
            'log_prob': log_prob.item(),
            'action_raw': action_np
        }

        return action_np, info

    def _get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability from actor"""
        actor_output = self.actor(state)
        mean, log_std = actor_output.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        if deterministic:
            action = mean
            log_prob = torch.zeros_like(mean)
        else:
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x_t)

            # Correct log probability for tanh squashing
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self, experiences: Optional[List[Experience]] = None) -> Dict[str, float]:
        """Update SAC agent"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critics
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self._update_actor(states)

        # Soft update target networks
        if self.training_step % self.target_update_interval == 0:
            self._soft_update_targets()

        self.training_step += 1

        total_loss = critic_loss + actor_loss
        self.episode_losses.append(total_loss)

        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'total_loss': total_loss
        }

    def _update_critics(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                       next_states: torch.Tensor, dones: torch.Tensor) -> float:
        """Update critic networks"""
        with torch.no_grad():
            next_actions, next_log_probs = self._get_action_and_log_prob(next_states)
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1).float()) * self.config.gamma * target_q

        # Current Q values
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))

        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return (critic1_loss + critic2_loss).item()

    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network"""
        actions, log_probs = self._get_action_and_log_prob(states)

        q1 = self.critic1(torch.cat([states, actions], dim=1))
        q2 = self.critic2(torch.cat([states, actions], dim=1))
        q = torch.min(q1, q2)

        # Actor loss (maximize Q - alpha * entropy)
        actor_loss = (self.alpha * log_probs - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _soft_update_targets(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save SAC agent"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }, filepath)

    def load(self, filepath: str):
        """Load SAC agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']