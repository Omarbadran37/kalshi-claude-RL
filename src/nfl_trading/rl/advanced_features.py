"""
Advanced RL Features

Multi-agent training, ensemble methods, transfer learning, and real-time adaptation
for enhanced trading performance.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path

# Import RL components
try:
    from .environment.nfl_trading_gym import NFLTradingGym
    from .agents.trading_agent import TradingAgent, PPOAgent, SACAgent, AgentConfig
    from .training.rl_trainer import RLTrainer, TrainingConfig, EpisodeResult
    from .risk_management.risk_manager import RiskManager
except ImportError:
    # Fallback imports
    pass

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent training"""
    num_agents: int = 4
    agent_types: List[str] = None  # ['PPO', 'SAC', 'PPO', 'SAC']
    diversity_bonus: float = 0.1
    competition_enabled: bool = True
    collaboration_enabled: bool = False
    experience_sharing: bool = True
    population_size: int = 10
    tournament_selection: bool = True
    mutation_rate: float = 0.1


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    ensemble_size: int = 5
    voting_method: str = "weighted"  # "majority", "weighted", "confidence"
    diversity_metric: str = "disagreement"  # "disagreement", "correlation"
    rebalance_frequency: int = 100  # episodes
    min_performance_threshold: float = 0.1


class PopulationBasedTraining:
    """
    Population-based training for evolving trading strategies.

    Maintains a population of agents and uses evolutionary methods
    to discover better hyperparameters and strategies.
    """

    def __init__(self, config: MultiAgentConfig, base_config: AgentConfig):
        self.config = config
        self.base_config = base_config
        self.population: List[Dict[str, Any]] = []
        self.generation = 0
        self.performance_history: List[List[float]] = []

    def initialize_population(self, env: NFLTradingGym) -> List[TradingAgent]:
        """Initialize population with diverse agents"""
        agents = []

        for i in range(self.config.population_size):
            # Create diverse agent configurations
            agent_config = self._mutate_config(self.base_config)

            # Alternate between agent types
            if i % 2 == 0:
                agent = PPOAgent(
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n,
                    config=agent_config,
                    action_type="continuous" if hasattr(env.action_space, 'shape') else "discrete"
                )
            else:
                agent = SACAgent(
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1,
                    config=agent_config
                )

            # Store agent with metadata
            agent_info = {
                'agent': agent,
                'config': agent_config,
                'performance': 0.0,
                'age': 0,
                'parent_ids': [],
                'mutations': []
            }

            self.population.append(agent_info)
            agents.append(agent)

        logger.info(f"Initialized population with {len(agents)} agents")
        return agents

    def _mutate_config(self, base_config: AgentConfig) -> AgentConfig:
        """Create mutated version of agent configuration"""
        config = copy.deepcopy(base_config)

        # Mutate learning rate
        if random.random() < self.config.mutation_rate:
            config.learning_rate *= random.uniform(0.5, 2.0)
            config.learning_rate = np.clip(config.learning_rate, 1e-5, 1e-2)

        # Mutate network architecture
        if random.random() < self.config.mutation_rate:
            config.hidden_dim = random.choice([128, 256, 512, 1024])

        # Mutate batch size
        if random.random() < self.config.mutation_rate:
            config.batch_size = random.choice([32, 64, 128, 256])

        # Mutate gamma
        if random.random() < self.config.mutation_rate:
            config.gamma = random.uniform(0.9, 0.999)

        return config

    def evolve_population(self, performance_scores: List[float]):
        """Evolve population based on performance"""
        self.generation += 1
        self.performance_history.append(performance_scores.copy())

        # Update performance scores
        for i, score in enumerate(performance_scores):
            self.population[i]['performance'] = score
            self.population[i]['age'] += 1

        # Sort by performance
        self.population.sort(key=lambda x: x['performance'], reverse=True)

        # Keep top performers
        survivors = self.population[:self.config.population_size // 2]

        # Create offspring from top performers
        offspring = []
        for i in range(self.config.population_size - len(survivors)):
            parent1, parent2 = random.choices(survivors, k=2)
            child_config = self._crossover_configs(parent1['config'], parent2['config'])
            child_config = self._mutate_config(child_config)

            # Create new agent
            if random.random() < 0.5:
                child_agent = PPOAgent(
                    state_dim=self.population[0]['agent'].state_dim,
                    action_dim=self.population[0]['agent'].action_dim,
                    config=child_config,
                    action_type="continuous"
                )
            else:
                child_agent = SACAgent(
                    state_dim=self.population[0]['agent'].state_dim,
                    action_dim=self.population[0]['agent'].action_dim,
                    config=child_config
                )

            child_info = {
                'agent': child_agent,
                'config': child_config,
                'performance': 0.0,
                'age': 0,
                'parent_ids': [parent1.get('id', 0), parent2.get('id', 0)],
                'mutations': []
            }
            offspring.append(child_info)

        # Update population
        self.population = survivors + offspring

        logger.info(f"Evolution generation {self.generation}: Best performance = {max(performance_scores):.3f}")

    def _crossover_configs(self, config1: AgentConfig, config2: AgentConfig) -> AgentConfig:
        """Create child configuration from two parents"""
        child_config = copy.deepcopy(config1)

        # Randomly choose attributes from each parent
        if random.random() < 0.5:
            child_config.learning_rate = config2.learning_rate
        if random.random() < 0.5:
            child_config.hidden_dim = config2.hidden_dim
        if random.random() < 0.5:
            child_config.batch_size = config2.batch_size
        if random.random() < 0.5:
            child_config.gamma = config2.gamma

        return child_config

    def get_best_agents(self, top_k: int = 3) -> List[TradingAgent]:
        """Get top performing agents"""
        sorted_population = sorted(self.population, key=lambda x: x['performance'], reverse=True)
        return [agent_info['agent'] for agent_info in sorted_population[:top_k]]


class AgentEnsemble:
    """
    Ensemble of trading agents with different voting strategies.

    Combines predictions from multiple agents to improve robustness
    and reduce overfitting.
    """

    def __init__(self, agents: List[TradingAgent], config: EnsembleConfig):
        self.agents = agents
        self.config = config
        self.agent_weights = np.ones(len(agents)) / len(agents)
        self.agent_performance_history: List[List[float]] = [[] for _ in agents]
        self.diversity_scores: List[float] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using ensemble voting"""
        agent_actions = []
        agent_infos = []

        # Get actions from all agents
        for agent in self.agents:
            try:
                action, info = agent.select_action(state, training)
                agent_actions.append(action)
                agent_infos.append(info)
            except Exception as e:
                logger.warning(f"Agent failed to select action: {e}")
                # Use default action if agent fails
                agent_actions.append(np.array([0.0]))
                agent_infos.append({})

        # Combine actions based on voting method
        if self.config.voting_method == "majority":
            ensemble_action = self._majority_vote(agent_actions)
        elif self.config.voting_method == "weighted":
            ensemble_action = self._weighted_vote(agent_actions)
        elif self.config.voting_method == "confidence":
            ensemble_action = self._confidence_vote(agent_actions, agent_infos)
        else:
            ensemble_action = self._weighted_vote(agent_actions)

        # Calculate diversity metrics
        diversity_score = self._calculate_diversity(agent_actions)
        self.diversity_scores.append(diversity_score)

        ensemble_info = {
            'ensemble_method': self.config.voting_method,
            'diversity_score': diversity_score,
            'agent_weights': self.agent_weights.tolist(),
            'num_agents': len(self.agents)
        }

        return ensemble_action, ensemble_info

    def _majority_vote(self, actions: List[np.ndarray]) -> np.ndarray:
        """Simple majority voting for discrete actions"""
        # Convert to discrete decisions (buy/sell/hold)
        discrete_actions = []
        for action in actions:
            if isinstance(action, np.ndarray) and len(action) > 0:
                if action[0] > 0.1:
                    discrete_actions.append(1)  # Buy
                elif action[0] < -0.1:
                    discrete_actions.append(-1)  # Sell
                else:
                    discrete_actions.append(0)  # Hold
            else:
                discrete_actions.append(0)

        # Find majority decision
        majority_action = max(set(discrete_actions), key=discrete_actions.count)
        return np.array([float(majority_action * 0.5)])  # Scale back to continuous

    def _weighted_vote(self, actions: List[np.ndarray]) -> np.ndarray:
        """Weighted voting based on agent performance"""
        weighted_action = np.zeros_like(actions[0])

        for i, action in enumerate(actions):
            weighted_action += self.agent_weights[i] * action

        return weighted_action

    def _confidence_vote(self, actions: List[np.ndarray], infos: List[Dict]) -> np.ndarray:
        """Confidence-weighted voting"""
        confidences = []
        for info in infos:
            # Extract confidence if available, otherwise use 1.0
            confidence = info.get('confidence', info.get('value', 1.0))
            confidences.append(abs(confidence) if confidence is not None else 1.0)

        # Normalize confidences
        total_confidence = sum(confidences)
        if total_confidence > 0:
            confidence_weights = [c / total_confidence for c in confidences]
        else:
            confidence_weights = [1.0 / len(actions)] * len(actions)

        # Weighted average
        weighted_action = np.zeros_like(actions[0])
        for i, action in enumerate(actions):
            weighted_action += confidence_weights[i] * action

        return weighted_action

    def _calculate_diversity(self, actions: List[np.ndarray]) -> float:
        """Calculate diversity among agent actions"""
        if len(actions) < 2:
            return 0.0

        # Calculate pairwise disagreement
        disagreements = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                diff = np.linalg.norm(actions[i] - actions[j])
                disagreements.append(diff)

        return np.mean(disagreements) if disagreements else 0.0

    def update_weights(self, episode_rewards: List[float]):
        """Update agent weights based on recent performance"""
        if len(episode_rewards) != len(self.agents):
            return

        # Store performance history
        for i, reward in enumerate(episode_rewards):
            self.agent_performance_history[i].append(reward)

        # Calculate recent performance (last 50 episodes)
        recent_performance = []
        for history in self.agent_performance_history:
            recent_rewards = history[-50:] if len(history) >= 50 else history
            avg_performance = np.mean(recent_rewards) if recent_rewards else 0.0
            recent_performance.append(avg_performance)

        # Update weights based on performance
        if max(recent_performance) > min(recent_performance):
            # Normalize performance to positive values
            min_perf = min(recent_performance)
            adjusted_performance = [p - min_perf + 0.1 for p in recent_performance]

            # Softmax for weights
            exp_performance = np.exp(np.array(adjusted_performance))
            self.agent_weights = exp_performance / np.sum(exp_performance)
        else:
            # Equal weights if all perform similarly
            self.agent_weights = np.ones(len(self.agents)) / len(self.agents)

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics"""
        return {
            'num_agents': len(self.agents),
            'agent_weights': self.agent_weights.tolist(),
            'recent_diversity': np.mean(self.diversity_scores[-100:]) if len(self.diversity_scores) >= 100 else np.mean(self.diversity_scores),
            'performance_std': [np.std(history[-50:]) if len(history) >= 50 else 0.0 for history in self.agent_performance_history]
        }


class TransferLearning:
    """
    Transfer learning capabilities for adapting agents across different
    game types and market conditions.
    """

    def __init__(self):
        self.source_models: Dict[str, TradingAgent] = {}
        self.adaptation_history: List[Dict[str, Any]] = []

    def save_source_model(self, agent: TradingAgent, domain: str, performance: float):
        """Save a trained agent as a source model for transfer learning"""
        # Create a copy of the agent
        agent_copy = copy.deepcopy(agent)

        self.source_models[domain] = {
            'agent': agent_copy,
            'performance': performance,
            'domain': domain,
            'save_time': pd.Timestamp.now()
        }

        logger.info(f"Saved source model for domain '{domain}' with performance {performance:.3f}")

    def transfer_to_target(
        self,
        target_agent: TradingAgent,
        source_domain: str,
        transfer_method: str = "fine_tuning"
    ) -> TradingAgent:
        """Transfer knowledge from source model to target agent"""
        if source_domain not in self.source_models:
            logger.warning(f"Source domain '{source_domain}' not found")
            return target_agent

        source_info = self.source_models[source_domain]
        source_agent = source_info['agent']

        if transfer_method == "fine_tuning":
            # Transfer learned weights with fine-tuning
            adapted_agent = self._fine_tune_transfer(target_agent, source_agent)
        elif transfer_method == "feature_extraction":
            # Freeze early layers, train only final layers
            adapted_agent = self._feature_extraction_transfer(target_agent, source_agent)
        else:
            logger.warning(f"Unknown transfer method: {transfer_method}")
            adapted_agent = target_agent

        # Record adaptation
        self.adaptation_history.append({
            'source_domain': source_domain,
            'transfer_method': transfer_method,
            'source_performance': source_info['performance'],
            'adaptation_time': pd.Timestamp.now()
        })

        logger.info(f"Applied {transfer_method} transfer from {source_domain}")
        return adapted_agent

    def _fine_tune_transfer(self, target_agent: TradingAgent, source_agent: TradingAgent) -> TradingAgent:
        """Fine-tuning transfer learning"""
        try:
            # For PyTorch agents, copy state dict
            if hasattr(target_agent, 'policy') and hasattr(source_agent, 'policy'):
                # Copy compatible layers
                source_state = source_agent.policy.state_dict()
                target_state = target_agent.policy.state_dict()

                # Transfer compatible parameters
                for name, param in source_state.items():
                    if name in target_state and param.shape == target_state[name].shape:
                        target_state[name] = param.clone()

                target_agent.policy.load_state_dict(target_state)

                # Reduce learning rate for fine-tuning
                for param_group in target_agent.optimizer.param_groups:
                    param_group['lr'] *= 0.1

        except Exception as e:
            logger.warning(f"Fine-tuning transfer failed: {e}")

        return target_agent

    def _feature_extraction_transfer(self, target_agent: TradingAgent, source_agent: TradingAgent) -> TradingAgent:
        """Feature extraction transfer learning"""
        try:
            # Similar to fine-tuning but freeze early layers
            if hasattr(target_agent, 'policy') and hasattr(source_agent, 'policy'):
                source_state = source_agent.policy.state_dict()
                target_state = target_agent.policy.state_dict()

                # Transfer and freeze feature extraction layers
                for name, param in target_agent.policy.named_parameters():
                    if 'feature_extractor' in name and name in source_state:
                        if param.shape == source_state[name].shape:
                            param.data = source_state[name].clone()
                            param.requires_grad = False  # Freeze parameter

        except Exception as e:
            logger.warning(f"Feature extraction transfer failed: {e}")

        return target_agent


class RealTimeAdaptation:
    """
    Real-time adaptation system for adjusting agent behavior
    based on changing market conditions.
    """

    def __init__(self, adaptation_window: int = 100, sensitivity: float = 0.1):
        self.adaptation_window = adaptation_window
        self.sensitivity = sensitivity
        self.performance_history = deque(maxlen=adaptation_window * 2)
        self.market_regime_history = deque(maxlen=adaptation_window)
        self.adaptation_triggers: List[Dict[str, Any]] = []

    def monitor_performance(self, episode_result: EpisodeResult, market_regime: str):
        """Monitor agent performance and market conditions"""
        self.performance_history.append(episode_result.total_return)
        self.market_regime_history.append(market_regime)

        # Check for adaptation triggers
        if len(self.performance_history) >= self.adaptation_window:
            self._check_adaptation_triggers()

    def _check_adaptation_triggers(self):
        """Check if adaptation is needed"""
        recent_performance = list(self.performance_history)[-self.adaptation_window:]
        older_performance = list(self.performance_history)[-2*self.adaptation_window:-self.adaptation_window]

        if len(older_performance) < self.adaptation_window:
            return

        # Performance degradation trigger
        recent_avg = np.mean(recent_performance)
        older_avg = np.mean(older_performance)

        if older_avg > 0 and (recent_avg - older_avg) / older_avg < -self.sensitivity:
            self.adaptation_triggers.append({
                'type': 'performance_degradation',
                'severity': abs((recent_avg - older_avg) / older_avg),
                'timestamp': pd.Timestamp.now()
            })

        # Market regime change trigger
        recent_regimes = list(self.market_regime_history)[-20:]
        if len(set(recent_regimes)) > 1:  # Multiple regimes detected
            self.adaptation_triggers.append({
                'type': 'regime_change',
                'regimes': list(set(recent_regimes)),
                'timestamp': pd.Timestamp.now()
            })

    def should_adapt(self) -> bool:
        """Check if adaptation should be triggered"""
        return len(self.adaptation_triggers) > 0

    def get_adaptation_recommendation(self) -> Dict[str, Any]:
        """Get adaptation recommendations based on triggers"""
        if not self.adaptation_triggers:
            return {}

        latest_trigger = self.adaptation_triggers[-1]

        if latest_trigger['type'] == 'performance_degradation':
            return {
                'action': 'reduce_learning_rate',
                'factor': 0.5,
                'reason': 'Performance degradation detected'
            }
        elif latest_trigger['type'] == 'regime_change':
            return {
                'action': 'increase_exploration',
                'factor': 2.0,
                'reason': 'Market regime change detected'
            }

        return {}

    def clear_triggers(self):
        """Clear adaptation triggers after handling"""
        self.adaptation_triggers.clear()


class MultiAgentTrainer:
    """
    Comprehensive multi-agent training system with competition,
    collaboration, and ensemble methods.
    """

    def __init__(
        self,
        env: NFLTradingGym,
        config: MultiAgentConfig,
        base_agent_config: AgentConfig
    ):
        self.env = env
        self.config = config
        self.base_agent_config = base_agent_config

        # Initialize components
        self.population_trainer = PopulationBasedTraining(config, base_agent_config)
        self.transfer_learning = TransferLearning()
        self.adaptation_system = RealTimeAdaptation()

        # Training state
        self.agents: List[TradingAgent] = []
        self.ensemble: Optional[AgentEnsemble] = None
        self.training_history: List[Dict[str, Any]] = []

    def initialize(self):
        """Initialize multi-agent training system"""
        logger.info("Initializing multi-agent training system...")

        # Create initial population
        self.agents = self.population_trainer.initialize_population(self.env)

        # Create ensemble if configured
        if self.config.num_agents > 1:
            ensemble_config = EnsembleConfig(ensemble_size=min(5, len(self.agents)))
            self.ensemble = AgentEnsemble(self.agents[:ensemble_config.ensemble_size], ensemble_config)

        logger.info(f"Multi-agent system initialized with {len(self.agents)} agents")

    def train_population(self, num_generations: int = 10, episodes_per_generation: int = 100):
        """Train population using evolutionary methods"""
        logger.info(f"Starting population training for {num_generations} generations")

        for generation in range(num_generations):
            logger.info(f"Training generation {generation + 1}/{num_generations}")

            # Evaluate all agents
            performance_scores = []
            for i, agent in enumerate(self.agents):
                # Train agent for specified episodes
                trainer = RLTrainer(
                    env=self.env,
                    agent=agent,
                    config=TrainingConfig(
                        total_episodes=episodes_per_generation,
                        eval_frequency=episodes_per_generation,  # Eval at end
                        log_frequency=episodes_per_generation // 10
                    )
                )

                training_result = trainer.train()
                performance_score = training_result.get('final_performance', {}).get('mean_sharpe', 0.0)
                performance_scores.append(performance_score)

                logger.info(f"Agent {i} performance: {performance_score:.3f}")

            # Evolve population
            self.population_trainer.evolve_population(performance_scores)

            # Update agents with evolved population
            self.agents = [agent_info['agent'] for agent_info in self.population_trainer.population]

            # Update ensemble with best agents
            if self.ensemble:
                best_agents = self.population_trainer.get_best_agents(top_k=self.ensemble.config.ensemble_size)
                self.ensemble.agents = best_agents

            # Record generation results
            generation_result = {
                'generation': generation,
                'best_performance': max(performance_scores),
                'mean_performance': np.mean(performance_scores),
                'diversity': np.std(performance_scores)
            }
            self.training_history.append(generation_result)

        logger.info("Population training completed")

    def get_best_agent(self) -> TradingAgent:
        """Get the best performing agent"""
        return self.population_trainer.get_best_agents(top_k=1)[0]

    def get_ensemble_agent(self) -> Optional[AgentEnsemble]:
        """Get the ensemble agent"""
        return self.ensemble

    def save_population(self, save_dir: str):
        """Save entire population for later use"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save best agents
        best_agents = self.population_trainer.get_best_agents(top_k=5)
        for i, agent in enumerate(best_agents):
            agent.save(str(save_path / f"best_agent_{i}.pt"))

        # Save training history
        with open(save_path / "training_history.pkl", "wb") as f:
            pickle.dump(self.training_history, f)

        logger.info(f"Population saved to {save_dir}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.training_history:
            return {}

        final_generation = self.training_history[-1]

        return {
            'total_generations': len(self.training_history),
            'final_best_performance': final_generation['best_performance'],
            'final_mean_performance': final_generation['mean_performance'],
            'performance_improvement': final_generation['best_performance'] - self.training_history[0]['best_performance'],
            'population_size': self.config.population_size,
            'ensemble_enabled': self.ensemble is not None,
            'transfer_learning_adaptations': len(self.transfer_learning.adaptation_history)
        }