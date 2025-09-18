"""
RL Training Framework

Comprehensive training system for NFL trading RL agents with curriculum learning,
evaluation, hyperparameter optimization, and model management.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from pathlib import Path
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque
import optuna
from torch.utils.tensorboard import SummaryWriter

# Import RL components
try:
    from ..environment.nfl_trading_gym import NFLTradingGym
    from ..agents.trading_agent import TradingAgent, PPOAgent, SACAgent, AgentConfig, Experience
except ImportError:
    from nfl_trading_gym import NFLTradingGym
    from trading_agent import TradingAgent, PPOAgent, SACAgent, AgentConfig, Experience

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for RL training"""
    # Training parameters
    total_episodes: int = 5000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 100
    eval_episodes: int = 10
    save_frequency: int = 500
    log_frequency: int = 10

    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_episodes_per_stage: int = 500
    curriculum_success_threshold: float = 0.1  # Success rate to advance

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 1000
    early_stopping_min_improvement: float = 0.01

    # Hyperparameter optimization
    hyperopt_enabled: bool = False
    hyperopt_trials: int = 50
    hyperopt_episodes_per_trial: int = 500

    # Paths
    output_dir: str = "rl_training_output"
    model_save_dir: str = "models"
    log_dir: str = "logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EpisodeResult:
    """Results from a single episode"""
    episode: int
    total_reward: float
    episode_length: int
    final_portfolio_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    transaction_costs: float
    win_rate: float
    agent_stats: Dict[str, Any]


class CurriculumScheduler:
    """Manages curriculum learning progression"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage = 0
        self.episodes_in_stage = 0
        self.stage_results = deque(maxlen=100)

        # Define curriculum stages
        self.stages = [
            {
                'name': 'basic_trading',
                'description': 'Learn basic buy/sell decisions',
                'max_position_size': 100,
                'transaction_cost_bps': 10.0,
                'risk_penalty_factor': 0.2,
                'action_type': 'discrete'
            },
            {
                'name': 'position_sizing',
                'description': 'Learn optimal position sizing',
                'max_position_size': 300,
                'transaction_cost_bps': 7.0,
                'risk_penalty_factor': 0.15,
                'action_type': 'continuous'
            },
            {
                'name': 'advanced_trading',
                'description': 'Full complexity trading',
                'max_position_size': 500,
                'transaction_cost_bps': 7.0,
                'risk_penalty_factor': 0.1,
                'action_type': 'continuous'
            }
        ]

    def get_current_stage(self) -> Dict[str, Any]:
        """Get current curriculum stage configuration"""
        return self.stages[min(self.current_stage, len(self.stages) - 1)]

    def should_advance_stage(self) -> bool:
        """Check if agent should advance to next stage"""
        if not self.config.curriculum_enabled:
            return False

        if self.current_stage >= len(self.stages) - 1:
            return False

        if self.episodes_in_stage < self.config.curriculum_episodes_per_stage:
            return False

        # Check if success threshold is met
        if len(self.stage_results) < 50:
            return False

        success_rate = sum(1 for r in self.stage_results if r > self.config.curriculum_success_threshold) / len(self.stage_results)
        return success_rate >= 0.7  # 70% success rate

    def advance_stage(self):
        """Advance to next curriculum stage"""
        self.current_stage += 1
        self.episodes_in_stage = 0
        self.stage_results.clear()
        logger.info(f"Advanced to curriculum stage {self.current_stage}: {self.get_current_stage()['name']}")

    def update(self, episode_result: EpisodeResult):
        """Update curriculum scheduler with episode result"""
        self.episodes_in_stage += 1
        self.stage_results.append(episode_result.total_return)

        if self.should_advance_stage():
            self.advance_stage()


class RLTrainer:
    """
    Comprehensive RL training framework for NFL trading agents.

    Features:
    - Stable training loop with logging
    - Curriculum learning
    - Evaluation and model selection
    - Hyperparameter optimization
    - Checkpointing and recovery
    """

    def __init__(
        self,
        env: NFLTradingGym,
        agent: TradingAgent,
        config: TrainingConfig,
        eval_env: Optional[NFLTradingGym] = None
    ):
        self.env = env
        self.agent = agent
        self.config = config
        self.eval_env = eval_env or env

        # Setup directories
        self.setup_directories()

        # Training state
        self.current_episode = 0
        self.best_eval_score = -np.inf
        self.no_improvement_count = 0
        self.training_start_time = None

        # Curriculum learning
        self.curriculum = CurriculumScheduler(config)

        # Results tracking
        self.training_results: List[EpisodeResult] = []
        self.eval_results: List[EpisodeResult] = []
        self.training_logs: Dict[str, List[float]] = {
            'episode_rewards': [],
            'episode_lengths': [],
            'portfolio_values': [],
            'total_returns': [],
            'sharpe_ratios': [],
            'agent_losses': []
        }

        # Tensorboard logging
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        logger.info(f"RLTrainer initialized with {type(agent).__name__}")

    def setup_directories(self):
        """Create necessary directories for training output"""
        self.output_dir = Path(self.config.output_dir)
        self.model_dir = self.output_dir / self.config.model_save_dir
        self.log_dir = self.output_dir / self.config.log_dir

        for directory in [self.output_dir, self.model_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        """
        Main training loop with curriculum learning and evaluation.

        Returns:
            Training summary with final performance metrics
        """
        logger.info("Starting RL training...")
        self.training_start_time = time.time()

        try:
            for episode in range(self.config.total_episodes):
                self.current_episode = episode

                # Update environment based on curriculum
                self.update_env_for_curriculum()

                # Run training episode
                episode_result = self.run_episode(training=True)
                self.training_results.append(episode_result)

                # Update curriculum
                self.curriculum.update(episode_result)

                # Log training progress
                self.log_training_progress(episode_result)

                # Evaluation
                if (episode + 1) % self.config.eval_frequency == 0:
                    eval_results = self.evaluate()
                    self.log_evaluation_results(eval_results)

                    # Model selection and saving
                    self.update_best_model(eval_results)

                # Regular model saving
                if (episode + 1) % self.config.save_frequency == 0:
                    self.save_checkpoint(episode)

                # Early stopping check
                if self.should_early_stop():
                    logger.info(f"Early stopping at episode {episode}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            self.cleanup_training()

        # Generate final training report
        training_summary = self.generate_training_summary()
        logger.info("Training completed successfully!")

        return training_summary

    def run_episode(self, training: bool = True) -> EpisodeResult:
        """Run a single episode and return results"""
        env = self.env if training else self.eval_env
        state, info = env.reset()

        episode_reward = 0
        episode_length = 0
        episode_experiences = []

        for step in range(self.config.max_steps_per_episode):
            # Select action
            action, action_info = self.agent.select_action(state, training=training)

            # Execute action
            next_state, reward, done, truncated, step_info = env.step(action)

            # Store experience for on-policy agents (PPO)
            if training and isinstance(self.agent, PPOAgent):
                self.agent.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done or truncated,
                    log_prob=action_info.get('log_prob', 0.0),
                    value=action_info.get('value', 0.0)
                )

            # Store experience for off-policy agents (SAC)
            elif training and isinstance(self.agent, SACAgent):
                self.agent.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done or truncated
                )

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done or truncated:
                break

        # Update agent (for PPO after episode, for SAC during episode)
        agent_stats = {}
        if training:
            if isinstance(self.agent, PPOAgent) and len(self.agent.experiences) > 0:
                agent_stats = self.agent.update()
            elif isinstance(self.agent, SACAgent):
                # SAC updates during episodes, just get stats
                agent_stats = self.agent.get_training_stats()

        # Get episode statistics from environment
        env_stats = env.get_episode_stats()

        # Create episode result
        episode_result = EpisodeResult(
            episode=self.current_episode,
            total_reward=episode_reward,
            episode_length=episode_length,
            final_portfolio_value=env_stats.get('final_portfolio_value', 0),
            total_return=env_stats.get('total_return', 0),
            sharpe_ratio=env_stats.get('sharpe_ratio', 0),
            max_drawdown=env_stats.get('max_drawdown', 0),
            num_trades=env_stats.get('num_trades', 0),
            transaction_costs=env_stats.get('transaction_costs', 0),
            win_rate=env_stats.get('win_rate', 0),
            agent_stats=agent_stats
        )

        return episode_result

    def evaluate(self) -> List[EpisodeResult]:
        """Run evaluation episodes"""
        logger.info(f"Running evaluation at episode {self.current_episode}")

        eval_results = []
        for eval_episode in range(self.config.eval_episodes):
            result = self.run_episode(training=False)
            eval_results.append(result)

        self.eval_results.extend(eval_results)
        return eval_results

    def update_env_for_curriculum(self):
        """Update environment parameters based on curriculum stage"""
        if not self.config.curriculum_enabled:
            return

        stage_config = self.curriculum.get_current_stage()

        # Update environment parameters
        self.env.max_position_size = stage_config['max_position_size']
        self.env.transaction_cost_bps = stage_config['transaction_cost_bps']
        self.env.risk_penalty_factor = stage_config['risk_penalty_factor']

        # Update action space if needed
        if hasattr(self.env, 'action_type') and self.env.action_type != stage_config['action_type']:
            logger.info(f"Updating action type to {stage_config['action_type']}")
            # This would require reinitializing the environment and agent
            # For now, we'll just log the change

    def log_training_progress(self, result: EpisodeResult):
        """Log training progress"""
        # Update training logs
        self.training_logs['episode_rewards'].append(result.total_reward)
        self.training_logs['episode_lengths'].append(result.episode_length)
        self.training_logs['portfolio_values'].append(result.final_portfolio_value)
        self.training_logs['total_returns'].append(result.total_return)
        self.training_logs['sharpe_ratios'].append(result.sharpe_ratio)

        if result.agent_stats:
            loss = result.agent_stats.get('total_loss', result.agent_stats.get('policy_loss', 0))
            self.training_logs['agent_losses'].append(loss)

        # Tensorboard logging
        self.writer.add_scalar('Training/Episode_Reward', result.total_reward, result.episode)
        self.writer.add_scalar('Training/Portfolio_Value', result.final_portfolio_value, result.episode)
        self.writer.add_scalar('Training/Total_Return', result.total_return, result.episode)
        self.writer.add_scalar('Training/Sharpe_Ratio', result.sharpe_ratio, result.episode)
        self.writer.add_scalar('Training/Num_Trades', result.num_trades, result.episode)

        if result.agent_stats:
            for key, value in result.agent_stats.items():
                self.writer.add_scalar(f'Agent/{key}', value, result.episode)

        # Console logging
        if (result.episode + 1) % self.config.log_frequency == 0:
            recent_rewards = self.training_logs['episode_rewards'][-self.config.log_frequency:]
            recent_returns = self.training_logs['total_returns'][-self.config.log_frequency:]

            logger.info(
                f"Episode {result.episode:5d} | "
                f"Reward: {np.mean(recent_rewards):8.3f} ± {np.std(recent_rewards):6.3f} | "
                f"Return: {np.mean(recent_returns):8.2%} ± {np.std(recent_returns):6.2%} | "
                f"Portfolio: ${result.final_portfolio_value:8.0f} | "
                f"Stage: {self.curriculum.get_current_stage()['name']}"
            )

    def log_evaluation_results(self, eval_results: List[EpisodeResult]):
        """Log evaluation results"""
        if not eval_results:
            return

        # Calculate evaluation metrics
        eval_rewards = [r.total_reward for r in eval_results]
        eval_returns = [r.total_return for r in eval_results]
        eval_sharpes = [r.sharpe_ratio for r in eval_results]

        mean_reward = np.mean(eval_rewards)
        mean_return = np.mean(eval_returns)
        mean_sharpe = np.mean(eval_sharpes)

        # Tensorboard logging
        self.writer.add_scalar('Evaluation/Mean_Reward', mean_reward, self.current_episode)
        self.writer.add_scalar('Evaluation/Mean_Return', mean_return, self.current_episode)
        self.writer.add_scalar('Evaluation/Mean_Sharpe', mean_sharpe, self.current_episode)

        logger.info(
            f"Evaluation | Episodes: {len(eval_results)} | "
            f"Mean Reward: {mean_reward:8.3f} | "
            f"Mean Return: {mean_return:8.2%} | "
            f"Mean Sharpe: {mean_sharpe:8.3f}"
        )

    def update_best_model(self, eval_results: List[EpisodeResult]):
        """Update best model based on evaluation results"""
        if not eval_results:
            return

        # Use mean Sharpe ratio as the evaluation metric
        eval_score = np.mean([r.sharpe_ratio for r in eval_results])

        if eval_score > self.best_eval_score:
            self.best_eval_score = eval_score
            self.no_improvement_count = 0

            # Save best model
            best_model_path = self.model_dir / "best_model.pt"
            self.agent.save(str(best_model_path))

            logger.info(f"New best model saved! Eval Sharpe: {eval_score:.3f}")
        else:
            self.no_improvement_count += 1

    def should_early_stop(self) -> bool:
        """Check if training should stop early"""
        if not self.config.early_stopping_enabled:
            return False

        return self.no_improvement_count >= self.config.early_stopping_patience

    def save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = self.model_dir / f"checkpoint_episode_{episode}.pt"
        self.agent.save(str(checkpoint_path))

        # Save training state
        state_path = self.model_dir / f"training_state_{episode}.json"
        training_state = {
            'episode': episode,
            'best_eval_score': self.best_eval_score,
            'no_improvement_count': self.no_improvement_count,
            'curriculum_stage': self.curriculum.current_stage,
            'training_logs': {k: v[-100:] for k, v in self.training_logs.items()}  # Save last 100
        }

        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)

        logger.info(f"Checkpoint saved at episode {episode}")

    def cleanup_training(self):
        """Clean up after training"""
        # Save final model
        final_model_path = self.model_dir / "final_model.pt"
        self.agent.save(str(final_model_path))

        # Save training logs
        logs_path = self.log_dir / "training_logs.json"
        with open(logs_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2)

        # Close tensorboard writer
        self.writer.close()

        logger.info("Training cleanup completed")

    def generate_training_summary(self) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        training_time = time.time() - self.training_start_time if self.training_start_time else 0

        # Calculate final metrics
        if self.training_results:
            final_rewards = [r.total_reward for r in self.training_results[-100:]]
            final_returns = [r.total_return for r in self.training_results[-100:]]
            final_sharpes = [r.sharpe_ratio for r in self.training_results[-100:]]

            training_summary = {
                'training_completed': True,
                'total_episodes': len(self.training_results),
                'training_time_seconds': training_time,
                'best_eval_score': self.best_eval_score,
                'final_performance': {
                    'mean_reward': np.mean(final_rewards),
                    'mean_return': np.mean(final_returns),
                    'mean_sharpe': np.mean(final_sharpes),
                    'std_reward': np.std(final_rewards),
                    'std_return': np.std(final_returns),
                    'std_sharpe': np.std(final_sharpes)
                },
                'curriculum_stages_completed': self.curriculum.current_stage + 1,
                'agent_type': type(self.agent).__name__,
                'config': asdict(self.config)
            }
        else:
            training_summary = {
                'training_completed': False,
                'total_episodes': 0,
                'training_time_seconds': training_time,
                'error': 'No training results collected'
            }

        # Save summary
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)

        return training_summary

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves and analysis"""
        if not self.training_results:
            logger.warning("No training results to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RL Training Analysis', fontsize=16)

        # Episode rewards
        episodes = [r.episode for r in self.training_results]
        rewards = [r.total_reward for r in self.training_results]
        axes[0, 0].plot(episodes, rewards, alpha=0.3)
        axes[0, 0].plot(episodes, pd.Series(rewards).rolling(50).mean(), color='red')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')

        # Portfolio values
        portfolio_values = [r.final_portfolio_value for r in self.training_results]
        axes[0, 1].plot(episodes, portfolio_values, alpha=0.3)
        axes[0, 1].plot(episodes, pd.Series(portfolio_values).rolling(50).mean(), color='red')
        axes[0, 1].set_title('Portfolio Values')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Portfolio Value ($)')

        # Total returns
        total_returns = [r.total_return for r in self.training_results]
        axes[0, 2].plot(episodes, total_returns, alpha=0.3)
        axes[0, 2].plot(episodes, pd.Series(total_returns).rolling(50).mean(), color='red')
        axes[0, 2].set_title('Total Returns')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Return')

        # Sharpe ratios
        sharpe_ratios = [r.sharpe_ratio for r in self.training_results]
        axes[1, 0].plot(episodes, sharpe_ratios, alpha=0.3)
        axes[1, 0].plot(episodes, pd.Series(sharpe_ratios).rolling(50).mean(), color='red')
        axes[1, 0].set_title('Sharpe Ratios')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Sharpe Ratio')

        # Number of trades
        num_trades = [r.num_trades for r in self.training_results]
        axes[1, 1].plot(episodes, num_trades, alpha=0.3)
        axes[1, 1].plot(episodes, pd.Series(num_trades).rolling(50).mean(), color='red')
        axes[1, 1].set_title('Number of Trades per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Trades')

        # Win rates
        win_rates = [r.win_rate for r in self.training_results]
        axes[1, 2].plot(episodes, win_rates, alpha=0.3)
        axes[1, 2].plot(episodes, pd.Series(win_rates).rolling(50).mean(), color='red')
        axes[1, 2].set_title('Win Rates')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Win Rate')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        else:
            plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')

        plt.show()

    def hyperparameter_optimization(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization using Optuna.

        Args:
            objective_function: Function that takes hyperparameters and returns score
            param_space: Dictionary defining parameter search space

        Returns:
            Best hyperparameters found
        """
        if not self.config.hyperopt_enabled:
            logger.info("Hyperparameter optimization disabled")
            return {}

        logger.info("Starting hyperparameter optimization...")

        def optuna_objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )

            # Evaluate hyperparameters
            try:
                score = objective_function(params)
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('-inf')

        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(optuna_objective, n_trials=self.config.hyperopt_trials)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Hyperparameter optimization completed!")
        logger.info(f"Best score: {best_score:.3f}")
        logger.info(f"Best parameters: {best_params}")

        # Save results
        hyperopt_results = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials),
            'study_trials': [
                {
                    'params': trial.params,
                    'value': trial.value,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }

        hyperopt_path = self.output_dir / "hyperopt_results.json"
        with open(hyperopt_path, 'w') as f:
            json.dump(hyperopt_results, f, indent=2)

        return best_params