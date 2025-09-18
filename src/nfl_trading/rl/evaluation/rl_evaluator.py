"""
RL Evaluation Framework

Comprehensive evaluation system for comparing RL agents against baseline strategies
with proper statistical testing and performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import warnings
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import components
try:
    from ..environment.nfl_trading_gym import NFLTradingGym
    from ..agents.trading_agent import TradingAgent
    from ..advanced_features import AgentEnsemble, MultiAgentTrainer
    from ...backtesting.strategies import (
        RuleBasedTrader, StatisticalTrader, RandomTrader,
        BuyAndHoldTrader, MomentumFollower
    )
    from ...backtesting.backtester import Backtester
except ImportError:
    logging.warning("Some imports failed - evaluation may be limited")

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Risk-adjusted metrics
    var_95: float
    cvar_95: float
    beta: float
    alpha: float

    # Trading metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Consistency metrics
    hit_ratio: float
    consistency_score: float
    stability_index: float


@dataclass
class StatisticalTest:
    """Statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    interpretation: str


class PerformanceAnalyzer:
    """Advanced performance analysis with statistical testing"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def calculate_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        trades: Optional[List[Dict]] = None
    ) -> EvaluationMetrics:
        """Calculate comprehensive performance metrics"""

        # Basic return metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = np.power(1 + total_return, 252 / len(returns)) - 1 if len(returns) > 0 else 0

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assume 2% risk-free rate

        # Downside deviation for Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0.01
        sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0

        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95

        # Market-relative metrics
        excess_return = 0
        beta = 0
        alpha = 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            excess_return = annualized_return - (np.power(1 + np.prod(1 + benchmark_returns) - 1, 252 / len(benchmark_returns)) - 1)
            if np.std(benchmark_returns) > 0:
                beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                alpha = annualized_return - (0.02 + beta * (np.mean(benchmark_returns) * 252 - 0.02))

        # Trading metrics
        total_trades = len(trades) if trades else 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

        if trades:
            profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

            win_rate = len(profitable_trades) / len(trades) if len(trades) > 0 else 0
            avg_win = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0

            total_profit = sum(t['pnl'] for t in profitable_trades)
            total_loss = sum(abs(t['pnl']) for t in losing_trades)
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Consistency metrics
        monthly_returns = self._calculate_monthly_returns(returns)
        hit_ratio = len([r for r in monthly_returns if r > 0]) / len(monthly_returns) if monthly_returns else 0
        consistency_score = 1 - np.std(monthly_returns) / np.mean(monthly_returns) if len(monthly_returns) > 0 and np.mean(monthly_returns) != 0 else 0
        stability_index = 1 - abs(max_drawdown) / total_return if total_return > 0 else 0

        return EvaluationMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=excess_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            hit_ratio=hit_ratio,
            consistency_score=consistency_score,
            stability_index=stability_index
        )

    def _calculate_monthly_returns(self, daily_returns: np.ndarray) -> List[float]:
        """Calculate monthly returns from daily returns"""
        if len(daily_returns) < 20:  # Need at least 20 days
            return []

        # Approximate monthly returns (every 21 trading days)
        monthly_returns = []
        for i in range(0, len(daily_returns), 21):
            month_returns = daily_returns[i:i+21]
            if len(month_returns) >= 10:  # At least 10 days
                monthly_return = np.prod(1 + month_returns) - 1
                monthly_returns.append(monthly_return)

        return monthly_returns

    def compare_strategies(
        self,
        strategy_returns: Dict[str, np.ndarray],
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compare multiple strategies with statistical tests"""

        results = {}

        # Calculate metrics for each strategy
        for name, returns in strategy_returns.items():
            metrics = self.calculate_metrics(returns, benchmark_returns)
            results[name] = {
                'metrics': metrics,
                'returns': returns
            }

        # Pairwise statistical comparisons
        strategy_names = list(strategy_returns.keys())
        comparison_matrix = {}

        for i, strategy1 in enumerate(strategy_names):
            for j, strategy2 in enumerate(strategy_names[i+1:], i+1):
                returns1 = strategy_returns[strategy1]
                returns2 = strategy_returns[strategy2]

                # Perform statistical tests
                tests = self._perform_statistical_tests(returns1, returns2, strategy1, strategy2)
                comparison_matrix[f"{strategy1}_vs_{strategy2}"] = tests

        # Rank strategies
        rankings = self._rank_strategies(results)

        return {
            'individual_results': results,
            'statistical_comparisons': comparison_matrix,
            'rankings': rankings,
            'summary': self._generate_comparison_summary(results, comparison_matrix)
        }

    def _perform_statistical_tests(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        name1: str,
        name2: str
    ) -> List[StatisticalTest]:
        """Perform comprehensive statistical tests"""

        tests = []

        # T-test for mean returns
        try:
            t_stat, t_p = ttest_ind(returns1, returns2)
            tests.append(StatisticalTest(
                test_name="Welch's t-test",
                statistic=t_stat,
                p_value=t_p,
                significant=t_p < self.alpha,
                confidence_level=self.confidence_level,
                interpretation=f"Mean return difference between {name1} and {name2} is {'significant' if t_p < self.alpha else 'not significant'}"
            ))
        except Exception as e:
            logger.warning(f"T-test failed: {e}")

        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p = mannwhitneyu(returns1, returns2, alternative='two-sided')
            tests.append(StatisticalTest(
                test_name="Mann-Whitney U test",
                statistic=u_stat,
                p_value=u_p,
                significant=u_p < self.alpha,
                confidence_level=self.confidence_level,
                interpretation=f"Distribution difference between {name1} and {name2} is {'significant' if u_p < self.alpha else 'not significant'}"
            ))
        except Exception as e:
            logger.warning(f"Mann-Whitney U test failed: {e}")

        # Kolmogorov-Smirnov test for distribution similarity
        try:
            ks_stat, ks_p = stats.ks_2samp(returns1, returns2)
            tests.append(StatisticalTest(
                test_name="Kolmogorov-Smirnov test",
                statistic=ks_stat,
                p_value=ks_p,
                significant=ks_p < self.alpha,
                confidence_level=self.confidence_level,
                interpretation=f"Return distributions of {name1} and {name2} are {'significantly different' if ks_p < self.alpha else 'similar'}"
            ))
        except Exception as e:
            logger.warning(f"KS test failed: {e}")

        # Variance test (F-test)
        try:
            f_stat = np.var(returns1, ddof=1) / np.var(returns2, ddof=1)
            df1, df2 = len(returns1) - 1, len(returns2) - 1
            f_p = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

            tests.append(StatisticalTest(
                test_name="F-test for equal variances",
                statistic=f_stat,
                p_value=f_p,
                significant=f_p < self.alpha,
                confidence_level=self.confidence_level,
                interpretation=f"Volatilities of {name1} and {name2} are {'significantly different' if f_p < self.alpha else 'similar'}"
            ))
        except Exception as e:
            logger.warning(f"F-test failed: {e}")

        return tests

    def _rank_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank strategies by multiple criteria"""

        rankings = {}

        # Define ranking criteria
        criteria = [
            ('total_return', 'desc'),
            ('sharpe_ratio', 'desc'),
            ('calmar_ratio', 'desc'),
            ('max_drawdown', 'asc'),  # Lower is better
            ('volatility', 'asc'),    # Lower is better
            ('consistency_score', 'desc')
        ]

        for criterion, direction in criteria:
            # Extract values for ranking
            values = []
            for name, result in results.items():
                value = getattr(result['metrics'], criterion)
                values.append((name, value))

            # Sort based on direction
            if direction == 'desc':
                values.sort(key=lambda x: x[1], reverse=True)
            else:
                values.sort(key=lambda x: x[1])

            # Create ranking
            rankings[criterion] = [name for name, _ in values]

        # Calculate composite score (weighted average of ranks)
        weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.30,
            'calmar_ratio': 0.20,
            'max_drawdown': 0.15,
            'volatility': 0.05,
            'consistency_score': 0.05
        }

        composite_scores = {}
        for name in results.keys():
            score = 0
            for criterion, weight in weights.items():
                rank = rankings[criterion].index(name) + 1  # 1-based ranking
                normalized_rank = 1 - (rank - 1) / (len(results) - 1) if len(results) > 1 else 1
                score += weight * normalized_rank
            composite_scores[name] = score

        # Final composite ranking
        composite_ranking = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['composite'] = [name for name, _ in composite_ranking]
        rankings['composite_scores'] = composite_scores

        return rankings

    def _generate_comparison_summary(
        self,
        results: Dict[str, Any],
        comparisons: Dict[str, List[StatisticalTest]]
    ) -> Dict[str, Any]:
        """Generate a summary of strategy comparisons"""

        summary = {
            'total_strategies': len(results),
            'best_strategy': None,
            'significant_differences': 0,
            'key_findings': []
        }

        # Find best performing strategy
        best_sharpe = -np.inf
        for name, result in results.items():
            sharpe = result['metrics'].sharpe_ratio
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                summary['best_strategy'] = name

        # Count significant differences
        for comparison_name, tests in comparisons.items():
            for test in tests:
                if test.significant:
                    summary['significant_differences'] += 1

        # Generate key findings
        if summary['best_strategy']:
            best_metrics = results[summary['best_strategy']]['metrics']
            summary['key_findings'].append(
                f"Best strategy: {summary['best_strategy']} "
                f"(Sharpe: {best_metrics.sharpe_ratio:.3f}, "
                f"Return: {best_metrics.total_return:.2%})"
            )

        # Find strategies with significant outperformance
        for comparison_name, tests in comparisons.items():
            strategy1, strategy2 = comparison_name.split('_vs_')
            for test in tests:
                if test.significant and 'mean return' in test.interpretation.lower():
                    summary['key_findings'].append(
                        f"Significant performance difference found between {strategy1} and {strategy2}"
                    )
                    break

        return summary


class RLEvaluator:
    """
    Comprehensive RL evaluation framework for comparing agents against baselines.
    """

    def __init__(
        self,
        data_files: List[str],
        baseline_strategies: Optional[List] = None,
        num_evaluation_runs: int = 50,
        confidence_level: float = 0.95
    ):
        self.data_files = data_files
        self.baseline_strategies = baseline_strategies or self._get_default_baselines()
        self.num_evaluation_runs = num_evaluation_runs
        self.confidence_level = confidence_level

        self.performance_analyzer = PerformanceAnalyzer(confidence_level)
        self.evaluation_results: Dict[str, Any] = {}

    def _get_default_baselines(self) -> List:
        """Get default baseline strategies for comparison"""
        return [
            RuleBasedTrader(touchdown_buy_size=30, hold_duration_minutes=3),
            StatisticalTrader(base_position_size=40, lookback_window=15),
            RandomTrader(trade_probability=0.03, random_seed=42),
            BuyAndHoldTrader(initial_position_size=50),
            MomentumFollower(position_size=35, momentum_threshold=0.015)
        ]

    def evaluate_rl_agent(
        self,
        agent: Union[TradingAgent, AgentEnsemble],
        agent_name: str = "RL_Agent",
        use_risk_management: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate RL agent performance across multiple runs.

        Returns comprehensive evaluation results with statistical analysis.
        """
        logger.info(f"Evaluating RL agent: {agent_name}")

        # Run multiple evaluation episodes
        agent_results = []

        for run in range(self.num_evaluation_runs):
            # Create fresh environment for each run
            env = NFLTradingGym(
                data_files=self.data_files,
                initial_capital=10000.0,
                max_position_size=500,
                action_type="continuous"
            )

            try:
                # Run single evaluation episode
                episode_result = self._run_agent_episode(env, agent, use_risk_management)
                agent_results.append(episode_result)

                if (run + 1) % 10 == 0:
                    logger.info(f"Completed {run + 1}/{self.num_evaluation_runs} evaluation runs")

            except Exception as e:
                logger.warning(f"Evaluation run {run} failed: {e}")

        if not agent_results:
            raise ValueError("No successful evaluation runs completed")

        # Extract performance metrics
        returns = np.array([r['total_return'] for r in agent_results])
        portfolio_values = np.array([r['final_portfolio_value'] for r in agent_results])

        # Calculate comprehensive metrics
        metrics = self.performance_analyzer.calculate_metrics(returns)

        return {
            'agent_name': agent_name,
            'num_runs': len(agent_results),
            'metrics': metrics,
            'raw_results': agent_results,
            'returns_distribution': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'median': np.median(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentiles': {
                    'p5': np.percentile(returns, 5),
                    'p25': np.percentile(returns, 25),
                    'p75': np.percentile(returns, 75),
                    'p95': np.percentile(returns, 95)
                }
            }
        }

    def _run_agent_episode(
        self,
        env: NFLTradingGym,
        agent: Union[TradingAgent, AgentEnsemble],
        use_risk_management: bool
    ) -> Dict[str, Any]:
        """Run single agent evaluation episode"""

        state, info = env.reset()
        episode_return = 0
        trades = []

        # Initialize risk management if enabled
        risk_manager = None
        if use_risk_management:
            from ..risk_management.risk_manager import RiskManager
            risk_manager = RiskManager()

        for step in range(1000):  # Max steps per episode
            # Get action from agent
            if isinstance(agent, AgentEnsemble):
                action, action_info = agent.select_action(state, training=False)
            else:
                action, action_info = agent.select_action(state, training=False)

            # Apply risk management if enabled
            if risk_manager:
                # This would require integration with the gym environment
                # For now, we'll use the action as-is
                pass

            # Execute action
            next_state, reward, done, truncated, step_info = env.step(action)

            episode_return += reward

            # Record trade if position changed
            if 'position_size' in step_info and step_info['position_size'] != 0:
                trades.append({
                    'step': step,
                    'action': action,
                    'reward': reward,
                    'position_size': step_info['position_size']
                })

            state = next_state

            if done or truncated:
                break

        # Get final episode statistics
        episode_stats = env.get_episode_stats()

        return {
            'total_return': episode_stats.get('total_return', 0.0),
            'final_portfolio_value': episode_stats.get('final_portfolio_value', 10000.0),
            'sharpe_ratio': episode_stats.get('sharpe_ratio', 0.0),
            'max_drawdown': episode_stats.get('max_drawdown', 0.0),
            'num_trades': len(trades),
            'episode_length': step + 1,
            'trades': trades
        }

    def evaluate_baseline_strategies(self) -> Dict[str, Any]:
        """Evaluate baseline strategies using backtesting framework"""
        logger.info("Evaluating baseline strategies...")

        baseline_results = {}

        for strategy in self.baseline_strategies:
            logger.info(f"Evaluating {strategy.name}")

            try:
                # Use backtesting framework for baseline evaluation
                backtester = Backtester(initial_capital=10000.0, fee_rate=0.07)

                strategy_results = []

                # Run multiple backtests with different data files
                for data_file in self.data_files[:min(len(self.data_files), self.num_evaluation_runs)]:
                    try:
                        result = backtester.run_backtest(
                            strategy=strategy,
                            data_files=[data_file],
                            timestep_seconds=60,
                            warmup_minutes=2
                        )

                        strategy_results.append({
                            'total_return': result.total_return,
                            'sharpe_ratio': result.sharpe_ratio,
                            'max_drawdown': result.max_drawdown,
                            'num_trades': result.total_trades,
                            'win_rate': result.win_rate
                        })

                    except Exception as e:
                        logger.warning(f"Baseline evaluation failed for {data_file}: {e}")

                if strategy_results:
                    returns = np.array([r['total_return'] for r in strategy_results])
                    metrics = self.performance_analyzer.calculate_metrics(returns)

                    baseline_results[strategy.name] = {
                        'metrics': metrics,
                        'raw_results': strategy_results,
                        'returns_distribution': {
                            'mean': np.mean(returns),
                            'std': np.std(returns),
                            'median': np.median(returns)
                        }
                    }

            except Exception as e:
                logger.error(f"Failed to evaluate {strategy.name}: {e}")

        return baseline_results

    def comprehensive_comparison(
        self,
        rl_agents: Dict[str, Union[TradingAgent, AgentEnsemble]],
        save_results: bool = True,
        output_dir: str = "evaluation_results"
    ) -> Dict[str, Any]:
        """
        Run comprehensive comparison between RL agents and baselines.

        Returns complete evaluation report with statistical analysis.
        """
        logger.info("Starting comprehensive evaluation...")

        # Evaluate RL agents
        rl_results = {}
        for name, agent in rl_agents.items():
            try:
                result = self.evaluate_rl_agent(agent, name)
                rl_results[name] = result
            except Exception as e:
                logger.error(f"Failed to evaluate RL agent {name}: {e}")

        # Evaluate baseline strategies
        baseline_results = self.evaluate_baseline_strategies()

        # Combine all results
        all_results = {**rl_results, **baseline_results}

        # Extract returns for statistical comparison
        strategy_returns = {}
        for name, result in all_results.items():
            if 'raw_results' in result:
                returns = np.array([r['total_return'] for r in result['raw_results']])
                strategy_returns[name] = returns

        # Perform statistical comparison
        comparison_results = self.performance_analyzer.compare_strategies(strategy_returns)

        # Generate comprehensive report
        evaluation_report = {
            'evaluation_summary': {
                'total_strategies_evaluated': len(all_results),
                'rl_agents': len(rl_results),
                'baseline_strategies': len(baseline_results),
                'evaluation_runs_per_strategy': self.num_evaluation_runs,
                'confidence_level': self.confidence_level
            },
            'individual_results': all_results,
            'statistical_comparison': comparison_results,
            'performance_rankings': comparison_results['rankings'],
            'key_findings': self._generate_key_findings(all_results, comparison_results)
        }

        # Save results if requested
        if save_results:
            self._save_evaluation_results(evaluation_report, output_dir)

        # Generate visualizations
        self._create_evaluation_plots(evaluation_report, output_dir)

        logger.info("Comprehensive evaluation completed!")
        return evaluation_report

    def _generate_key_findings(
        self,
        all_results: Dict[str, Any],
        comparison_results: Dict[str, Any]
    ) -> List[str]:
        """Generate key findings from evaluation"""

        findings = []

        # Best overall performer
        if 'composite' in comparison_results['rankings']:
            best_strategy = comparison_results['rankings']['composite'][0]
            findings.append(f"Best overall strategy: {best_strategy}")

        # RL vs baseline comparison
        rl_names = [name for name in all_results.keys() if 'RL' in name or 'Agent' in name or 'Ensemble' in name]
        baseline_names = [name for name in all_results.keys() if name not in rl_names]

        if rl_names and baseline_names:
            # Find best RL and best baseline
            best_rl_sharpe = -np.inf
            best_rl_name = None
            for name in rl_names:
                sharpe = all_results[name]['metrics'].sharpe_ratio
                if sharpe > best_rl_sharpe:
                    best_rl_sharpe = sharpe
                    best_rl_name = name

            best_baseline_sharpe = -np.inf
            best_baseline_name = None
            for name in baseline_names:
                sharpe = all_results[name]['metrics'].sharpe_ratio
                if sharpe > best_baseline_sharpe:
                    best_baseline_sharpe = sharpe
                    best_baseline_name = name

            if best_rl_name and best_baseline_name:
                if best_rl_sharpe > best_baseline_sharpe:
                    improvement = ((best_rl_sharpe / best_baseline_sharpe) - 1) * 100
                    findings.append(f"Best RL agent ({best_rl_name}) outperforms best baseline ({best_baseline_name}) by {improvement:.1f}% in Sharpe ratio")
                else:
                    findings.append(f"Best baseline strategy ({best_baseline_name}) outperforms best RL agent ({best_rl_name})")

        # Consistency analysis
        most_consistent = None
        highest_consistency = -np.inf
        for name, result in all_results.items():
            consistency = result['metrics'].consistency_score
            if consistency > highest_consistency:
                highest_consistency = consistency
                most_consistent = name

        if most_consistent:
            findings.append(f"Most consistent strategy: {most_consistent} (consistency score: {highest_consistency:.3f})")

        # Risk analysis
        lowest_risk = None
        lowest_drawdown = np.inf
        for name, result in all_results.items():
            drawdown = abs(result['metrics'].max_drawdown)
            if drawdown < lowest_drawdown:
                lowest_drawdown = drawdown
                lowest_risk = name

        if lowest_risk:
            findings.append(f"Lowest risk strategy: {lowest_risk} (max drawdown: {lowest_drawdown:.2%})")

        return findings

    def _save_evaluation_results(self, report: Dict[str, Any], output_dir: str):
        """Save evaluation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main report
        with open(output_path / "evaluation_report.json", "w") as f:
            # Convert numpy arrays and custom objects to serializable format
            serializable_report = self._make_serializable(report)
            json.dump(serializable_report, f, indent=2)

        # Save performance rankings
        rankings_df = pd.DataFrame(report['performance_rankings'])
        rankings_df.to_csv(output_path / "performance_rankings.csv", index=False)

        logger.info(f"Evaluation results saved to {output_dir}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return asdict(obj) if hasattr(obj, '__dataclass_fields__') else str(obj)
        else:
            return obj

    def _create_evaluation_plots(self, report: Dict[str, Any], output_dir: str):
        """Create evaluation visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Performance comparison plot
        plt.figure(figsize=(15, 10))

        # Extract key metrics for plotting
        strategies = list(report['individual_results'].keys())
        sharpe_ratios = [report['individual_results'][s]['metrics'].sharpe_ratio for s in strategies]
        total_returns = [report['individual_results'][s]['metrics'].total_return for s in strategies]
        max_drawdowns = [abs(report['individual_results'][s]['metrics'].max_drawdown) for s in strategies]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Strategy Performance Comparison', fontsize=16)

        # Sharpe ratio comparison
        axes[0, 0].bar(strategies, sharpe_ratios)
        axes[0, 0].set_title('Sharpe Ratios')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Total return comparison
        axes[0, 1].bar(strategies, [r * 100 for r in total_returns])
        axes[0, 1].set_title('Total Returns')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Risk-return scatter
        axes[1, 0].scatter([r * 100 for r in total_returns], sharpe_ratios)
        for i, strategy in enumerate(strategies):
            axes[1, 0].annotate(strategy, (total_returns[i] * 100, sharpe_ratios[i]))
        axes[1, 0].set_xlabel('Total Return (%)')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].set_title('Risk-Return Profile')

        # Max drawdown comparison
        axes[1, 1].bar(strategies, [d * 100 for d in max_drawdowns])
        axes[1, 1].set_title('Maximum Drawdowns')
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Evaluation plots saved to {output_dir}")


# Convenience function for quick evaluation
def evaluate_rl_vs_baselines(
    rl_agents: Dict[str, Union[TradingAgent, AgentEnsemble]],
    data_files: List[str],
    num_runs: int = 50,
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive RL vs baseline evaluation.

    Args:
        rl_agents: Dictionary of RL agents to evaluate
        data_files: List of data files for evaluation
        num_runs: Number of evaluation runs per strategy
        output_dir: Directory to save results

    Returns:
        Comprehensive evaluation report
    """
    evaluator = RLEvaluator(
        data_files=data_files,
        num_evaluation_runs=num_runs,
        confidence_level=0.95
    )

    return evaluator.comprehensive_comparison(
        rl_agents=rl_agents,
        save_results=True,
        output_dir=output_dir
    )