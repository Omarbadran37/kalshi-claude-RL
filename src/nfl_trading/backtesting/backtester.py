"""
Backtesting Engine

Comprehensive backtesting framework with walk-forward validation,
performance metrics, and statistical significance testing.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from scipy import stats
import warnings

try:
    from .trading_environment import TradingEnvironment
    from .strategies import BaseStrategy
except ImportError:
    from trading_environment import TradingEnvironment
    from strategies import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Performance metrics
        self.total_return: float = 0.0
        self.annualized_return: float = 0.0
        self.volatility: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_drawdown_duration: float = 0.0
        self.calmar_ratio: float = 0.0
        
        # Trading metrics
        self.total_trades: int = 0
        self.win_rate: float = 0.0
        self.profit_factor: float = 0.0
        self.avg_trade_pnl: float = 0.0
        self.avg_winning_trade: float = 0.0
        self.avg_losing_trade: float = 0.0
        self.largest_win: float = 0.0
        self.largest_loss: float = 0.0
        
        # Risk metrics
        self.var_95: float = 0.0  # Value at Risk
        self.cvar_95: float = 0.0  # Conditional VaR
        self.skewness: float = 0.0
        self.kurtosis: float = 0.0
        
        # Raw data
        self.portfolio_timeseries: pd.DataFrame = pd.DataFrame()
        self.trade_history: pd.DataFrame = pd.DataFrame()
        self.positions_history: List[Dict] = []
        
        # Game-level results
        self.game_results: List[Dict] = []


class Backtester:
    """
    Comprehensive backtesting engine for NFL trading strategies.
    
    Features:
    - Walk-forward validation across multiple games
    - Comprehensive performance and risk metrics
    - Statistical significance testing
    - Trade-level attribution analysis
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.07,
        slippage_bps: float = 0.5,
        risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe calculation
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.risk_free_rate = risk_free_rate
        
        self.results: Dict[str, BacktestResult] = {}
        
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data_files: List[str],
        timestep_seconds: int = 60,
        warmup_minutes: int = 5
    ) -> BacktestResult:
        """
        Run backtest for a single strategy across multiple game files.
        
        Args:
            strategy: Trading strategy to test
            data_files: List of JSON data files to test on
            timestep_seconds: Simulation timestep in seconds
            warmup_minutes: Minutes to wait before starting trading
        """
        logger.info(f"Starting backtest for {strategy.name} on {len(data_files)} games")
        
        result = BacktestResult(strategy.name)
        result.start_time = datetime.now()
        
        all_portfolio_data = []
        all_trade_data = []
        game_results = []
        
        for i, data_file in enumerate(data_files):
            logger.info(f"Processing game {i+1}/{len(data_files)}: {Path(data_file).name}")
            
            game_result = self._run_single_game(
                strategy, data_file, timestep_seconds, warmup_minutes
            )
            
            if game_result:
                game_results.append(game_result)
                
                # Aggregate portfolio data
                if not game_result['portfolio_timeseries'].empty:
                    portfolio_df = game_result['portfolio_timeseries'].copy()
                    portfolio_df['game_id'] = i
                    portfolio_df['game_file'] = Path(data_file).name
                    all_portfolio_data.append(portfolio_df)
                
                # Aggregate trade data
                if not game_result['trade_history'].empty:
                    trade_df = game_result['trade_history'].copy()
                    trade_df['game_id'] = i
                    trade_df['game_file'] = Path(data_file).name
                    all_trade_data.append(trade_df)
        
        # Combine all data
        if all_portfolio_data:
            result.portfolio_timeseries = pd.concat(all_portfolio_data, ignore_index=True)
        if all_trade_data:
            result.trade_history = pd.concat(all_trade_data, ignore_index=True)
        
        result.game_results = game_results
        result.end_time = datetime.now()
        
        # Calculate comprehensive metrics
        self._calculate_performance_metrics(result)
        
        # Store result
        self.results[strategy.name] = result
        
        logger.info(f"Backtest completed for {strategy.name}")
        logger.info(f"Total Return: {result.total_return:.2%}")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"Win Rate: {result.win_rate:.2%}")
        
        return result
    
    def _run_single_game(
        self,
        strategy: BaseStrategy,
        data_file: str,
        timestep_seconds: int,
        warmup_minutes: int
    ) -> Optional[Dict[str, Any]]:
        """Run backtest for a single game"""
        
        try:
            # Create fresh trading environment
            env = TradingEnvironment(
                initial_capital=self.initial_capital,
                fee_rate=self.fee_rate,
                slippage_bps=self.slippage_bps
            )
            
            # Load market data
            env.load_market_data(data_file)
            
            if not env.market_data:
                logger.warning(f"No market data loaded from {data_file}")
                return None
            
            # Get time range
            all_timestamps = []
            for ticker_data in env.market_data.values():
                all_timestamps.extend([ms.timestamp for ms in ticker_data])
            
            if not all_timestamps:
                logger.warning(f"No timestamps found in {data_file}")
                return None
            
            start_timestamp = min(all_timestamps)
            end_timestamp = max(all_timestamps)
            warmup_end = start_timestamp + (warmup_minutes * 60)
            
            # Reset strategy
            strategy.reset()
            
            # Run simulation
            current_timestamp = start_timestamp
            
            while current_timestamp <= end_timestamp:
                # Advance environment
                env.advance_time(current_timestamp)
                
                # Skip warmup period for strategy signals
                if current_timestamp >= warmup_end:
                    # Get current market state
                    market_data = {}
                    for ticker in env.market_data:
                        market_state = env.get_current_market_state(ticker)
                        if market_state:
                            market_data[ticker] = market_state
                    
                    # Generate trading signals
                    if market_data:
                        signals = strategy.generate_signals(env, market_data)
                        
                        # Execute signals
                        for ticker, side, size, order_type, limit_price in signals:
                            try:
                                env.place_order(ticker, side, size, order_type, limit_price)
                            except Exception as e:
                                logger.warning(f"Failed to place order: {e}")
                
                # Advance time
                current_timestamp += timestep_seconds
            
            # Get final results
            portfolio_df = env.get_portfolio_timeseries()
            trade_df = env.get_trade_history_df()
            final_summary = env.get_portfolio_summary()
            
            return {
                'data_file': data_file,
                'portfolio_timeseries': portfolio_df,
                'trade_history': trade_df,
                'final_summary': final_summary,
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp
            }
            
        except Exception as e:
            logger.error(f"Error processing {data_file}: {e}")
            return None
    
    def _calculate_performance_metrics(self, result: BacktestResult):
        """Calculate comprehensive performance metrics"""
        
        if result.portfolio_timeseries.empty:
            logger.warning(f"No portfolio data for {result.strategy_name}")
            return
        
        # Basic return metrics
        returns = result.portfolio_timeseries['returns'].dropna()
        portfolio_values = result.portfolio_timeseries['portfolio_value']
        
        if len(returns) == 0 or len(portfolio_values) == 0:
            return
        
        result.total_return = (portfolio_values.iloc[-1] / self.initial_capital) - 1
        
        # Annualized return (assuming each game is ~3 hours)
        # This is approximate - could be improved with actual time duration
        num_games = result.portfolio_timeseries['game_id'].nunique() if 'game_id' in result.portfolio_timeseries else 1
        total_hours = num_games * 3  # Approximate game duration
        annualization_factor = (365 * 24) / total_hours  # Scale to annual
        result.annualized_return = (1 + result.total_return) ** annualization_factor - 1
        
        # Volatility
        if len(returns) > 1:
            result.volatility = returns.std() * np.sqrt(252 * 24)  # Annualized daily vol
        
        # Sharpe ratio
        if result.volatility > 0:
            excess_return = result.annualized_return - self.risk_free_rate
            result.sharpe_ratio = excess_return / result.volatility
        
        # Drawdown metrics
        cumulative_returns = (portfolio_values / self.initial_capital)
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        result.max_drawdown = abs(drawdowns.min())
        
        # Max drawdown duration (approximate)
        if len(drawdowns) > 0:
            in_drawdown = drawdowns < -0.001  # In drawdown if down >0.1%
            if in_drawdown.any():
                # Find longest consecutive drawdown period
                drawdown_lengths = []
                current_length = 0
                for is_dd in in_drawdown:
                    if is_dd:
                        current_length += 1
                    else:
                        if current_length > 0:
                            drawdown_lengths.append(current_length)
                        current_length = 0
                if current_length > 0:
                    drawdown_lengths.append(current_length)
                
                if drawdown_lengths:
                    result.max_drawdown_duration = max(drawdown_lengths)
        
        # Calmar ratio
        if result.max_drawdown > 0:
            result.calmar_ratio = result.annualized_return / result.max_drawdown
        
        # Trading metrics
        if not result.trade_history.empty:
            trades = result.trade_history
            result.total_trades = len(trades)
            
            # PnL analysis
            pnls = trades['pnl'].values
            winning_trades = pnls[pnls > 0]
            losing_trades = pnls[pnls < 0]
            
            result.win_rate = len(winning_trades) / len(pnls) if len(pnls) > 0 else 0
            result.avg_trade_pnl = np.mean(pnls) if len(pnls) > 0 else 0
            
            if len(winning_trades) > 0:
                result.avg_winning_trade = np.mean(winning_trades)
                result.largest_win = np.max(winning_trades)
            
            if len(losing_trades) > 0:
                result.avg_losing_trade = np.mean(losing_trades)
                result.largest_loss = np.min(losing_trades)
            
            # Profit factor
            total_wins = np.sum(winning_trades) if len(winning_trades) > 0 else 0
            total_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Risk metrics
        if len(returns) > 20:  # Need sufficient data
            # VaR and CVaR
            result.var_95 = np.percentile(returns, 5)
            cvar_mask = returns <= result.var_95
            result.cvar_95 = returns[cvar_mask].mean() if cvar_mask.any() else result.var_95
            
            # Higher moments
            result.skewness = stats.skew(returns)
            result.kurtosis = stats.kurtosis(returns)
    
    def compare_strategies(self, metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """Compare strategies by a specific metric"""
        
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for strategy_name, result in self.results.items():
            comparison_data.append({
                'strategy': strategy_name,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades,
                'var_95': result.var_95,
                'skewness': result.skewness
            })
        
        df = pd.DataFrame(comparison_data)
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df
    
    def statistical_significance_test(
        self, 
        strategy1: str, 
        strategy2: str,
        test_type: str = 'ttest'
    ) -> Dict[str, Any]:
        """
        Test statistical significance between two strategies.
        
        Args:
            strategy1: Name of first strategy
            strategy2: Name of second strategy
            test_type: 'ttest' or 'wilcoxon'
        """
        
        if strategy1 not in self.results or strategy2 not in self.results:
            raise ValueError("Both strategies must be in results")
        
        result1 = self.results[strategy1]
        result2 = self.results[strategy2]
        
        if result1.portfolio_timeseries.empty or result2.portfolio_timeseries.empty:
            return {'error': 'Insufficient data for comparison'}
        
        returns1 = result1.portfolio_timeseries['returns'].dropna()
        returns2 = result2.portfolio_timeseries['returns'].dropna()
        
        if len(returns1) == 0 or len(returns2) == 0:
            return {'error': 'No valid returns data'}
        
        # Align returns by game if possible
        if 'game_id' in result1.portfolio_timeseries.columns and 'game_id' in result2.portfolio_timeseries.columns:
            # Calculate per-game returns
            game_returns1 = result1.portfolio_timeseries.groupby('game_id')['returns'].sum()
            game_returns2 = result2.portfolio_timeseries.groupby('game_id')['returns'].sum()
            
            # Only use common games
            common_games = set(game_returns1.index) & set(game_returns2.index)
            if common_games:
                returns1 = game_returns1[list(common_games)]
                returns2 = game_returns2[list(common_games)]
        
        # Perform statistical test
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_rel(returns1, returns2)
            test_name = "Paired t-test"
        elif test_type == 'wilcoxon':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = stats.wilcoxon(returns1, returns2, alternative='two-sided')
            test_name = "Wilcoxon signed-rank test"
        else:
            raise ValueError("test_type must be 'ttest' or 'wilcoxon'")
        
        # Effect size (Cohen's d for t-test)
        effect_size = np.nan
        if test_type == 'ttest':
            pooled_std = np.sqrt(((len(returns1) - 1) * returns1.var() + 
                                (len(returns2) - 1) * returns2.var()) / 
                               (len(returns1) + len(returns2) - 2))
            if pooled_std > 0:
                effect_size = (returns1.mean() - returns2.mean()) / pooled_std
        
        return {
            'strategy1': strategy1,
            'strategy2': strategy2,
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05,
            'significant_1pct': p_value < 0.01,
            'effect_size': effect_size,
            'mean_return_diff': returns1.mean() - returns2.mean(),
            'sample_size': min(len(returns1), len(returns2))
        }
    
    def get_trade_attribution(self, strategy_name: str) -> Dict[str, Any]:
        """Analyze trade attribution for a strategy"""
        
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        result = self.results[strategy_name]
        
        if result.trade_history.empty:
            return {'error': 'No trade history available'}
        
        trades = result.trade_history
        
        # PnL attribution by ticker
        ticker_pnl = trades.groupby('ticker')['pnl'].agg(['sum', 'count', 'mean']).round(4)
        ticker_pnl.columns = ['total_pnl', 'num_trades', 'avg_pnl']
        
        # PnL attribution by side
        side_pnl = trades.groupby('side')['pnl'].agg(['sum', 'count', 'mean']).round(4)
        side_pnl.columns = ['total_pnl', 'num_trades', 'avg_pnl']
        
        # Time-based attribution
        trades['hour'] = pd.to_datetime(trades['datetime']).dt.hour
        hourly_pnl = trades.groupby('hour')['pnl'].agg(['sum', 'count', 'mean']).round(4)
        hourly_pnl.columns = ['total_pnl', 'num_trades', 'avg_pnl']
        
        # Game-based attribution if available
        game_pnl = pd.DataFrame()
        if 'game_file' in trades.columns:
            game_pnl = trades.groupby('game_file')['pnl'].agg(['sum', 'count', 'mean']).round(4)
            game_pnl.columns = ['total_pnl', 'num_trades', 'avg_pnl']
        
        return {
            'strategy': strategy_name,
            'ticker_attribution': ticker_pnl.to_dict('index'),
            'side_attribution': side_pnl.to_dict('index'),
            'hourly_attribution': hourly_pnl.to_dict('index'),
            'game_attribution': game_pnl.to_dict('index') if not game_pnl.empty else {},
            'total_pnl': trades['pnl'].sum(),
            'best_trade': trades.loc[trades['pnl'].idxmax()].to_dict() if len(trades) > 0 else {},
            'worst_trade': trades.loc[trades['pnl'].idxmin()].to_dict() if len(trades) > 0 else {}
        }
    
    def export_results(self, filename: str, format: str = 'json'):
        """Export backtest results to file"""
        
        export_data = {}
        
        for strategy_name, result in self.results.items():
            strategy_data = {
                'strategy_name': result.strategy_name,
                'start_time': result.start_time.isoformat() if result.start_time else None,
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'metrics': {
                    'total_return': result.total_return,
                    'annualized_return': result.annualized_return,
                    'volatility': result.volatility,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'calmar_ratio': result.calmar_ratio,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'total_trades': result.total_trades,
                    'var_95': result.var_95,
                    'skewness': result.skewness,
                    'kurtosis': result.kurtosis
                },
                'trade_history': result.trade_history.to_dict('records') if not result.trade_history.empty else [],
                'portfolio_timeseries': result.portfolio_timeseries.to_dict('records') if not result.portfolio_timeseries.empty else []
            }
            export_data[strategy_name] = strategy_data
        
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError("Only JSON format supported currently")
        
        logger.info(f"Results exported to {filename}")