"""
Standalone Backtesting Framework Test

Tests the backtesting infrastructure independently without
relying on the main package imports.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List

# Add the backtesting directory to the path
sys.path.insert(0, "/Users/omarbadran/Desktop/kalshi-claude-RL/src/nfl_trading/backtesting")

# Import backtesting components directly
from trading_environment import TradingEnvironment
from strategies import (
    RuleBasedTrader,
    StatisticalTrader,
    RandomTrader,
    BuyAndHoldTrader,
    MomentumFollower
)
from backtester import Backtester
from performance_analyzer import PerformanceAnalyzer
from reporting import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_data_files(data_dir: str) -> List[str]:
    """Find all JSON data files in the directory"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    json_files = list(data_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} data files in {data_dir}")
    
    return [str(f) for f in json_files]


def test_trading_environment():
    """Test the trading environment with sample data"""
    logger.info("Testing TradingEnvironment...")
    
    try:
        env = TradingEnvironment(initial_capital=10000.0)
        
        # Test basic functionality
        assert env.initial_capital == 10000.0
        assert env.cash == 10000.0
        assert len(env.positions) == 0
        
        logger.info("✓ TradingEnvironment basic tests passed")
        
        # Test loading actual data
        data_file = "/Users/omarbadran/Desktop/kalshi-claude-RL/nfl_candlesticks_data/2025-09-04_Dallas_Cowboys_at_Philadelphia_Eagles_KXNFLGAME-25SEP04DALPHI.json"
        
        if os.path.exists(data_file):
            env.load_market_data(data_file)
            logger.info(f"✓ Successfully loaded data for {len(env.market_data)} tickers")
            
            # Test market state retrieval
            for ticker in env.market_data:
                market_state = env.get_current_market_state(ticker)
                if market_state:
                    logger.info(f"✓ Market state for {ticker}: ${market_state.mid_price:.4f}")
                    break
        else:
            logger.warning(f"Test data file not found: {data_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ TradingEnvironment test failed: {e}")
        return False


def test_strategies():
    """Test strategy creation and signal generation"""
    logger.info("Testing strategies...")
    
    try:
        strategies = [
            RuleBasedTrader(),
            StatisticalTrader(),
            RandomTrader(random_seed=42),
            BuyAndHoldTrader(),
            MomentumFollower()
        ]
        
        logger.info(f"✓ Created {len(strategies)} strategies")
        
        # Test signal generation with mock data
        from trading_environment import MarketState
        
        mock_market_data = {
            "TEST-TICKER": MarketState(
                timestamp=1757028000,
                ticker="TEST-TICKER",
                bid_price=0.45,
                ask_price=0.47,
                mid_price=0.46,
                volume=1000,
                open_interest=50000
            )
        }
        
        env = TradingEnvironment()
        
        for strategy in strategies:
            signals = strategy.generate_signals(env, mock_market_data)
            logger.info(f"✓ {strategy.name} generated {len(signals)} signals")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Strategy test failed: {e}")
        return False


def test_backtester():
    """Test the backtester with actual data"""
    logger.info("Testing Backtester...")
    
    try:
        # Find data files
        data_dir = "/Users/omarbadran/Desktop/kalshi-claude-RL/nfl_candlesticks_data"
        data_files = find_data_files(data_dir)
        
        if not data_files:
            logger.warning("No data files found for backtesting")
            return False
        
        # Use only the first file for quick testing
        test_file = data_files[0]
        logger.info(f"Testing with file: {Path(test_file).name}")
        
        # Create backtester
        backtester = Backtester(
            initial_capital=10000.0,
            fee_rate=0.07,
            slippage_bps=0.5
        )
        
        # Test with a simple strategy
        strategy = RandomTrader(
            trade_probability=0.1,
            random_seed=42
        )
        
        # Run backtest
        result = backtester.run_backtest(
            strategy=strategy,
            data_files=[test_file],
            timestep_seconds=60,
            warmup_minutes=2
        )
        
        logger.info(f"✓ Backtest completed for {strategy.name}")
        logger.info(f"  Total Return: {result.total_return:.2%}")
        logger.info(f"  Total Trades: {result.total_trades}")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Backtester test failed: {e}")
        return False


def run_mini_benchmark():
    """Run a mini benchmark with multiple strategies"""
    logger.info("Running mini benchmark...")
    
    try:
        # Find data files
        data_dir = "/Users/omarbadran/Desktop/kalshi-claude-RL/nfl_candlesticks_data"
        data_files = find_data_files(data_dir)
        
        if not data_files:
            logger.warning("No data files found")
            return False
        
        # Use only first file for speed
        test_files = data_files[:1]
        
        # Create backtester
        backtester = Backtester(initial_capital=10000.0)
        
        # Create strategies
        strategies = [
            RuleBasedTrader(touchdown_buy_size=25, hold_duration_minutes=2),
            RandomTrader(trade_probability=0.05, random_seed=42),
            BuyAndHoldTrader(initial_position_size=50)
        ]
        
        # Run backtests
        results = {}
        for strategy in strategies:
            logger.info(f"Running {strategy.name}...")
            
            try:
                result = backtester.run_backtest(
                    strategy=strategy,
                    data_files=test_files,
                    timestep_seconds=60,
                    warmup_minutes=2
                )
                results[strategy.name] = result
                
            except Exception as e:
                logger.warning(f"Backtest failed for {strategy.name}: {e}")
        
        # Display results
        if results:
            logger.info("\n" + "="*50)
            logger.info("MINI BENCHMARK RESULTS")
            logger.info("="*50)
            
            for name, result in results.items():
                logger.info(f"\n{name}:")
                logger.info(f"  Total Return: {result.total_return:8.2%}")
                logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:8.3f}")
                logger.info(f"  Max Drawdown: {result.max_drawdown:8.2%}")
                logger.info(f"  Total Trades: {result.total_trades:8d}")
                logger.info(f"  Win Rate:     {result.win_rate:8.2%}")
            
            # Find best strategy
            best_strategy = max(results.items(), key=lambda x: x[1].sharpe_ratio)
            logger.info(f"\n✓ Best performing strategy: {best_strategy[0]} (Sharpe: {best_strategy[1].sharpe_ratio:.3f})")
            
            # Basic comparison
            comparison_df = backtester.compare_strategies('sharpe_ratio')
            logger.info(f"\n✓ Strategy comparison completed - {len(comparison_df)} strategies ranked")
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"✗ Mini benchmark failed: {e}")
        return False


def test_performance_analysis():
    """Test performance analysis capabilities"""
    logger.info("Testing performance analysis...")
    
    try:
        # Create a simple backtester with results
        backtester = Backtester()
        
        # We'll create mock results for testing
        from backtester import BacktestResult
        import pandas as pd
        import numpy as np
        
        # Create mock result
        result = BacktestResult("TestStrategy")
        result.total_return = 0.15
        result.sharpe_ratio = 1.2
        result.max_drawdown = 0.08
        result.win_rate = 0.65
        result.total_trades = 50
        
        # Create mock portfolio timeseries
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        portfolio_values = 10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)))
        
        result.portfolio_timeseries = pd.DataFrame({
            'datetime': dates,
            'portfolio_value': portfolio_values,
            'returns': np.random.normal(0.001, 0.02, 100)
        })
        
        backtester.results['TestStrategy'] = result
        
        # Test analyzer
        analyzer = PerformanceAnalyzer(backtester)
        
        # Test dashboard creation
        dashboard = analyzer.create_strategy_comparison_dashboard()
        logger.info(f"✓ Created dashboard with {len(dashboard)} charts")
        
        # Test report generation
        report = analyzer.generate_performance_report()
        logger.info(f"✓ Generated performance report ({len(report)} characters)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance analysis test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("NFL TRADING BACKTESTING FRAMEWORK - STANDALONE TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Trading Environment", test_trading_environment),
        ("Strategies", test_strategies),
        ("Backtester", test_backtester),
        ("Mini Benchmark", run_mini_benchmark),
        ("Performance Analysis", test_performance_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Backtesting framework is working!")
        logger.info("\nThe framework provides:")
        logger.info("• Realistic trading simulation with fees and slippage")
        logger.info("• 5 baseline trading strategies for benchmarking")
        logger.info("• Comprehensive performance and risk metrics")
        logger.info("• Statistical significance testing")
        logger.info("• Interactive visualization dashboards")
        logger.info("• Detailed reporting and data export")
        logger.info("\nML models must beat these baseline performance benchmarks!")
    else:
        logger.error(f"❌ {total - passed} tests failed - framework needs debugging")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)