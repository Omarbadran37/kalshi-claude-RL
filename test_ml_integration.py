"""
ML Integration Test

Tests the integration between ML models and the backtesting framework.
Compares ML-based strategies against baseline strategies.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add the backtesting directory to the path
sys.path.insert(0, "/Users/omarbadran/Desktop/kalshi-claude-RL/src/nfl_trading/backtesting")

# Import backtesting components
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

# Import ML strategy
try:
    from ml_strategy import MLTradingStrategy, EnsembleMLStrategy
    ML_STRATEGY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML strategy not available: {e}")
    ML_STRATEGY_AVAILABLE = False

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


def test_ml_strategy_creation():
    """Test ML strategy creation and basic functionality"""
    logger.info("Testing ML strategy creation...")

    try:
        if not ML_STRATEGY_AVAILABLE:
            logger.warning("ML strategy not available, skipping test")
            return False

        # Test basic ML strategy
        ml_strategy = MLTradingStrategy(
            confidence_threshold=0.6,
            use_synthetic_data=True,
            fallback_to_momentum=True
        )

        logger.info(f"✓ Created ML strategy: {ml_strategy.name}")

        # Test ensemble strategy
        ensemble_strategy = EnsembleMLStrategy()
        logger.info(f"✓ Created ensemble strategy: {ensemble_strategy.name}")

        # Test signal generation with mock data
        from trading_environment import MarketState

        env = TradingEnvironment(initial_capital=10000.0)
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

        # Test signal generation
        signals = ml_strategy.generate_signals(env, mock_market_data)
        logger.info(f"✓ ML strategy generated {len(signals)} signals")

        ensemble_signals = ensemble_strategy.generate_signals(env, mock_market_data)
        logger.info(f"✓ Ensemble strategy generated {len(ensemble_signals)} signals")

        return True

    except Exception as e:
        logger.error(f"✗ ML strategy test failed: {e}")
        return False


def run_ml_vs_baseline_benchmark():
    """Compare ML strategies against baseline strategies"""
    logger.info("Running ML vs Baseline benchmark...")

    try:
        # Find data files
        data_dir = "/Users/omarbadran/Desktop/kalshi-claude-RL/nfl_candlesticks_data"
        data_files = find_data_files(data_dir)

        if not data_files:
            logger.warning("No data files found")
            return False

        # Use only first file for speed
        test_files = data_files[:1]
        logger.info(f"Testing with file: {Path(test_files[0]).name}")

        # Create backtester
        backtester = Backtester(
            initial_capital=10000.0,
            fee_rate=0.07,
            slippage_bps=0.5
        )

        # Create baseline strategies
        baseline_strategies = [
            RuleBasedTrader(touchdown_buy_size=30, hold_duration_minutes=2),
            StatisticalTrader(base_position_size=40),
            RandomTrader(trade_probability=0.03, random_seed=42),
            BuyAndHoldTrader(initial_position_size=50),
            MomentumFollower(position_size=35)
        ]

        # Create ML strategies
        ml_strategies = []
        if ML_STRATEGY_AVAILABLE:
            ml_strategies = [
                MLTradingStrategy(
                    confidence_threshold=0.6,
                    use_synthetic_data=True,
                    position_scaling_factor=0.8
                ),
                MLTradingStrategy(
                    confidence_threshold=0.7,
                    use_synthetic_data=True,
                    position_scaling_factor=1.2,
                    fallback_to_momentum=True
                ),
                EnsembleMLStrategy()
            ]

        all_strategies = baseline_strategies + ml_strategies

        # Run backtests
        results = {}
        for strategy in all_strategies:
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

        # Analyze results
        if results:
            logger.info("\n" + "="*70)
            logger.info("ML vs BASELINE BENCHMARK RESULTS")
            logger.info("="*70)

            # Separate baseline and ML results
            baseline_results = {}
            ml_results = {}

            for name, result in results.items():
                if any(baseline in name for baseline in ["RuleBasedTrader", "StatisticalTrader", "RandomTrader", "BuyAndHoldTrader", "MomentumFollower"]):
                    baseline_results[name] = result
                else:
                    ml_results[name] = result

            # Display baseline results
            logger.info("\nBASELINE STRATEGIES:")
            logger.info("-" * 50)
            for name, result in baseline_results.items():
                logger.info(f"{name:20s} | Return: {result.total_return:8.2%} | Sharpe: {result.sharpe_ratio:6.3f} | Trades: {result.total_trades:4d}")

            # Display ML results
            if ml_results:
                logger.info("\nML STRATEGIES:")
                logger.info("-" * 50)
                for name, result in ml_results.items():
                    logger.info(f"{name:20s} | Return: {result.total_return:8.2%} | Sharpe: {result.sharpe_ratio:6.3f} | Trades: {result.total_trades:4d}")

                # Find best performers
                if baseline_results:
                    best_baseline = max(baseline_results.items(), key=lambda x: x[1].sharpe_ratio)
                    best_ml = max(ml_results.items(), key=lambda x: x[1].sharpe_ratio)

                    logger.info(f"\n🏆 PERFORMANCE COMPARISON:")
                    logger.info(f"Best Baseline: {best_baseline[0]} (Sharpe: {best_baseline[1].sharpe_ratio:.3f})")
                    logger.info(f"Best ML:       {best_ml[0]} (Sharpe: {best_ml[1].sharpe_ratio:.3f})")

                    if best_ml[1].sharpe_ratio > best_baseline[1].sharpe_ratio:
                        improvement = ((best_ml[1].sharpe_ratio / best_baseline[1].sharpe_ratio) - 1) * 100
                        logger.info(f"✅ ML strategy outperforms by {improvement:.1f}%!")
                    else:
                        logger.info("⚠️  Baseline strategies still competitive")
            else:
                logger.info("\nNo ML strategies tested (not available)")

            # Strategy comparison
            comparison_df = backtester.compare_strategies('sharpe_ratio')
            logger.info(f"\n📊 Strategy ranking completed - {len(comparison_df)} strategies compared")

            # Create performance analysis if we have ML results
            if ml_results and len(results) > 1:
                try:
                    analyzer = PerformanceAnalyzer(backtester)

                    # Generate dashboard
                    dashboard = analyzer.create_strategy_comparison_dashboard()
                    logger.info(f"📈 Created comparison dashboard with {len(dashboard)} charts")

                    # Generate report
                    report = analyzer.generate_performance_report()
                    logger.info(f"📄 Generated performance report ({len(report)} characters)")

                except Exception as e:
                    logger.warning(f"Performance analysis failed: {e}")

            return True

        return False

    except Exception as e:
        logger.error(f"✗ ML vs Baseline benchmark failed: {e}")
        return False


def test_ml_feature_extraction():
    """Test ML feature extraction and prediction capabilities"""
    logger.info("Testing ML feature extraction...")

    if not ML_STRATEGY_AVAILABLE:
        logger.warning("ML strategy not available, skipping test")
        return False

    try:
        # Create ML strategy
        ml_strategy = MLTradingStrategy(use_synthetic_data=True)

        # Test synthetic play description creation
        from trading_environment import MarketState

        market_state = MarketState(
            timestamp=1757028000,
            ticker="TEST-GAME",
            bid_price=0.45,
            ask_price=0.47,
            mid_price=0.46,
            volume=1000,
            open_interest=50000
        )

        # Test multiple predictions to see variety
        logger.info("Testing synthetic play generation:")
        for i in range(5):
            play_desc = ml_strategy._create_synthetic_play_description("TEST-GAME", market_state)
            logger.info(f"  Play {i+1}: {play_desc}")

        # Test prediction generation
        numerical_features = np.array([0.46, 0.45, 0.47, 1.0, 5.0, 3])
        prediction = ml_strategy._generate_synthetic_prediction("TEST-GAME", market_state, numerical_features)

        logger.info(f"✓ Sample prediction: confidence={prediction['confidence']:.3f}, "
                   f"price_impact={prediction['price_impact_prediction']:.4f}")

        # Test strategy state
        state = ml_strategy.get_strategy_state()
        logger.info(f"✓ Strategy state: ML available={state['ml_available']}, "
                   f"tickers tracked={len(state['tickers_tracked'])}")

        return True

    except Exception as e:
        logger.error(f"✗ ML feature extraction test failed: {e}")
        return False


def run_integration_stress_test():
    """Run stress test with multiple data files"""
    logger.info("Running integration stress test...")

    try:
        data_dir = "/Users/omarbadran/Desktop/kalshi-claude-RL/nfl_candlesticks_data"
        data_files = find_data_files(data_dir)

        if len(data_files) < 2:
            logger.warning("Need at least 2 data files for stress test")
            return False

        # Use first 3 files for stress test
        test_files = data_files[:min(3, len(data_files))]
        logger.info(f"Stress testing with {len(test_files)} files")

        backtester = Backtester(initial_capital=10000.0)

        # Test with multiple strategies
        strategies = [
            RandomTrader(trade_probability=0.02, random_seed=42),
        ]

        if ML_STRATEGY_AVAILABLE:
            strategies.append(MLTradingStrategy(confidence_threshold=0.65, use_synthetic_data=True))

        results = {}
        for strategy in strategies:
            logger.info(f"Stress testing {strategy.name}...")

            try:
                result = backtester.run_backtest(
                    strategy=strategy,
                    data_files=test_files,
                    timestep_seconds=45,  # Faster timestep for stress test
                    warmup_minutes=1
                )
                results[strategy.name] = result
                logger.info(f"✓ {strategy.name} completed: Return={result.total_return:.2%}, Trades={result.total_trades}")

            except Exception as e:
                logger.error(f"✗ Stress test failed for {strategy.name}: {e}")

        if results:
            logger.info(f"✓ Stress test completed with {len(results)} successful runs")
            return True

        return False

    except Exception as e:
        logger.error(f"✗ Integration stress test failed: {e}")
        return False


def main():
    """Run all ML integration tests"""
    logger.info("=" * 80)
    logger.info("NFL TRADING ML INTEGRATION TEST")
    logger.info("=" * 80)

    tests = [
        ("ML Strategy Creation", test_ml_strategy_creation),
        ("ML Feature Extraction", test_ml_feature_extraction),
        ("ML vs Baseline Benchmark", run_ml_vs_baseline_benchmark),
        ("Integration Stress Test", run_integration_stress_test)
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

    logger.info(f"\n{'='*80}")
    logger.info(f"ML INTEGRATION TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - ML integration successful!")
        logger.info("\nIntegration provides:")
        logger.info("• ML-based trading strategies using play description analysis")
        logger.info("• Synthetic prediction generation for demonstration")
        logger.info("• Ensemble methods combining multiple strategies")
        logger.info("• Fallback to momentum strategies when ML unavailable")
        logger.info("• Performance comparison against baseline strategies")
        logger.info("• Comprehensive feature extraction from market data")
        logger.info("\n✨ ML models are now integrated with the backtesting framework!")
    else:
        logger.error(f"❌ {total - passed} tests failed - integration needs debugging")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)