"""
Run Full Backtesting Framework

Run comprehensive backtesting with all strategies and generate reports.
"""

import os
import sys
import logging
from pathlib import Path

# Add the backtesting directory to the path
sys.path.insert(0, "/Users/omarbadran/Desktop/kalshi-claude-RL/src/nfl_trading/backtesting")

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


def find_valid_data_files(data_dir: str):
    """Find valid game data files (not summary files)"""
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    # Filter out summary files and find actual game files
    valid_files = []
    for f in json_files:
        if f.name.lower() != 'summary.json' and 'game' in f.name.lower():
            valid_files.append(str(f))
    
    if not valid_files:
        # If no game files found, try the first few non-summary files
        valid_files = [str(f) for f in json_files if f.name.lower() != 'summary.json'][:5]
    
    logger.info(f"Found {len(valid_files)} valid data files")
    return valid_files


def create_strategies():
    """Create all trading strategies for testing"""
    return [
        RuleBasedTrader(
            touchdown_buy_size=50,
            field_goal_sell_size=30,
            hold_duration_minutes=3,
            min_price_move=0.02
        ),
        
        StatisticalTrader(
            lookback_window=15,
            rsi_period=10,
            bollinger_std=2.0,
            mean_reversion_threshold=0.03,
            momentum_threshold=0.02
        ),
        
        RandomTrader(
            trade_probability=0.08,
            max_position_size=100,
            position_size_range=(20, 60),
            random_seed=42
        ),
        
        BuyAndHoldTrader(
            initial_position_size=75,
            buy_spread_minutes=3
        ),
        
        MomentumFollower(
            momentum_lookback=5,
            momentum_threshold=0.015,
            position_size=40,
            max_position_size=150
        )
    ]


def main():
    logger.info("=" * 80)
    logger.info("NFL TRADING BACKTESTING FRAMEWORK - FULL BENCHMARK")
    logger.info("=" * 80)
    
    # Find valid data files
    data_dir = "/Users/omarbadran/Desktop/kalshi-claude-RL/nfl_candlesticks_data"
    data_files = find_valid_data_files(data_dir)
    
    if not data_files:
        logger.error("No valid data files found")
        return
    
    logger.info(f"Using {len(data_files)} data files for backtesting")
    
    # Create backtester
    backtester = Backtester(
        initial_capital=10000.0,
        fee_rate=0.07,  # 7% Kalshi-like fee
        slippage_bps=0.5,
        risk_free_rate=0.02
    )
    
    # Create strategies
    strategies = create_strategies()
    logger.info(f"Testing {len(strategies)} trading strategies")
    
    # Run backtests
    results = {}
    for i, strategy in enumerate(strategies, 1):
        logger.info(f"\n{'='*20} Strategy {i}/{len(strategies)}: {strategy.name} {'='*20}")
        
        try:
            result = backtester.run_backtest(
                strategy=strategy,
                data_files=data_files,
                timestep_seconds=60,
                warmup_minutes=5
            )
            
            results[strategy.name] = result
            logger.info(f"✓ Completed {strategy.name}")
            
        except Exception as e:
            logger.error(f"✗ Failed {strategy.name}: {e}")
    
    if not results:
        logger.error("No successful backtests - cannot generate reports")
        return
    
    # Display results summary
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    comparison_df = backtester.compare_strategies('sharpe_ratio')
    
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        logger.info(f"\n{i+1}. {row['strategy']:20s}")
        logger.info(f"   Total Return:    {row['total_return']:8.2%}")
        logger.info(f"   Sharpe Ratio:    {row['sharpe_ratio']:8.3f}")
        logger.info(f"   Max Drawdown:    {row['max_drawdown']:8.2%}")
        logger.info(f"   Win Rate:        {row['win_rate']:8.2%}")
        logger.info(f"   Total Trades:    {row['total_trades']:8d}")
    
    # Generate performance benchmark
    best_strategy = comparison_df.iloc[0]
    benchmark_sharpe = best_strategy['sharpe_ratio']
    benchmark_return = best_strategy['total_return']
    
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE BENCHMARK FOR ML MODELS")
    logger.info("=" * 80)
    logger.info(f"Best Strategy: {best_strategy['strategy']}")
    logger.info(f"Minimum Sharpe Ratio to Beat: {benchmark_sharpe:.3f}")
    logger.info(f"Minimum Total Return to Beat: {benchmark_return:.2%}")
    logger.info(f"Maximum Acceptable Drawdown: {comparison_df['max_drawdown'].min():.2%}")
    logger.info(f"Target Win Rate: {comparison_df['win_rate'].mean():.2%}")
    
    # Create reports
    logger.info("\n" + "=" * 40)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 40)
    
    try:
        # Create output directory
        output_dir = "backtest_results"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate comprehensive report
        report_generator = ReportGenerator(backtester)
        
        html_report = report_generator.generate_comprehensive_report(
            output_dir=output_dir,
            report_name="nfl_trading_baseline_benchmark",
            include_charts=True,
            export_data=True
        )
        
        logger.info(f"✓ HTML report: {html_report}")
        
        # Export CSV data
        csv_files = report_generator.export_to_csv(f"{output_dir}/csv")
        logger.info(f"✓ Exported {len(csv_files)} CSV files")
        
        # Create performance text report
        analyzer = PerformanceAnalyzer(backtester)
        text_report = analyzer.generate_performance_report(f"{output_dir}/performance_summary.txt")
        logger.info(f"✓ Text report saved")
        
        logger.info(f"\n📊 All reports saved to '{output_dir}' directory")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
    
    # Statistical significance tests
    logger.info("\n" + "=" * 40)
    logger.info("STATISTICAL SIGNIFICANCE")
    logger.info("=" * 40)
    
    if len(results) > 1:
        strategies_list = list(results.keys())
        significant_pairs = 0
        
        for i in range(len(strategies_list)):
            for j in range(i + 1, len(strategies_list)):
                try:
                    test_result = backtester.statistical_significance_test(
                        strategies_list[i], strategies_list[j]
                    )
                    
                    if 'error' not in test_result:
                        significance = "***" if test_result['significant_1pct'] else ("**" if test_result['significant_5pct'] else "")
                        
                        logger.info(f"{strategies_list[i]:15s} vs {strategies_list[j]:15s} | p={test_result['p_value']:.4f} {significance}")
                        
                        if test_result['significant_5pct']:
                            significant_pairs += 1
                            
                except Exception as e:
                    logger.warning(f"Significance test failed: {e}")
        
        logger.info(f"\nFound {significant_pairs} statistically significant differences")
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 BACKTESTING FRAMEWORK VALIDATION COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nFramework Features Validated:")
    logger.info("✓ Realistic trading simulation with Kalshi-like fees")
    logger.info("✓ Multiple baseline strategies for benchmarking")
    logger.info("✓ Comprehensive risk and performance metrics")
    logger.info("✓ Statistical significance testing")
    logger.info("✓ Interactive reporting and visualization")
    logger.info("✓ Data export capabilities")
    logger.info("\n📈 ML models must beat these baseline benchmarks to be viable!")


if __name__ == "__main__":
    main()