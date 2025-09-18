"""
Comprehensive Test Framework

Tests the entire backtesting infrastructure with historical data
and generates baseline performance benchmarks.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_trading.backtesting import (
    TradingEnvironment,
    Backtester,
    PerformanceAnalyzer,
    ReportGenerator,
    RuleBasedTrader,
    StatisticalTrader,
    RandomTrader,
    BuyAndHoldTrader,
    MomentumFollower
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


def create_baseline_strategies() -> List:
    """Create all baseline trading strategies"""
    
    strategies = [
        # Rule-based strategy
        RuleBasedTrader(
            touchdown_buy_size=50,
            field_goal_sell_size=30,
            hold_duration_minutes=3,
            min_price_move=0.02
        ),
        
        # Statistical strategy
        StatisticalTrader(
            lookback_window=20,
            rsi_period=14,
            bollinger_std=2.0,
            mean_reversion_threshold=0.03,
            momentum_threshold=0.02
        ),
        
        # Random strategy (for comparison)
        RandomTrader(
            trade_probability=0.05,
            max_position_size=100,
            position_size_range=(10, 50),
            random_seed=42  # For reproducible results
        ),
        
        # Buy and hold strategy
        BuyAndHoldTrader(
            initial_position_size=100,
            buy_spread_minutes=5
        ),
        
        # Momentum following strategy
        MomentumFollower(
            momentum_lookback=5,
            momentum_threshold=0.015,
            position_size=40,
            max_position_size=200
        )
    ]
    
    logger.info(f"Created {len(strategies)} baseline strategies")
    return strategies


def run_comprehensive_backtest(
    data_dir: str = "/Users/omarbadran/Desktop/kalshi-claude-RL/nfl_candlesticks_data",
    output_dir: str = "backtest_results"
) -> Backtester:
    """Run comprehensive backtest on all strategies"""
    
    logger.info("Starting comprehensive backtest...")
    
    # Find data files
    data_files = find_data_files(data_dir)
    
    if not data_files:
        raise ValueError("No data files found")
    
    # Create backtester
    backtester = Backtester(
        initial_capital=10000.0,
        fee_rate=0.07,  # 7% fee (Kalshi-like)
        slippage_bps=0.5,
        risk_free_rate=0.02
    )
    
    # Create strategies
    strategies = create_baseline_strategies()
    
    # Run backtest for each strategy
    results = {}
    for i, strategy in enumerate(strategies, 1):
        logger.info(f"Running backtest {i}/{len(strategies)}: {strategy.name}")
        
        try:
            result = backtester.run_backtest(
                strategy=strategy,
                data_files=data_files,
                timestep_seconds=60,  # 1-minute steps
                warmup_minutes=5      # 5-minute warmup
            )
            results[strategy.name] = result
            
        except Exception as e:
            logger.error(f"Backtest failed for {strategy.name}: {e}")
            continue
    
    logger.info(f"Completed backtests for {len(results)} strategies")
    return backtester


def generate_performance_benchmark(backtester: Backtester) -> dict:
    """Generate performance benchmark data"""
    
    logger.info("Generating performance benchmarks...")
    
    # Strategy comparison
    comparison_df = backtester.compare_strategies('sharpe_ratio')
    
    if comparison_df.empty:
        logger.warning("No comparison data available")
        return {}
    
    # Calculate benchmark metrics
    benchmark = {
        'best_strategy': {
            'name': comparison_df.iloc[0]['strategy'],
            'sharpe_ratio': comparison_df.iloc[0]['sharpe_ratio'],
            'total_return': comparison_df.iloc[0]['total_return'],
            'max_drawdown': comparison_df.iloc[0]['max_drawdown'],
            'win_rate': comparison_df.iloc[0]['win_rate']
        },
        'worst_strategy': {
            'name': comparison_df.iloc[-1]['strategy'],
            'sharpe_ratio': comparison_df.iloc[-1]['sharpe_ratio'],
            'total_return': comparison_df.iloc[-1]['total_return'],
            'max_drawdown': comparison_df.iloc[-1]['max_drawdown'],
            'win_rate': comparison_df.iloc[-1]['win_rate']
        },
        'average_metrics': {
            'sharpe_ratio': comparison_df['sharpe_ratio'].mean(),
            'total_return': comparison_df['total_return'].mean(),
            'max_drawdown': comparison_df['max_drawdown'].mean(),
            'win_rate': comparison_df['win_rate'].mean(),
            'volatility': comparison_df['volatility'].mean()
        },
        'benchmark_thresholds': {
            'minimum_sharpe_to_beat': comparison_df['sharpe_ratio'].max(),
            'minimum_return_to_beat': comparison_df['total_return'].max(),
            'maximum_drawdown_acceptable': comparison_df['max_drawdown'].min(),
            'minimum_win_rate_target': comparison_df['win_rate'].mean() + comparison_df['win_rate'].std()
        }
    }
    
    logger.info("Performance benchmarks generated")
    logger.info(f"Best strategy: {benchmark['best_strategy']['name']} (Sharpe: {benchmark['best_strategy']['sharpe_ratio']:.3f})")
    logger.info(f"ML models must beat Sharpe ratio of: {benchmark['benchmark_thresholds']['minimum_sharpe_to_beat']:.3f}")
    
    return benchmark


def run_statistical_analysis(backtester: Backtester):
    """Run statistical significance analysis"""
    
    logger.info("Running statistical significance analysis...")
    
    strategies = list(backtester.results.keys())
    
    if len(strategies) < 2:
        logger.warning("Need at least 2 strategies for significance testing")
        return
    
    # Test all pairs
    significant_differences = []
    
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            try:
                test_result = backtester.statistical_significance_test(
                    strategies[i], strategies[j]
                )
                
                if 'error' not in test_result and test_result['significant_5pct']:
                    significant_differences.append({
                        'strategy1': strategies[i],
                        'strategy2': strategies[j],
                        'p_value': test_result['p_value'],
                        'mean_diff': test_result['mean_return_diff']
                    })
                    
            except Exception as e:
                logger.warning(f"Statistical test failed for {strategies[i]} vs {strategies[j]}: {e}")
    
    logger.info(f"Found {len(significant_differences)} statistically significant differences")
    
    for diff in significant_differences:
        logger.info(f"{diff['strategy1']} vs {diff['strategy2']}: p={diff['p_value']:.4f}")


def create_comprehensive_reports(backtester: Backtester, output_dir: str = "backtest_results"):
    """Create comprehensive reports"""
    
    logger.info("Creating comprehensive reports...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create report generator
    report_generator = ReportGenerator(backtester)
    
    # Generate main HTML report
    html_report = report_generator.generate_comprehensive_report(
        output_dir=output_dir,
        report_name="nfl_trading_baseline_benchmark",
        include_charts=True,
        export_data=True
    )
    
    logger.info(f"HTML report created: {html_report}")
    
    # Create individual strategy tearsheets
    for strategy_name in backtester.results.keys():
        try:
            safe_name = strategy_name.replace(' ', '_').replace('/', '_')
            tearsheet_file = output_path / f"tearsheet_{safe_name}.html"
            
            report_generator.create_strategy_tearsheet(
                strategy_name=strategy_name,
                output_file=str(tearsheet_file)
            )
            
            logger.info(f"Tearsheet created for {strategy_name}: {tearsheet_file}")
            
        except Exception as e:
            logger.warning(f"Could not create tearsheet for {strategy_name}: {e}")
    
    # Export CSV files
    csv_files = report_generator.export_to_csv(output_dir + "/csv_exports")
    logger.info(f"CSV files exported: {len(csv_files)} files")
    
    # Export JSON
    json_file = report_generator.export_to_json(str(output_path / "results.json"))
    logger.info(f"JSON results exported: {json_file}")
    
    # Create text performance report
    analyzer = PerformanceAnalyzer(backtester)
    text_report = analyzer.generate_performance_report(
        str(output_path / "performance_summary.txt")
    )
    
    logger.info("All reports generated successfully")


def test_individual_components():
    """Test individual components in isolation"""
    
    logger.info("Testing individual components...")
    
    # Test trading environment
    logger.info("Testing TradingEnvironment...")
    env = TradingEnvironment()
    
    # Test loading data
    data_file = "/Users/omarbadran/Desktop/kalshi-claude-RL/nfl_candlesticks_data/2025-09-04_Dallas_Cowboys_at_Philadelphia_Eagles_KXNFLGAME-25SEP04DALPHI.json"
    if os.path.exists(data_file):
        try:
            env.load_market_data(data_file)
            logger.info(f"✓ Successfully loaded market data: {len(env.market_data)} tickers")
        except Exception as e:
            logger.error(f"✗ Failed to load market data: {e}")
    else:
        logger.warning(f"Test data file not found: {data_file}")
    
    # Test strategy creation
    logger.info("Testing strategy creation...")
    try:
        strategies = create_baseline_strategies()
        logger.info(f"✓ Successfully created {len(strategies)} strategies")
        
        # Test strategy signal generation
        strategy = strategies[0]
        if env.market_data:
            market_data = {}
            for ticker in env.market_data:
                market_state = env.get_current_market_state(ticker)
                if market_state:
                    market_data[ticker] = market_state
            
            if market_data:
                signals = strategy.generate_signals(env, market_data)
                logger.info(f"✓ Strategy {strategy.name} generated {len(signals)} signals")
        
    except Exception as e:
        logger.error(f"✗ Strategy testing failed: {e}")
    
    logger.info("Component testing completed")


def main():
    """Main test function"""
    
    logger.info("=" * 80)
    logger.info("NFL TRADING BACKTESTING FRAMEWORK - COMPREHENSIVE TEST")
    logger.info("=" * 80)
    
    try:
        # Test individual components first
        test_individual_components()
        
        # Run comprehensive backtest
        backtester = run_comprehensive_backtest()
        
        if not backtester.results:
            logger.error("No backtest results generated - cannot continue")
            return
        
        # Generate performance benchmarks
        benchmark = generate_performance_benchmark(backtester)
        
        # Run statistical analysis
        run_statistical_analysis(backtester)
        
        # Create comprehensive reports
        create_comprehensive_reports(backtester)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 80)
        
        comparison_df = backtester.compare_strategies('sharpe_ratio')
        
        logger.info(f"Strategies tested: {len(backtester.results)}")
        logger.info(f"Best performer: {comparison_df.iloc[0]['strategy']} (Sharpe: {comparison_df.iloc[0]['sharpe_ratio']:.3f})")
        logger.info(f"Worst performer: {comparison_df.iloc[-1]['strategy']} (Sharpe: {comparison_df.iloc[-1]['sharpe_ratio']:.3f})")
        
        if benchmark:
            logger.info(f"\nBENCHMARK FOR ML MODELS:")
            logger.info(f"Minimum Sharpe ratio to beat: {benchmark['benchmark_thresholds']['minimum_sharpe_to_beat']:.3f}")
            logger.info(f"Minimum return to beat: {benchmark['benchmark_thresholds']['minimum_return_to_beat']:.2%}")
            logger.info(f"Maximum acceptable drawdown: {benchmark['benchmark_thresholds']['maximum_drawdown_acceptable']:.2%}")
        
        logger.info("\n✓ Comprehensive test completed successfully!")
        logger.info("Check the 'backtest_results' directory for detailed reports.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()