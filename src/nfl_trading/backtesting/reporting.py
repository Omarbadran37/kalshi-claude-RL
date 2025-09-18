"""
Reporting System

Comprehensive reporting system with interactive visualizations,
HTML reports, and data export capabilities.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import logging
import base64
from io import BytesIO

try:
    from .backtester import Backtester, BacktestResult
    from .performance_analyzer import PerformanceAnalyzer
except ImportError:
    from backtester import Backtester, BacktestResult
    from performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Comprehensive reporting system for backtesting results.
    
    Features:
    - Interactive HTML reports with Plotly visualizations
    - CSV/JSON data export
    - Performance summary dashboards
    - Risk analysis reports
    - Trade-level analysis
    """
    
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        self.analyzer = PerformanceAnalyzer(backtester)
        
    def generate_comprehensive_report(
        self,
        output_dir: str = "backtest_reports",
        report_name: str = None,
        include_charts: bool = True,
        export_data: bool = True
    ) -> str:
        """
        Generate comprehensive HTML report with all analysis.
        
        Args:
            output_dir: Directory to save reports
            report_name: Name for the report (auto-generated if None)
            include_charts: Include interactive charts
            export_data: Export raw data files
            
        Returns:
            Path to generated HTML report
        """
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate report name
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"backtest_report_{timestamp}"
        
        html_file = output_path / f"{report_name}.html"
        
        logger.info(f"Generating comprehensive report: {html_file}")
        
        # Generate HTML content
        html_content = self._create_html_report(include_charts)
        
        # Save HTML report
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Export data files if requested
        if export_data:
            self._export_data_files(output_path, report_name)
        
        logger.info(f"Report generated successfully: {html_file}")
        return str(html_file)
    
    def _create_html_report(self, include_charts: bool = True) -> str:
        """Create comprehensive HTML report"""
        
        html_parts = []
        
        # HTML header and CSS
        html_parts.append(self._get_html_header())
        
        # Executive summary
        html_parts.append(self._create_executive_summary())
        
        # Strategy comparison table
        html_parts.append(self._create_strategy_table_html())
        
        if include_charts and self.backtester.results:
            # Performance charts
            html_parts.append(self._create_charts_section())
            
            # Risk analysis
            html_parts.append(self._create_risk_analysis_section())
            
            # Trade analysis
            html_parts.append(self._create_trade_analysis_section())
        
        # Statistical significance tests
        html_parts.append(self._create_significance_tests_section())
        
        # Footer
        html_parts.append(self._get_html_footer())
        
        return "\n".join(html_parts)
    
    def _get_html_header(self) -> str:
        """Get HTML header with CSS styling"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFL Trading Strategy Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }
        h3 {
            color: #5d6d7e;
            margin-top: 25px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: white;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
        }
        .chart-container {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }
        .alert {
            padding: 15px;
            margin: 20px 0;
            border: 1px solid transparent;
            border-radius: 4px;
        }
        .alert-info {
            color: #31708f;
            background-color: #d9edf7;
            border-color: #bce8f1;
        }
        .alert-warning {
            color: #8a6d3b;
            background-color: #fcf8e3;
            border-color: #faebcc;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>NFL Trading Strategy Backtest Report</h1>
        <div class="alert alert-info">
            <strong>Report Generated:</strong> ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''<br>
            <strong>Strategies Analyzed:</strong> ''' + str(len(self.backtester.results)) + '''
        </div>
'''
    
    def _create_executive_summary(self) -> str:
        """Create executive summary section"""
        
        if not self.backtester.results:
            return "<h2>Executive Summary</h2><p>No backtest results available.</p>"
        
        # Find best performing strategy
        best_strategy = max(self.backtester.results.items(), 
                          key=lambda x: x[1].sharpe_ratio)
        
        # Calculate aggregate statistics
        total_trades = sum(result.total_trades for result in self.backtester.results.values())
        avg_return = sum(result.total_return for result in self.backtester.results.values()) / len(self.backtester.results)
        avg_sharpe = sum(result.sharpe_ratio for result in self.backtester.results.values()) / len(self.backtester.results)
        
        html = f'''
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Best Strategy</h3>
                <div class="value">{best_strategy[0]}</div>
                <div>Sharpe: {best_strategy[1].sharpe_ratio:.3f}</div>
            </div>
            <div class="summary-card">
                <h3>Average Return</h3>
                <div class="value">{avg_return:.2%}</div>
                <div>Across all strategies</div>
            </div>
            <div class="summary-card">
                <h3>Total Trades</h3>
                <div class="value">{total_trades:,}</div>
                <div>All strategies combined</div>
            </div>
            <div class="summary-card">
                <h3>Average Sharpe</h3>
                <div class="value">{avg_sharpe:.3f}</div>
                <div>Risk-adjusted performance</div>
            </div>
        </div>
        
        <h3>Key Findings</h3>
        <ul>
            <li><strong>Top Performer:</strong> {best_strategy[0]} achieved the highest Sharpe ratio of {best_strategy[1].sharpe_ratio:.3f}</li>
            <li><strong>Return Range:</strong> Strategy returns ranged from {min(r.total_return for r in self.backtester.results.values()):.2%} to {max(r.total_return for r in self.backtester.results.values()):.2%}</li>
            <li><strong>Risk Profile:</strong> Maximum drawdowns ranged from {min(r.max_drawdown for r in self.backtester.results.values()):.2%} to {max(r.max_drawdown for r in self.backtester.results.values()):.2%}</li>
        </ul>
        '''
        
        return html
    
    def _create_strategy_table_html(self) -> str:
        """Create strategy comparison table"""
        
        if not self.backtester.results:
            return "<h2>Strategy Performance</h2><p>No results available.</p>"
        
        comparison_df = self.backtester.compare_strategies('sharpe_ratio')
        
        html = '''
        <h2>Strategy Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Strategy</th>
                    <th>Total Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Total Trades</th>
                    <th>Profit Factor</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            return_class = "positive" if row['total_return'] > 0 else "negative"
            html += f'''
                <tr>
                    <td>{i+1}</td>
                    <td><strong>{row['strategy']}</strong></td>
                    <td class="{return_class}">{row['total_return']:.2%}</td>
                    <td>{row['sharpe_ratio']:.3f}</td>
                    <td class="negative">{row['max_drawdown']:.2%}</td>
                    <td>{row['win_rate']:.2%}</td>
                    <td>{row['total_trades']}</td>
                    <td>{row['profit_factor']:.2f}</td>
                </tr>
            '''
        
        html += '''
            </tbody>
        </table>
        '''
        
        return html
    
    def _create_charts_section(self) -> str:
        """Create performance charts section"""
        
        html = '<h2>Performance Visualizations</h2>'
        
        try:
            # Get dashboard figures
            dashboard = self.analyzer.create_strategy_comparison_dashboard()
            
            # Cumulative returns chart
            if 'cumulative_returns' in dashboard:
                fig_html = pio.to_html(dashboard['cumulative_returns'], 
                                     include_plotlyjs=False, div_id="cumulative_returns")
                html += f'''
                <div class="chart-container">
                    <h3>Cumulative Returns</h3>
                    {fig_html}
                </div>
                '''
            
            # Risk-return scatter
            if 'risk_return_scatter' in dashboard:
                fig_html = pio.to_html(dashboard['risk_return_scatter'], 
                                     include_plotlyjs=False, div_id="risk_return")
                html += f'''
                <div class="chart-container">
                    <h3>Risk-Return Analysis</h3>
                    {fig_html}
                </div>
                '''
            
            # Drawdown analysis
            if 'drawdown_analysis' in dashboard:
                fig_html = pio.to_html(dashboard['drawdown_analysis'], 
                                     include_plotlyjs=False, div_id="drawdowns")
                html += f'''
                <div class="chart-container">
                    <h3>Drawdown Analysis</h3>
                    {fig_html}
                </div>
                '''
                
        except Exception as e:
            logger.warning(f"Error creating charts: {e}")
            html += '<div class="alert alert-warning">Charts could not be generated due to an error.</div>'
        
        return html
    
    def _create_risk_analysis_section(self) -> str:
        """Create risk analysis section"""
        
        html = '<h2>Risk Analysis</h2>'
        
        # Create risk metrics table
        html += '''
        <h3>Risk Metrics Summary</h3>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Volatility</th>
                    <th>VaR (95%)</th>
                    <th>CVaR (95%)</th>
                    <th>Skewness</th>
                    <th>Kurtosis</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for strategy_name, result in self.backtester.results.items():
            html += f'''
                <tr>
                    <td><strong>{strategy_name}</strong></td>
                    <td>{result.volatility:.2%}</td>
                    <td class="negative">{result.var_95:.2%}</td>
                    <td class="negative">{result.cvar_95:.2%}</td>
                    <td>{result.skewness:.3f}</td>
                    <td>{result.kurtosis:.3f}</td>
                </tr>
            '''
        
        html += '''
            </tbody>
        </table>
        '''
        
        return html
    
    def _create_trade_analysis_section(self) -> str:
        """Create trade analysis section"""
        
        html = '<h2>Trade Analysis</h2>'
        
        # Trade statistics table
        html += '''
        <h3>Trading Statistics</h3>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Total Trades</th>
                    <th>Win Rate</th>
                    <th>Avg Trade P&L</th>
                    <th>Best Trade</th>
                    <th>Worst Trade</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for strategy_name, result in self.backtester.results.items():
            avg_pnl_class = "positive" if result.avg_trade_pnl > 0 else "negative"
            html += f'''
                <tr>
                    <td><strong>{strategy_name}</strong></td>
                    <td>{result.total_trades}</td>
                    <td>{result.win_rate:.2%}</td>
                    <td class="{avg_pnl_class}">${result.avg_trade_pnl:.2f}</td>
                    <td class="positive">${result.largest_win:.2f}</td>
                    <td class="negative">${result.largest_loss:.2f}</td>
                </tr>
            '''
        
        html += '''
            </tbody>
        </table>
        '''
        
        return html
    
    def _create_significance_tests_section(self) -> str:
        """Create statistical significance tests section"""
        
        html = '<h2>Statistical Significance Tests</h2>'
        
        if len(self.backtester.results) < 2:
            html += '<p>Need at least 2 strategies for significance testing.</p>'
            return html
        
        html += '''
        <p>Pairwise statistical tests to determine if performance differences are statistically significant.</p>
        <table>
            <thead>
                <tr>
                    <th>Strategy 1</th>
                    <th>Strategy 2</th>
                    <th>Test</th>
                    <th>P-value</th>
                    <th>Significant (5%)</th>
                    <th>Mean Diff</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        strategies = list(self.backtester.results.keys())
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                try:
                    test_result = self.backtester.statistical_significance_test(
                        strategies[i], strategies[j]
                    )
                    
                    if 'error' not in test_result:
                        significance = "Yes" if test_result['significant_5pct'] else "No"
                        sig_class = "positive" if test_result['significant_5pct'] else ""
                        
                        html += f'''
                            <tr>
                                <td>{strategies[i]}</td>
                                <td>{strategies[j]}</td>
                                <td>Paired t-test</td>
                                <td>{test_result['p_value']:.4f}</td>
                                <td class="{sig_class}">{significance}</td>
                                <td>{test_result['mean_return_diff']:.4f}</td>
                            </tr>
                        '''
                except Exception as e:
                    logger.warning(f"Could not run significance test: {e}")
        
        html += '''
            </tbody>
        </table>
        <p><em>* Significance tested at 5% level. "Yes" indicates statistically significant difference in performance.</em></p>
        '''
        
        return html
    
    def _get_html_footer(self) -> str:
        """Get HTML footer"""
        return '''
        <div class="footer">
            <p>NFL Trading Strategy Backtest Report</p>
            <p>Generated by NFL Trading Backtesting Framework</p>
        </div>
    </div>
</body>
</html>
'''
    
    def _export_data_files(self, output_dir: Path, report_name: str):
        """Export raw data files"""
        
        logger.info("Exporting data files...")
        
        # Export strategy comparison
        comparison_df = self.backtester.compare_strategies()
        comparison_file = output_dir / f"{report_name}_strategy_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        # Export individual strategy data
        for strategy_name, result in self.backtester.results.items():
            safe_name = strategy_name.replace(' ', '_').replace('/', '_')
            
            # Portfolio timeseries
            if not result.portfolio_timeseries.empty:
                portfolio_file = output_dir / f"{report_name}_{safe_name}_portfolio.csv"
                result.portfolio_timeseries.to_csv(portfolio_file, index=False)
            
            # Trade history
            if not result.trade_history.empty:
                trades_file = output_dir / f"{report_name}_{safe_name}_trades.csv"
                result.trade_history.to_csv(trades_file, index=False)
        
        # Export JSON summary
        self.backtester.export_results(str(output_dir / f"{report_name}_results.json"))
        
        logger.info(f"Data files exported to {output_dir}")
    
    def create_strategy_tearsheet(self, strategy_name: str, output_file: str = None) -> str:
        """Create detailed tearsheet for a single strategy"""
        
        if strategy_name not in self.backtester.results:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        result = self.backtester.results[strategy_name]
        
        # Generate detailed analysis
        risk_analysis = self.analyzer.analyze_strategy_risk_decomposition(strategy_name)
        trade_attribution = self.backtester.get_trade_attribution(strategy_name)
        
        html = self._get_html_header()
        
        html += f'''
        <h1>{strategy_name} - Strategy Tearsheet</h1>
        
        <h2>Performance Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Return</h3>
                <div class="value {'positive' if result.total_return > 0 else 'negative'}">{result.total_return:.2%}</div>
            </div>
            <div class="summary-card">
                <h3>Sharpe Ratio</h3>
                <div class="value">{result.sharpe_ratio:.3f}</div>
            </div>
            <div class="summary-card">
                <h3>Max Drawdown</h3>
                <div class="value negative">{result.max_drawdown:.2%}</div>
            </div>
            <div class="summary-card">
                <h3>Win Rate</h3>
                <div class="value">{result.win_rate:.2%}</div>
            </div>
        </div>
        '''
        
        # Add detailed metrics
        html += f'''
        <h2>Detailed Metrics</h2>
        <table>
            <tr><td><strong>Annualized Return</strong></td><td>{result.annualized_return:.2%}</td></tr>
            <tr><td><strong>Volatility</strong></td><td>{result.volatility:.2%}</td></tr>
            <tr><td><strong>Calmar Ratio</strong></td><td>{result.calmar_ratio:.3f}</td></tr>
            <tr><td><strong>Total Trades</strong></td><td>{result.total_trades}</td></tr>
            <tr><td><strong>Profit Factor</strong></td><td>{result.profit_factor:.2f}</td></tr>
            <tr><td><strong>VaR (95%)</strong></td><td>{result.var_95:.2%}</td></tr>
            <tr><td><strong>Skewness</strong></td><td>{result.skewness:.3f}</td></tr>
            <tr><td><strong>Kurtosis</strong></td><td>{result.kurtosis:.3f}</td></tr>
        </table>
        '''
        
        # Add charts if data available
        try:
            timing_chart = self.analyzer.create_trade_timing_analysis(strategy_name)
            chart_html = pio.to_html(timing_chart, include_plotlyjs=False, div_id="timing_analysis")
            html += f'''
            <div class="chart-container">
                <h3>Trade Timing Analysis</h3>
                {chart_html}
            </div>
            '''
        except Exception as e:
            logger.warning(f"Could not create timing chart: {e}")
        
        html += self._get_html_footer()
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Strategy tearsheet saved to {output_file}")
        
        return html
    
    def export_to_csv(self, output_dir: str = "exports") -> Dict[str, str]:
        """Export all results to CSV files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_files = {}
        
        # Strategy comparison
        comparison_df = self.backtester.compare_strategies()
        comparison_file = output_path / "strategy_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        exported_files['comparison'] = str(comparison_file)
        
        # Individual strategy data
        for strategy_name, result in self.backtester.results.items():
            safe_name = strategy_name.replace(' ', '_').replace('/', '_')
            
            if not result.portfolio_timeseries.empty:
                portfolio_file = output_path / f"{safe_name}_portfolio.csv"
                result.portfolio_timeseries.to_csv(portfolio_file, index=False)
                exported_files[f'{safe_name}_portfolio'] = str(portfolio_file)
            
            if not result.trade_history.empty:
                trades_file = output_path / f"{safe_name}_trades.csv"
                result.trade_history.to_csv(trades_file, index=False)
                exported_files[f'{safe_name}_trades'] = str(trades_file)
        
        logger.info(f"CSV files exported to {output_dir}")
        return exported_files
    
    def export_to_json(self, output_file: str = "backtest_results.json") -> str:
        """Export all results to JSON file"""
        
        self.backtester.export_results(output_file)
        logger.info(f"JSON results exported to {output_file}")
        return output_file