"""
Performance Analytics

Comprehensive performance analysis tools for strategy comparison,
risk decomposition, and detailed analytics.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings
import logging

try:
    from .backtester import BacktestResult, Backtester
except ImportError:
    from backtester import BacktestResult, Backtester

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    
    Features:
    - Strategy comparison dashboard
    - Risk decomposition analysis
    - Trade timing analysis
    - Portfolio heat maps and drawdown analysis
    - Rolling performance metrics
    """
    
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        self.results = backtester.results
        
    def create_strategy_comparison_dashboard(self) -> Dict[str, go.Figure]:
        """Create comprehensive strategy comparison dashboard"""
        
        if not self.results:
            logger.warning("No backtest results available")
            return {}
        
        figures = {}
        
        # 1. Performance Summary Table
        figures['performance_table'] = self._create_performance_table()
        
        # 2. Cumulative Returns Chart
        figures['cumulative_returns'] = self._create_cumulative_returns_chart()
        
        # 3. Risk-Return Scatter Plot
        figures['risk_return_scatter'] = self._create_risk_return_scatter()
        
        # 4. Drawdown Analysis
        figures['drawdown_analysis'] = self._create_drawdown_analysis()
        
        # 5. Monthly Returns Heatmap
        figures['monthly_returns'] = self._create_monthly_returns_heatmap()
        
        # 6. Trade Distribution Analysis
        figures['trade_distribution'] = self._create_trade_distribution_analysis()
        
        # 7. Rolling Performance Metrics
        figures['rolling_metrics'] = self._create_rolling_metrics_chart()
        
        return figures
    
    def _create_performance_table(self) -> go.Figure:
        """Create performance metrics comparison table"""
        
        headers = [
            'Strategy', 'Total Return', 'Annualized Return', 'Volatility', 
            'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Win Rate', 
            'Total Trades', 'Profit Factor'
        ]
        
        data = []
        for strategy_name, result in self.results.items():
            row = [
                strategy_name,
                f"{result.total_return:.2%}",
                f"{result.annualized_return:.2%}",
                f"{result.volatility:.2%}",
                f"{result.sharpe_ratio:.3f}",
                f"{result.max_drawdown:.2%}",
                f"{result.calmar_ratio:.3f}",
                f"{result.win_rate:.2%}",
                str(result.total_trades),
                f"{result.profit_factor:.2f}"
            ]
            data.append(row)
        
        # Sort by Sharpe ratio
        if data:
            data.sort(key=lambda x: float(x[4]), reverse=True)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='lightblue',
                align='center',
                font_size=12,
                height=30
            ),
            cells=dict(
                values=list(zip(*data)) if data else [[] for _ in headers],
                fill_color='white',
                align='center',
                font_size=11,
                height=25
            )
        )])
        
        fig.update_layout(
            title="Strategy Performance Comparison",
            height=300 + len(data) * 25
        )
        
        return fig
    
    def _create_cumulative_returns_chart(self) -> go.Figure:
        """Create cumulative returns comparison chart"""
        
        fig = go.Figure()
        
        for strategy_name, result in self.results.items():
            if not result.portfolio_timeseries.empty:
                df = result.portfolio_timeseries.copy()
                
                # Calculate cumulative returns
                if 'cumulative_returns' not in df.columns:
                    df['cumulative_returns'] = (df['portfolio_value'] / df['portfolio_value'].iloc[0]) - 1
                
                fig.add_trace(go.Scatter(
                    x=df['datetime'] if 'datetime' in df.columns else df.index,
                    y=df['cumulative_returns'] * 100,  # Convert to percentage
                    mode='lines',
                    name=strategy_name,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Time",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def _create_risk_return_scatter(self) -> go.Figure:
        """Create risk-return scatter plot"""
        
        returns = []
        volatilities = []
        sharpe_ratios = []
        names = []
        
        for strategy_name, result in self.results.items():
            returns.append(result.annualized_return * 100)
            volatilities.append(result.volatility * 100)
            sharpe_ratios.append(result.sharpe_ratio)
            names.append(strategy_name)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=names,
            textposition="top center",
            marker=dict(
                size=[(sr + 1) * 10 for sr in sharpe_ratios],  # Size based on Sharpe ratio
                color=sharpe_ratios,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                line=dict(width=2, color='black')
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Volatility: %{x:.2f}%<br>"
                "Return: %{y:.2f}%<br>"
                "Sharpe: %{marker.color:.3f}<br>"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Volatility (%)",
            yaxis_title="Annualized Return (%)",
            height=500
        )
        
        return fig
    
    def _create_drawdown_analysis(self) -> go.Figure:
        """Create drawdown analysis chart"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Values', 'Drawdowns'),
            vertical_spacing=0.1
        )
        
        for strategy_name, result in self.results.items():
            if not result.portfolio_timeseries.empty:
                df = result.portfolio_timeseries.copy()
                
                # Calculate drawdowns
                portfolio_values = df['portfolio_value']
                running_max = portfolio_values.expanding().max()
                drawdowns = (portfolio_values - running_max) / running_max * 100
                
                x_axis = df['datetime'] if 'datetime' in df.columns else df.index
                
                # Portfolio values
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=portfolio_values,
                        mode='lines',
                        name=f"{strategy_name} Portfolio",
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
                
                # Drawdowns
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=drawdowns,
                        mode='lines',
                        name=f"{strategy_name} Drawdown",
                        fill='tonexty' if strategy_name == list(self.results.keys())[0] else None,
                        line=dict(width=1)
                    ),
                    row=2, col=1
                )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        fig.update_layout(
            title="Portfolio Values and Drawdown Analysis",
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_monthly_returns_heatmap(self) -> go.Figure:
        """Create monthly returns heatmap for each strategy"""
        
        # This is a simplified version since we have game-level data
        # In practice, you'd aggregate by actual months
        
        fig = make_subplots(
            rows=len(self.results), cols=1,
            subplot_titles=[f"{name} Game Returns" for name in self.results.keys()],
            vertical_spacing=0.1
        )
        
        for i, (strategy_name, result) in enumerate(self.results.items(), 1):
            if not result.portfolio_timeseries.empty and 'game_id' in result.portfolio_timeseries.columns:
                # Calculate per-game returns
                game_returns = result.portfolio_timeseries.groupby('game_id')['returns'].sum() * 100
                
                # Create a simple heatmap-like visualization
                fig.add_trace(
                    go.Bar(
                        x=[f"Game {gid}" for gid in game_returns.index],
                        y=game_returns.values,
                        name=f"{strategy_name} Returns",
                        marker_color=game_returns.values,
                        colorscale='RdYlGn',
                        showlegend=False
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title="Game-by-Game Returns Analysis",
            height=200 * len(self.results),
            showlegend=False
        )
        
        return fig
    
    def _create_trade_distribution_analysis(self) -> go.Figure:
        """Create trade P&L distribution analysis"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('P&L Distribution', 'Trade Size vs P&L'),
            column_widths=[0.6, 0.4]
        )
        
        for strategy_name, result in self.results.items():
            if not result.trade_history.empty:
                pnls = result.trade_history['pnl']
                sizes = result.trade_history['size']
                
                # P&L histogram
                fig.add_trace(
                    go.Histogram(
                        x=pnls,
                        name=f"{strategy_name} P&L",
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=1, col=1
                )
                
                # Trade size vs P&L scatter
                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=pnls,
                        mode='markers',
                        name=f"{strategy_name} Trades",
                        opacity=0.6,
                        marker=dict(size=6)
                    ),
                    row=1, col=2
                )
        
        fig.update_xaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Trade Size", row=1, col=2)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=2)
        
        fig.update_layout(
            title="Trade Distribution Analysis",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _create_rolling_metrics_chart(self) -> go.Figure:
        """Create rolling performance metrics chart"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility', 
                          'Rolling Win Rate', 'Rolling Max Drawdown'),
            vertical_spacing=0.15
        )
        
        for strategy_name, result in self.results.items():
            if not result.portfolio_timeseries.empty and len(result.portfolio_timeseries) > 20:
                df = result.portfolio_timeseries.copy()
                returns = df['returns'].dropna()
                
                if len(returns) > 10:
                    # Rolling metrics (using 10-period windows)
                    window = min(10, len(returns) // 3)
                    
                    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std()
                    rolling_vol = returns.rolling(window).std()
                    
                    # Rolling win rate (simplified)
                    rolling_win_rate = (returns > 0).rolling(window).mean() * 100
                    
                    # Rolling drawdown
                    portfolio_values = df['portfolio_value']
                    rolling_max = portfolio_values.rolling(window).max()
                    rolling_dd = (portfolio_values - rolling_max) / rolling_max * 100
                    
                    x_axis = df['datetime'].iloc[window-1:] if 'datetime' in df.columns else df.index[window-1:]
                    
                    # Rolling Sharpe
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=rolling_sharpe.iloc[window-1:],
                            mode='lines',
                            name=f"{strategy_name}",
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                    
                    # Rolling volatility
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=rolling_vol.iloc[window-1:] * 100,
                            mode='lines',
                            name=f"{strategy_name}",
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                    
                    # Rolling win rate
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=rolling_win_rate.iloc[window-1:],
                            mode='lines',
                            name=f"{strategy_name}",
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    # Rolling max drawdown
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=rolling_dd.iloc[window-1:],
                            mode='lines',
                            name=f"{strategy_name}",
                            showlegend=False
                        ),
                        row=2, col=2
                    )
        
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)
        
        fig.update_layout(
            title="Rolling Performance Metrics",
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def analyze_strategy_risk_decomposition(self, strategy_name: str) -> Dict[str, Any]:
        """Perform detailed risk decomposition for a strategy"""
        
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        result = self.results[strategy_name]
        
        if result.portfolio_timeseries.empty:
            return {'error': 'No portfolio data available'}
        
        returns = result.portfolio_timeseries['returns'].dropna()
        
        if len(returns) == 0:
            return {'error': 'No valid returns data'}
        
        # Basic risk metrics
        risk_metrics = {
            'volatility': returns.std(),
            'downside_volatility': returns[returns < 0].std() if (returns < 0).any() else 0,
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'max_daily_loss': returns.min(),
            'max_daily_gain': returns.max()
        }
        
        # CVaR calculation
        var_95 = risk_metrics['var_95']
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95
        risk_metrics['cvar_95'] = cvar_95
        
        # Upside/downside capture
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        risk_metrics.update({
            'upside_capture': len(positive_returns) / len(returns) if len(returns) > 0 else 0,
            'downside_capture': len(negative_returns) / len(returns) if len(returns) > 0 else 0,
            'avg_positive_return': positive_returns.mean() if len(positive_returns) > 0 else 0,
            'avg_negative_return': negative_returns.mean() if len(negative_returns) > 0 else 0
        })
        
        # Tail risk analysis
        extreme_losses = returns[returns <= np.percentile(returns, 5)]
        extreme_gains = returns[returns >= np.percentile(returns, 95)]
        
        risk_metrics.update({
            'tail_ratio': abs(extreme_gains.mean()) / abs(extreme_losses.mean()) if len(extreme_losses) > 0 and extreme_losses.mean() != 0 else float('inf'),
            'extreme_loss_frequency': len(extreme_losses) / len(returns) if len(returns) > 0 else 0,
            'extreme_gain_frequency': len(extreme_gains) / len(returns) if len(returns) > 0 else 0
        })
        
        return {
            'strategy': strategy_name,
            'risk_metrics': risk_metrics,
            'total_observations': len(returns)
        }
    
    def create_trade_timing_analysis(self, strategy_name: str) -> go.Figure:
        """Analyze trade timing patterns"""
        
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        result = self.results[strategy_name]
        
        if result.trade_history.empty:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        trades = result.trade_history.copy()
        
        # Convert timestamp to datetime if needed
        if 'datetime' not in trades.columns:
            trades['datetime'] = pd.to_datetime(trades['timestamp'], unit='s')
        
        trades['hour'] = trades['datetime'].dt.hour
        trades['minute'] = trades['datetime'].dt.minute
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('P&L by Hour', 'Trade Count by Hour', 
                          'Cumulative P&L Timeline', 'P&L by Trade Size'),
            vertical_spacing=0.15
        )
        
        # P&L by hour
        hourly_pnl = trades.groupby('hour')['pnl'].sum()
        fig.add_trace(
            go.Bar(x=hourly_pnl.index, y=hourly_pnl.values, name='Hourly P&L'),
            row=1, col=1
        )
        
        # Trade count by hour
        hourly_count = trades.groupby('hour').size()
        fig.add_trace(
            go.Bar(x=hourly_count.index, y=hourly_count.values, name='Trade Count'),
            row=1, col=2
        )
        
        # Cumulative P&L timeline
        trades_sorted = trades.sort_values('datetime')
        cumulative_pnl = trades_sorted['pnl'].cumsum()
        fig.add_trace(
            go.Scatter(x=trades_sorted['datetime'], y=cumulative_pnl, 
                      mode='lines', name='Cumulative P&L'),
            row=2, col=1
        )
        
        # P&L by trade size
        fig.add_trace(
            go.Scatter(x=trades['size'], y=trades['pnl'], mode='markers',
                      name='P&L vs Size', opacity=0.6),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Trade Timing Analysis - {strategy_name}",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_matrix(self) -> go.Figure:
        """Create correlation matrix between strategies"""
        
        if len(self.results) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 strategies for correlation analysis", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Align returns data
        returns_data = {}
        
        for strategy_name, result in self.results.items():
            if not result.portfolio_timeseries.empty:
                returns = result.portfolio_timeseries['returns'].dropna()
                if len(returns) > 0:
                    returns_data[strategy_name] = returns
        
        if len(returns_data) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient returns data for correlation analysis", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Align data by index (assuming common time index)
        df_returns = pd.DataFrame(returns_data)
        correlation_matrix = df_returns.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Strategy Returns Correlation Matrix",
            height=500
        )
        
        return fig
    
    def generate_performance_report(self, output_file: str = None) -> str:
        """Generate a comprehensive text-based performance report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE STRATEGY PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of strategies analyzed: {len(self.results)}")
        report_lines.append("")
        
        if not self.results:
            report_lines.append("No backtest results available.")
            report = "\n".join(report_lines)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report)
            return report
        
        # Strategy rankings
        comparison_df = self.backtester.compare_strategies('sharpe_ratio')
        
        report_lines.append("STRATEGY RANKINGS (by Sharpe Ratio)")
        report_lines.append("-" * 50)
        for i, row in comparison_df.iterrows():
            report_lines.append(f"{i+1:2d}. {row['strategy']:20s} | Sharpe: {row['sharpe_ratio']:6.3f}")
        report_lines.append("")
        
        # Detailed metrics for each strategy
        for strategy_name, result in self.results.items():
            report_lines.append(f"STRATEGY: {strategy_name}")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Return:      {result.total_return:8.2%}")
            report_lines.append(f"Annualized Return: {result.annualized_return:8.2%}")
            report_lines.append(f"Volatility:        {result.volatility:8.2%}")
            report_lines.append(f"Sharpe Ratio:      {result.sharpe_ratio:8.3f}")
            report_lines.append(f"Max Drawdown:      {result.max_drawdown:8.2%}")
            report_lines.append(f"Calmar Ratio:      {result.calmar_ratio:8.3f}")
            report_lines.append(f"Win Rate:          {result.win_rate:8.2%}")
            report_lines.append(f"Total Trades:      {result.total_trades:8d}")
            report_lines.append(f"Profit Factor:     {result.profit_factor:8.2f}")
            
            if result.total_trades > 0:
                report_lines.append(f"Avg Trade P&L:     ${result.avg_trade_pnl:7.2f}")
                report_lines.append(f"Largest Win:       ${result.largest_win:7.2f}")
                report_lines.append(f"Largest Loss:      ${result.largest_loss:7.2f}")
            
            report_lines.append("")
        
        # Statistical significance tests (if multiple strategies)
        if len(self.results) > 1:
            report_lines.append("STATISTICAL SIGNIFICANCE TESTS")
            report_lines.append("-" * 40)
            
            strategies = list(self.results.keys())
            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    try:
                        test_result = self.backtester.statistical_significance_test(
                            strategies[i], strategies[j]
                        )
                        
                        if 'error' not in test_result:
                            significance = "***" if test_result['significant_1pct'] else ("**" if test_result['significant_5pct'] else "")
                            report_lines.append(
                                f"{strategies[i]:15s} vs {strategies[j]:15s} | "
                                f"p-value: {test_result['p_value']:.4f} {significance}"
                            )
                    except Exception as e:
                        logger.warning(f"Could not run significance test: {e}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Performance report saved to {output_file}")
        
        return report