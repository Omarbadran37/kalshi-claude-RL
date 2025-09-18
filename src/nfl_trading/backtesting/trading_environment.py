"""
Trading Environment Simulator

Simulates realistic trading conditions for backtesting NFL trading strategies.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    ticker: str
    side: OrderSide
    size: int
    order_type: OrderType
    limit_price: Optional[float] = None
    timestamp: float = 0.0
    filled_price: Optional[float] = None
    filled_size: int = 0
    status: str = "pending"  # pending, filled, partial, rejected


@dataclass
class Position:
    """Represents a trading position"""
    ticker: str
    size: int = 0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_volume: int = 0


@dataclass
class TradeRecord:
    """Records executed trades"""
    timestamp: float
    ticker: str
    side: OrderSide
    size: int
    price: float
    fee: float
    pnl: float = 0.0


@dataclass
class MarketState:
    """Current market state at a given timestamp"""
    timestamp: float
    ticker: str
    bid_price: float
    ask_price: float
    mid_price: float
    volume: int
    open_interest: int
    spread: float = field(init=False)
    
    def __post_init__(self):
        self.spread = self.ask_price - self.bid_price


class TradingEnvironment:
    """
    Simulates realistic trading environment for backtesting.
    
    Features:
    - Historical game replay with 1-minute timesteps
    - Realistic order execution with bid-ask spreads
    - Kalshi-like fee structure
    - Slippage modeling
    - Position and PnL tracking
    - Risk metrics calculation
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.07,  # 7% fee on profits (Kalshi-like)
        slippage_bps: float = 0.5,  # 0.5 basis points slippage
        max_position_size: int = 1000,  # Max contracts per position
        min_tick_size: float = 0.01  # Minimum price increment
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        self.max_position_size = max_position_size
        self.min_tick_size = min_tick_size
        
        # State tracking
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_timestamp = 0.0
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[TradeRecord] = []
        self.market_data: Dict[str, List[MarketState]] = {}
        self.data_index: Dict[str, int] = {}
        
        # Performance tracking
        self.portfolio_values: List[Tuple[float, float]] = [(0.0, self.initial_capital)]
        self.drawdowns: List[float] = []
        self.max_portfolio_value = self.initial_capital
        
    def load_market_data(self, data_file: str):
        """Load historical market data from JSON file"""
        logger.info(f"Loading market data from {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Extract candlestick data for each team
        for team, team_data in data['team_markets'].items():
            ticker = team_data['ticker']
            candlesticks = team_data['candlesticks']['candlesticks']
            
            market_states = []
            for candle in candlesticks:
                # Calculate mid price from bid/ask
                bid_price = candle['yes_bid']['close'] / 100.0  # Convert cents to dollars
                ask_price = candle['yes_ask']['close'] / 100.0
                mid_price = candle['price']['close'] / 100.0
                
                market_state = MarketState(
                    timestamp=candle['end_period_ts'],
                    ticker=ticker,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    mid_price=mid_price,
                    volume=candle['volume'],
                    open_interest=candle['open_interest']
                )
                market_states.append(market_state)
            
            self.market_data[ticker] = sorted(market_states, key=lambda x: x.timestamp)
            self.data_index[ticker] = 0
            
        logger.info(f"Loaded data for {len(self.market_data)} tickers")
        
    def get_current_market_state(self, ticker: str) -> Optional[MarketState]:
        """Get current market state for a ticker"""
        if ticker not in self.market_data:
            return None
            
        data_list = self.market_data[ticker]
        index = self.data_index[ticker]
        
        if index < len(data_list):
            return data_list[index]
        return None
    
    def advance_time(self, target_timestamp: float):
        """Advance simulation to target timestamp"""
        self.current_timestamp = target_timestamp
        
        # Update data indices to current timestamp
        for ticker in self.market_data:
            data_list = self.market_data[ticker]
            current_index = self.data_index[ticker]
            
            # Find the latest data point <= target_timestamp
            while (current_index < len(data_list) - 1 and 
                   data_list[current_index + 1].timestamp <= target_timestamp):
                current_index += 1
            
            self.data_index[ticker] = current_index
            
        # Process pending orders
        self._process_orders()
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
    
    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        size: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None
    ) -> str:
        """Place a trading order"""
        
        # Validate order
        if size <= 0:
            raise ValueError("Order size must be positive")
            
        if size > self.max_position_size:
            raise ValueError(f"Order size exceeds maximum position size: {self.max_position_size}")
        
        # Generate order ID
        order_id = f"order_{len(self.orders)}_{int(self.current_timestamp)}"
        
        order = Order(
            order_id=order_id,
            ticker=ticker,
            side=side,
            size=size,
            order_type=order_type,
            limit_price=limit_price,
            timestamp=self.current_timestamp
        )
        
        self.orders.append(order)
        logger.debug(f"Placed {order_type.value} {side.value} order: {size} {ticker} @ {limit_price}")
        
        return order_id
    
    def _process_orders(self):
        """Process pending orders against current market data"""
        for order in self.orders:
            if order.status != "pending":
                continue
                
            market_state = self.get_current_market_state(order.ticker)
            if not market_state:
                continue
            
            # Check if order can be filled
            fill_price = self._get_fill_price(order, market_state)
            if fill_price is not None:
                self._execute_order(order, fill_price, market_state.timestamp)
    
    def _get_fill_price(self, order: Order, market_state: MarketState) -> Optional[float]:
        """Determine if order can be filled and at what price"""
        
        if order.order_type == OrderType.MARKET:
            # Market orders fill at bid/ask with slippage
            if order.side == OrderSide.BUY:
                base_price = market_state.ask_price
            else:
                base_price = market_state.bid_price
                
            # Apply slippage
            slippage = base_price * self.slippage_bps
            if order.side == OrderSide.BUY:
                fill_price = base_price + slippage
            else:
                fill_price = base_price - slippage
                
            # Round to minimum tick size
            fill_price = round(fill_price / self.min_tick_size) * self.min_tick_size
            fill_price = max(0.01, min(0.99, fill_price))  # Clamp to valid range
            
            return fill_price
            
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill when market price crosses limit
            if order.side == OrderSide.BUY and market_state.ask_price <= order.limit_price:
                return min(order.limit_price, market_state.ask_price)
            elif order.side == OrderSide.SELL and market_state.bid_price >= order.limit_price:
                return max(order.limit_price, market_state.bid_price)
                
        return None
    
    def _execute_order(self, order: Order, fill_price: float, timestamp: float):
        """Execute an order at the given price"""
        
        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            required_cash = order.size * fill_price
            if required_cash > self.cash:
                order.status = "rejected"
                logger.warning(f"Order rejected: insufficient cash. Required: {required_cash}, Available: {self.cash}")
                return
        
        # Calculate fees (Kalshi charges on profits, simplified here as percentage of notional)
        notional = order.size * fill_price
        fee = notional * self.fee_rate
        
        # Update position
        if order.ticker not in self.positions:
            self.positions[order.ticker] = Position(ticker=order.ticker)
            
        position = self.positions[order.ticker]
        
        # Calculate PnL for position changes
        pnl = 0.0
        if order.side == OrderSide.BUY:
            # Update average entry price
            total_cost = position.size * position.avg_entry_price + order.size * fill_price
            total_size = position.size + order.size
            if total_size > 0:
                position.avg_entry_price = total_cost / total_size
            position.size += order.size
            self.cash -= notional + fee
        else:  # SELL
            if position.size >= order.size:
                # Calculate realized PnL
                pnl = order.size * (fill_price - position.avg_entry_price)
                position.realized_pnl += pnl
                position.size -= order.size
                self.cash += notional - fee
            else:
                order.status = "rejected"
                logger.warning(f"Order rejected: insufficient position size. Required: {order.size}, Available: {position.size}")
                return
        
        # Record trade
        trade_record = TradeRecord(
            timestamp=timestamp,
            ticker=order.ticker,
            side=order.side,
            size=order.size,
            price=fill_price,
            fee=fee,
            pnl=pnl
        )
        self.trade_history.append(trade_record)
        
        # Update order status
        order.status = "filled"
        order.filled_price = fill_price
        order.filled_size = order.size
        
        position.total_volume += order.size
        
        logger.debug(f"Executed order: {order.side.value} {order.size} {order.ticker} @ {fill_price:.4f}")
    
    def _update_portfolio_metrics(self):
        """Update portfolio value and risk metrics"""
        # Calculate current portfolio value
        portfolio_value = self.cash
        
        for ticker, position in self.positions.items():
            if position.size > 0:
                market_state = self.get_current_market_state(ticker)
                if market_state:
                    # Use mid price for portfolio valuation
                    market_value = position.size * market_state.mid_price
                    portfolio_value += market_value
                    
                    # Update unrealized PnL
                    position.unrealized_pnl = position.size * (market_state.mid_price - position.avg_entry_price)
        
        self.portfolio_values.append((self.current_timestamp, portfolio_value))
        
        # Calculate drawdown
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
            
        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        self.drawdowns.append(drawdown)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_volume = sum(pos.total_volume for pos in self.positions.values())
        
        current_portfolio_value = self.portfolio_values[-1][1] if self.portfolio_values else self.initial_capital
        total_return = (current_portfolio_value - self.initial_capital) / self.initial_capital
        
        return {
            'timestamp': self.current_timestamp,
            'cash': self.cash,
            'portfolio_value': current_portfolio_value,
            'total_return': total_return,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'total_volume': total_volume,
            'num_trades': len(self.trade_history),
            'num_positions': len([pos for pos in self.positions.values() if pos.size > 0]),
            'max_drawdown': max(self.drawdowns) if self.drawdowns else 0.0
        }
    
    def get_position_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all positions"""
        summary = {}
        for ticker, position in self.positions.items():
            if position.size > 0:
                market_state = self.get_current_market_state(ticker)
                current_price = market_state.mid_price if market_state else position.avg_entry_price
                
                summary[ticker] = {
                    'size': position.size,
                    'avg_entry_price': position.avg_entry_price,
                    'current_price': current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'total_volume': position.total_volume
                }
        return summary
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
            
        data = []
        for trade in self.trade_history:
            data.append({
                'timestamp': trade.timestamp,
                'datetime': datetime.fromtimestamp(trade.timestamp),
                'ticker': trade.ticker,
                'side': trade.side.value,
                'size': trade.size,
                'price': trade.price,
                'fee': trade.fee,
                'pnl': trade.pnl
            })
        
        return pd.DataFrame(data)
    
    def get_portfolio_timeseries(self) -> pd.DataFrame:
        """Get portfolio value time series as DataFrame"""
        if not self.portfolio_values:
            return pd.DataFrame()
            
        data = []
        for timestamp, value in self.portfolio_values:
            data.append({
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now(),
                'portfolio_value': value
            })
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            df['returns'] = df['portfolio_value'].pct_change()
            df['cumulative_returns'] = (df['portfolio_value'] / self.initial_capital) - 1
        
        return df