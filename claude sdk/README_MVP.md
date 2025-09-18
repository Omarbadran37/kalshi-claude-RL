# Kalshi NFL Trading System MVP

A simple 2-agent system using Claude Code SDK architecture for NFL game analysis and trading recommendations.

## Architecture

### Sports Agent
- Analyzes NFL game dynamics, team performance, and momentum
- Considers current score, time remaining, recent events
- Outputs win probabilities and confidence assessment
- Uses Claude Code SDK for sophisticated sports analysis

### Trade Agent
- Receives sports analysis as input
- Analyzes market data from Kalshi API
- Makes final BUY/SELL/PASS recommendation
- Combines sports insights with market inefficiency detection

## Key Features

- **Modular Design**: Clear separation between sports and trading analysis
- **Real API Integration**: Built for Kalshi API (with mock fallback for testing)
- **Structured Communication**: Clean data structures between agents
- **Error Handling**: Robust error handling with fallback mechanisms
- **Extensible**: Easy to add more agents or enhance existing ones

## Installation

```bash
pip install -r requirements_mvp.txt
```

## Usage

### Basic Usage

```python
from kalshi_nfl_mvp import KalshiNFLTradingSystem, GameInput

# Create game input
game_input = GameInput(
    team_a="Buffalo Bills",
    team_b="Miami Dolphins",
    current_score={"team_a": 14, "team_b": 7},
    quarter=3,
    time_remaining="8:45",
    recent_events=["Bills touchdown", "Dolphins turnover"]
)

# Initialize system
system = KalshiNFLTradingSystem(kalshi_api_key="your_api_key")

# Run analysis
result = await system.analyze_game(game_input)
```

### Testing

```bash
# Test with requirements example
python test_mvp.py

# Choose option 1 for single test, option 2 for multiple scenarios
```

## Data Structures

### GameInput
```python
@dataclass
class GameInput:
    team_a: str
    team_b: str
    current_score: Dict[str, int]
    quarter: int
    time_remaining: str
    recent_events: List[str]
```

### SportsAnalysis
```python
@dataclass
class SportsAnalysis:
    predicted_winner: str
    win_probability_team_a: float
    win_probability_team_b: float
    confidence: float
    key_factors: List[str]
    momentum_assessment: str
    reasoning: str
```

### TradingRecommendation
```python
@dataclass
class TradingRecommendation:
    action: str  # "BUY", "SELL", "PASS"
    target: str  # "team_a_yes", "team_b_yes", etc.
    quantity: int
    confidence: float
    reasoning: str
    sports_signal_strength: float
    market_edge_detected: bool
    risk_assessment: str
```

## Example Output

```json
{
  "sports_analysis": {
    "predicted_winner": "team_a",
    "win_probability_team_a": 0.65,
    "win_probability_team_b": 0.35,
    "confidence": 0.75,
    "key_factors": ["momentum_shift", "score_differential", "time_remaining"],
    "momentum_assessment": "strong_team_a"
  },
  "trading_recommendation": {
    "action": "BUY",
    "target": "team_a_yes",
    "quantity": 150,
    "confidence": 0.68,
    "market_edge_detected": true,
    "risk_assessment": "medium"
  }
}
```

## File Structure

```
/claude sdk/
├── kalshi_nfl_mvp.py      # Main system implementation
├── test_mvp.py            # Test script with scenarios
├── requirements_mvp.txt   # Dependencies
└── README_MVP.md         # This file
```

## MVP Limitations

1. **Mock Data**: Uses mock market data when Kalshi API key not provided
2. **Basic Error Handling**: Simple fallback mechanisms
3. **No Persistence**: Results not stored in database
4. **Limited Market Analysis**: Basic market efficiency detection
5. **No Position Management**: Single trade recommendations only

## Extension Points

1. **Additional Agents**: Risk management, portfolio optimization
2. **Real-time Data**: Live game feeds, streaming market data
3. **Position Tracking**: Portfolio management and tracking
4. **Advanced Analytics**: More sophisticated market analysis
5. **Backtesting**: Historical performance evaluation

## Requirements Met

✅ 2-agent system (Sports + Trade)  
✅ Claude Code SDK integration  
✅ Modular architecture  
✅ Real Kalshi API integration  
✅ Structured agent communication  
✅ Clean data structures  
✅ Error handling  
✅ End-to-end workflow  

## Next Steps

1. Add real Kalshi API key for live testing
2. Implement position sizing optimization
3. Add backtesting framework
4. Create web interface
5. Add real-time game data feeds