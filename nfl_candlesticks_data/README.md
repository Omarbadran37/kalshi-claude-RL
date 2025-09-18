# NFL Candlesticks Data Archive

## Overview

This folder contains candlestick data for NFL games that have been played so far in the 2025 season. The data was fetched using the NFL Candlesticks MCP tool on **September 16, 2025**.

## Summary

- **Total Games Available**: 97 NFL games with market tickers
- **Total Market Tickers**: 194 (2 per game - one for each team)
- **Games Matched & Processed**: 17 games that have been played
- **Date Filter**: Past games only (games that have already occurred)
- **Candlestick Period**: 1-minute intervals
- **Time Window**: 1 hour before kickoff to 6 hours after kickoff (7-hour total window)

## Data Structure

Each JSON file contains:

```json
{
  "event_ticker": "KXNFLGAME-25SEP04DALPHI",
  "game_info": {
    "home_team": "Philadelphia Eagles",
    "away_team": "Dallas Cowboys", 
    "date": "2025-09-04",
    "time": "8:20 PM",
    "week": "Week 1"
  },
  "time_window": {
    "start_ts": 1757028000,
    "end_ts": 1757053200,
    "start_time": "2025-09-04 19:20:00 UTC",
    "end_time": "2025-09-05 02:20:00 UTC"
  },
  "team_markets": {
    "DAL": {
      "ticker": "KXNFLGAME-25SEP04DALPHI-DAL",
      "candlesticks": {
        "candlesticks": [
          {
            "end_period_ts": 1757028000,
            "yes_bid": { "open": 21, "low": 20, "high": 21, "close": 21 },
            "yes_ask": { "open": 22, "low": 22, "high": 22, "close": 22 },
            "price": { "open": 22, "low": 21, "high": 22, "close": 22 },
            "volume": 0
          }
          // ... more candlesticks
        ]
      }
    },
    "BAL": { /* Similar structure for Baltimore market */ }
  }
}
```

## Files Description

### Games Played (17 games)

1. **2025-09-04**: Dallas Cowboys at Philadelphia Eagles (Week 1) - 311 candlesticks per team
2. **2025-09-07**: Multiple Week 1 games (8 games) - ~240-270 candlesticks per team
3. **2025-09-08**: Minnesota Vikings at Chicago Bears (Week 1) - 261 candlesticks per team  
4. **2025-09-14**: Week 2 games (3 games) - ~240-252 candlesticks per team
5. **2025-09-15**: Week 2 games (2 games) - ~242-246 candlesticks per team

### Candlestick Data Volume

- **Most Data**: Dallas vs Philadelphia (733,695 bytes, 311 candlesticks per team)
- **Typical Range**: 570KB-640KB per game file
- **Candlesticks per Team**: 224-311 data points (1-minute intervals)
- **Total Archive Size**: ~10MB

## API Calls Made

For each game, two API calls were made:
```
GET /series/KXNFLGAME/markets/{TICKER}-{TEAM}/candlesticks
?start_ts={GAME_START-1hr}&end_ts={GAME_START+6hr}&period_interval=1
```

Example:
- `KXNFLGAME-25SEP04DALPHI-DAL` (Dallas market)
- `KXNFLGAME-25SEP04DALPHI-PHI` (Philadelphia market)

## Data Quality

✅ **Successfully Retrieved**: All 17 matched games have complete candlestick data  
✅ **Both Team Markets**: Each game includes data for both team prediction markets  
✅ **Proper Time Windows**: 7-hour windows (1hr before to 6hr after kickoff)  
✅ **Rich Data**: OHLC prices, bid/ask spreads, volume, timestamps  

## Usage

This archive provides comprehensive market data for:
- Market analysis and visualization
- Trading strategy backtesting  
- Price movement analysis around game times
- Comparison of market behavior across different games
- Research into prediction market dynamics

## Notes

- All times are in UTC
- Prices are in cents (divide by 100 for dollar amounts)
- Volume data included for each candlestick period
- Some warnings in logs indicate ticker parsing issues for certain games (primarily preseason)
- Only regular season games that have been played are included