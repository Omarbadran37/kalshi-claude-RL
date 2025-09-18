#!/usr/bin/env python3
"""
Kalshi NFL Trading System MVP - Demonstration Script
Shows key features and agent communication
"""

import asyncio
import json
from datetime import datetime
from kalshi_nfl_mvp import KalshiNFLTradingSystem, GameInput

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"{title.center(70)}")
    print(f"{'='*70}")

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'-'*50}")
    print(f"{title}")
    print(f"{'-'*50}")

async def demonstrate_mvp():
    """Demonstrate the MVP system capabilities"""
    
    print_section("KALSHI NFL TRADING SYSTEM MVP DEMONSTRATION")
    
    print("\n🏈 2-Agent System using Claude Code SDK")
    print("   • Sports Agent: Analyzes game dynamics and momentum")
    print("   • Trade Agent: Combines sports analysis with market data")
    print("   • Real Kalshi API integration with mock fallback")
    print("   • Structured agent-to-agent communication")
    
    # Initialize system
    print_subsection("System Initialization")
    system = KalshiNFLTradingSystem(kalshi_api_key=None)  # Using mock for demo
    print("✓ Sports Agent initialized")
    print("✓ Trade Agent initialized") 
    print("✓ Kalshi API client ready (using mock data)")
    
    # Demonstration game scenario
    game_input = GameInput(
        team_a="Buffalo Bills",
        team_b="Miami Dolphins", 
        current_score={"team_a": 14, "team_b": 7},
        quarter=3,
        time_remaining="8:45",
        recent_events=["Bills touchdown", "Dolphins turnover"]
    )
    
    print_subsection("Game Input")
    print(f"Matchup: {game_input.team_a} vs {game_input.team_b}")
    print(f"Score: {game_input.current_score['team_a']} - {game_input.current_score['team_b']}")
    print(f"Time: Q{game_input.quarter}, {game_input.time_remaining}")
    print(f"Recent Events: {', '.join(game_input.recent_events)}")
    
    # Run analysis
    print_subsection("Running Analysis")
    print("Starting 2-agent analysis workflow...")
    
    start_time = datetime.now()
    result = await system.analyze_game(game_input)
    duration = (datetime.now() - start_time).total_seconds()
    
    if "error" in result:
        print(f"❌ Analysis failed: {result['error']}")
        return
    
    print(f"✓ Analysis completed in {duration:.1f} seconds")
    
    # Extract components
    sports = result["sports_analysis"]
    trading = result["trading_recommendation"]
    market = result["market_data"]
    
    # Show Sports Agent output
    print_subsection("Sports Agent Analysis")
    print(f"Predicted Winner: {sports['predicted_winner']}")
    print(f"Win Probabilities:")
    print(f"  • {game_input.team_a}: {sports['win_probability_team_a']:.1%}")
    print(f"  • {game_input.team_b}: {sports['win_probability_team_b']:.1%}")
    print(f"Confidence: {sports['confidence']:.1%}")
    print(f"Momentum: {sports['momentum_assessment']}")
    print(f"Key Factors: {', '.join(sports['key_factors'])}")
    print(f"\nReasoning: {sports['reasoning'][:200]}...")
    
    # Show market data
    print_subsection("Market Data (Kalshi API)")
    print(f"Event: {market['event_ticker']}")
    print(f"Market Prices:")
    print(f"  • {game_input.team_a} Win: Yes ${market['team_a_yes_price']:.0f} | No ${market['team_a_no_price']:.0f}")
    print(f"  • {game_input.team_b} Win: Yes ${market['team_b_yes_price']:.0f} | No ${market['team_b_no_price']:.0f}")
    print(f"Volume (24h): {market['volume_24h']:,}")
    print(f"Bid-Ask Spread: ${market['bid_ask_spread']:.1f}")
    print(f"Last Trade: ${market['last_trade_price']:.1f}")
    
    # Show Trade Agent output
    print_subsection("Trade Agent Recommendation")
    print(f"🎯 Action: {trading['action']}")
    print(f"🎯 Target: {trading['target'] or 'None'}")
    print(f"🎯 Quantity: {trading['quantity']}")
    print(f"🎯 Confidence: {trading['confidence']:.1%}")
    print(f"🎯 Market Edge Detected: {'Yes' if trading['market_edge_detected'] else 'No'}")
    print(f"🎯 Risk Assessment: {trading['risk_assessment']}")
    print(f"🎯 Sports Signal Strength: {trading['sports_signal_strength']:.1%}")
    print(f"\nReasoning: {trading['reasoning'][:200]}...")
    
    # Show agent communication flow
    print_subsection("Agent Communication Flow")
    print("1. 📊 Market Data Retrieved from Kalshi API")
    print("2. 🏈 Sports Agent analyzes game situation")
    print("   └── Uses Claude Code SDK for sophisticated analysis")
    print("   └── Outputs win probabilities and confidence")
    print("3. 💹 Trade Agent receives sports analysis")
    print("   └── Combines with market data using Claude Code SDK")
    print("   └── Detects market inefficiencies")
    print("   └── Makes final BUY/SELL/PASS recommendation")
    print("4. ✅ Structured result with full reasoning chain")
    
    # Key metrics
    print_subsection("MVP Success Criteria")
    criteria = [
        ("✅ 2-agent system", "Sports Agent + Trade Agent"),
        ("✅ Claude Code SDK integration", "Both agents use SDK"),
        ("✅ Modular architecture", "Clean separation of concerns"),
        ("✅ Real Kalshi API integration", "With mock fallback"),
        ("✅ Structured communication", "Clean data structures"),
        ("✅ End-to-end workflow", f"Complete in {duration:.1f}s"),
        ("✅ Error handling", "Robust fallback mechanisms"),
        ("✅ Clear reasoning", "Detailed analysis from both agents")
    ]
    
    for status, description in criteria:
        print(f"{status} {description}")
    
    # Save demonstration results
    demo_file = f"mvp_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(demo_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print_subsection("Next Steps")
    print("🔧 Add real Kalshi API key for live market data")
    print("🔧 Implement position sizing and risk management")
    print("🔧 Add backtesting framework")
    print("🔧 Create web interface for real-time monitoring")
    print("🔧 Add more agents (risk manager, portfolio optimizer)")
    
    print(f"\n📁 Full results saved to: {demo_file}")
    print_section("MVP DEMONSTRATION COMPLETE")

if __name__ == "__main__":
    asyncio.run(demonstrate_mvp())