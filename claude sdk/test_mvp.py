#!/usr/bin/env python3
"""
Test script for Kalshi NFL Trading System MVP
Demonstrates the 2-agent system with various game scenarios
"""

import asyncio
import json
from datetime import datetime
from kalshi_nfl_mvp import KalshiNFLTradingSystem, GameInput

async def test_game_scenarios():
    """Test the system with different game scenarios"""
    
    # Initialize the system
    system = KalshiNFLTradingSystem(kalshi_api_key=None)  # Using mock data for MVP
    
    # Test scenarios
    scenarios = [
        {
            "name": "Close Game - Late 4th Quarter",
            "game": GameInput(
                team_a="Kansas City Chiefs",
                team_b="Buffalo Bills",
                current_score={"team_a": 21, "team_b": 20},
                quarter=4,
                time_remaining="2:15",
                recent_events=["Chiefs field goal", "Bills timeout", "2-minute warning"]
            )
        },
        {
            "name": "Blowout - Early Lead",
            "game": GameInput(
                team_a="San Francisco 49ers",
                team_b="Arizona Cardinals",
                current_score={"team_a": 28, "team_b": 7},
                quarter=2,
                time_remaining="8:30",
                recent_events=["49ers touchdown", "Cardinals fumble", "49ers interception"]
            )
        },
        {
            "name": "Comeback Attempt",
            "game": GameInput(
                team_a="Green Bay Packers",
                team_b="Chicago Bears",
                current_score={"team_a": 14, "team_b": 21},
                quarter=4,
                time_remaining="6:45",
                recent_events=["Packers touchdown", "Bears punt", "Packers onside kick recovery"]
            )
        }
    ]
    
    print("="*70)
    print("KALSHI NFL TRADING SYSTEM MVP - TEST SCENARIOS")
    print("="*70)
    
    all_results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'-'*50}")
        print(f"TEST {i}: {scenario['name']}")
        print(f"{'-'*50}")
        
        try:
            result = await system.analyze_game(scenario['game'])
            
            if "error" not in result:
                # Extract key information
                game = result['game_input']
                sports = result['sports_analysis']
                trading = result['trading_recommendation']
                
                print(f"Game: {game['team_a']} {game['current_score']['team_a']} - {game['team_b']} {game['current_score']['team_b']}")
                print(f"Time: Q{game['quarter']}, {game['time_remaining']}")
                print(f"Recent: {', '.join(game['recent_events'])}")
                
                print(f"\nSports Analysis:")
                print(f"  Winner: {sports['predicted_winner']}")
                print(f"  {game['team_a']}: {sports['win_probability_team_a']:.1%}")
                print(f"  {game['team_b']}: {sports['win_probability_team_b']:.1%}")
                print(f"  Confidence: {sports['confidence']:.1%}")
                print(f"  Momentum: {sports['momentum_assessment']}")
                
                print(f"\nTrading Decision:")
                print(f"  Action: {trading['action']}")
                print(f"  Target: {trading['target']}")
                print(f"  Quantity: {trading['quantity']}")
                print(f"  Confidence: {trading['confidence']:.1%}")
                print(f"  Edge Detected: {trading['market_edge_detected']}")
                print(f"  Risk: {trading['risk_assessment']}")
                
                print(f"\nDuration: {result['analysis_duration_seconds']:.2f}s")
                
                all_results.append({
                    "scenario": scenario['name'],
                    "result": result
                })
                
            else:
                print(f"ERROR: {result['error']}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Save all test results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"mvp_test_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"All test results saved to: {output_file}")
    print(f"MVP testing completed successfully!")
    print(f"{'='*70}")

async def test_single_game():
    """Test with the specific example from requirements"""
    
    system = KalshiNFLTradingSystem(kalshi_api_key=None)
    
    # Use the exact example from requirements
    game_input = GameInput(
        team_a="Buffalo Bills",
        team_b="Miami Dolphins",
        current_score={"team_a": 14, "team_b": 7},
        quarter=3,
        time_remaining="8:45",
        recent_events=["Bills touchdown", "Dolphins turnover"]
    )
    
    print("="*70)
    print("TESTING REQUIREMENTS EXAMPLE")
    print("="*70)
    
    result = await system.analyze_game(game_input)
    
    # Pretty print the complete result
    if "error" not in result:
        print("\nCOMPLETE ANALYSIS RESULT:")
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Single game (requirements example)")
    print("2. Multiple scenarios")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_single_game())
    else:
        asyncio.run(test_game_scenarios())