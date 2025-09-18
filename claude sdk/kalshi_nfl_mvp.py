#!/usr/bin/env python3
"""
Kalshi NFL Trading System MVP
2-Agent System using Claude Code SDK architecture
"""

import os
import json
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GameInput:
    """Input structure for game analysis"""
    team_a: str
    team_b: str
    current_score: Dict[str, int]  # {"team_a": 14, "team_b": 7}
    quarter: int
    time_remaining: str  # "8:45"
    recent_events: List[str]  # ["Bills touchdown", "Dolphins turnover"]

@dataclass
class MarketData:
    """Market data from Kalshi API"""
    event_ticker: str
    team_a_yes_price: float
    team_a_no_price: float
    team_b_yes_price: float
    team_b_no_price: float
    volume_24h: int
    bid_ask_spread: float
    last_trade_price: Optional[float] = None

@dataclass
class SportsAnalysis:
    """Output from Sports Agent"""
    predicted_winner: str  # "team_a", "team_b", "too_close"
    win_probability_team_a: float  # 0.0-1.0
    win_probability_team_b: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    key_factors: List[str]
    momentum_assessment: str
    reasoning: str

@dataclass
class TradingRecommendation:
    """Final output from Trade Agent"""
    action: str  # "BUY", "SELL", "PASS"
    target: str  # "team_a_yes", "team_a_no", "team_b_yes", "team_b_no"
    quantity: int
    confidence: float
    reasoning: str
    sports_signal_strength: float
    market_edge_detected: bool
    risk_assessment: str

class KalshiAPIClient:
    """Simple Kalshi API client for market data"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.kalshi.com/trade-api/v2"):
        self.api_key = api_key
        self.base_url = base_url
        
    async def get_market_data(self, event_ticker: str) -> MarketData:
        """Get market data for event ticker"""
        if not self.api_key:
            # Return mock data for MVP
            return self._mock_market_data(event_ticker)
            
        # Real API implementation would go here
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.base_url}/markets/{event_ticker}"
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_market_data(event_ticker, data)
                    else:
                        logger.warning(f"API request failed: {response.status}, using mock data")
                        return self._mock_market_data(event_ticker)
            except Exception as e:
                logger.error(f"API error: {e}, using mock data")
                return self._mock_market_data(event_ticker)
    
    def _mock_market_data(self, event_ticker: str) -> MarketData:
        """Mock market data for testing"""
        return MarketData(
            event_ticker=event_ticker,
            team_a_yes_price=48.0,
            team_a_no_price=52.0,
            team_b_yes_price=52.0,
            team_b_no_price=48.0,
            volume_24h=15000,
            bid_ask_spread=2.0,
            last_trade_price=47.5
        )
    
    def _parse_market_data(self, event_ticker: str, api_data: Dict) -> MarketData:
        """Parse real Kalshi API response"""
        # Implementation would parse actual Kalshi response format
        # For now, return mock data
        return self._mock_market_data(event_ticker)

class SportsAgent:
    """Sports analysis agent using Claude Code SDK"""
    
    def __init__(self):
        self.agent_name = "Sports Analyst"
        
    async def analyze_game(self, game_input: GameInput) -> SportsAnalysis:
        """Analyze NFL game dynamics and provide sports-focused trading signal"""
        
        try:
            # Import Claude Code SDK
            from claude_code_sdk import query, ClaudeCodeOptions
            
            # Create comprehensive sports analysis prompt
            prompt = self._create_sports_prompt(game_input)
            
            # Configure Claude options for sports analysis
            options = ClaudeCodeOptions(
                system_prompt="""You are an expert NFL analyst with deep knowledge of game dynamics, 
                team performance, momentum shifts, and win probability assessment. You analyze live 
                game situations to predict outcomes with high accuracy.""",
                max_thinking_tokens=4000,
                model="claude-3-5-sonnet-20241022"
            )
            
            # Get analysis from Claude
            result_content = ""
            async for message in query(prompt=prompt, options=options):
                if hasattr(message, 'content') and message.content:
                    for block in message.content:
                        if hasattr(block, 'text'):
                            result_content += block.text
            
            # Parse the JSON response
            return self._parse_sports_analysis(result_content)
            
        except ImportError:
            logger.error("Claude Code SDK not available, using mock analysis")
            return self._mock_sports_analysis(game_input)
        except Exception as e:
            logger.error(f"Sports analysis error: {e}")
            return self._mock_sports_analysis(game_input)
    
    def _create_sports_prompt(self, game_input: GameInput) -> str:
        """Create comprehensive sports analysis prompt"""
        
        return f"""
Analyze this live NFL game situation and provide a sports-focused trading assessment:

GAME SITUATION:
- Matchup: {game_input.team_a} vs {game_input.team_b}
- Current Score: {game_input.team_a} {game_input.current_score['team_a']} - {game_input.team_b} {game_input.current_score['team_b']}
- Quarter: {game_input.quarter}
- Time Remaining: {game_input.time_remaining}
- Recent Events: {', '.join(game_input.recent_events)}

ANALYSIS FRAMEWORK:
1. MOMENTUM ASSESSMENT: Which team has the momentum right now?
2. SCORING PROBABILITY: Who is more likely to score next?
3. GAME FLOW: Is this following expected patterns or are there surprises?
4. TIME MANAGEMENT: How does remaining time affect win probability?
5. SITUATIONAL FACTORS: What situational advantages exist?

Provide your analysis in this exact JSON format:
{{
    "predicted_winner": "team_a|team_b|too_close",
    "win_probability_team_a": 0.0-1.0,
    "win_probability_team_b": 0.0-1.0,
    "confidence": 0.0-1.0,
    "key_factors": ["factor1", "factor2", "factor3"],
    "momentum_assessment": "strong_team_a|slight_team_a|neutral|slight_team_b|strong_team_b",
    "reasoning": "Detailed sports analysis explaining your assessment"
}}

Focus on game dynamics, momentum, and win probability. Be specific about why you believe one team has an edge.
"""
    
    def _parse_sports_analysis(self, content: str) -> SportsAnalysis:
        """Parse Claude's response into SportsAnalysis object"""
        try:
            # Extract JSON from response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                # Try to find JSON in the content
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                else:
                    raise ValueError("No JSON found in response")
            
            data = json.loads(json_str)
            
            return SportsAnalysis(
                predicted_winner=data.get("predicted_winner", "too_close"),
                win_probability_team_a=data.get("win_probability_team_a", 0.5),
                win_probability_team_b=data.get("win_probability_team_b", 0.5),
                confidence=data.get("confidence", 0.5),
                key_factors=data.get("key_factors", []),
                momentum_assessment=data.get("momentum_assessment", "neutral"),
                reasoning=data.get("reasoning", "Analysis completed")
            )
            
        except Exception as e:
            logger.error(f"Error parsing sports analysis: {e}")
            # Return safe fallback
            return SportsAnalysis(
                predicted_winner="too_close",
                win_probability_team_a=0.5,
                win_probability_team_b=0.5,
                confidence=0.3,
                key_factors=["parsing_error"],
                momentum_assessment="neutral",
                reasoning=f"Error parsing analysis: {str(e)}"
            )
    
    def _mock_sports_analysis(self, game_input: GameInput) -> SportsAnalysis:
        """Mock sports analysis for testing"""
        team_a_score = game_input.current_score['team_a']
        team_b_score = game_input.current_score['team_b']
        
        if team_a_score > team_b_score:
            predicted_winner = "team_a"
            prob_a, prob_b = 0.65, 0.35
        elif team_b_score > team_a_score:
            predicted_winner = "team_b"
            prob_a, prob_b = 0.35, 0.65
        else:
            predicted_winner = "too_close"
            prob_a, prob_b = 0.5, 0.5
        
        return SportsAnalysis(
            predicted_winner=predicted_winner,
            win_probability_team_a=prob_a,
            win_probability_team_b=prob_b,
            confidence=0.7,
            key_factors=["score_differential", "game_time", "recent_events"],
            momentum_assessment="neutral",
            reasoning="Mock analysis based on current score"
        )

class TradeAgent:
    """Trading agent that combines sports analysis with market data"""
    
    def __init__(self):
        self.agent_name = "Trade Analyst"
        
    async def make_trading_decision(self, sports_analysis: SportsAnalysis, 
                                  market_data: MarketData, 
                                  game_input: GameInput) -> TradingRecommendation:
        """Make final trading recommendation combining sports and market analysis"""
        
        try:
            # Import Claude Code SDK
            from claude_code_sdk import query, ClaudeCodeOptions
            
            # Create trading analysis prompt
            prompt = self._create_trading_prompt(sports_analysis, market_data, game_input)
            
            # Configure Claude options for trading analysis
            options = ClaudeCodeOptions(
                system_prompt="""You are an expert quantitative trader specializing in sports betting 
                markets. You combine sports analysis with market microstructure to identify profitable 
                trading opportunities. You understand market inefficiencies and know when to act.""",
                max_thinking_tokens=4000,
                model="claude-3-5-sonnet-20241022"
            )
            
            # Get trading recommendation from Claude
            result_content = ""
            async for message in query(prompt=prompt, options=options):
                if hasattr(message, 'content') and message.content:
                    for block in message.content:
                        if hasattr(block, 'text'):
                            result_content += block.text
            
            # Parse the trading recommendation
            return self._parse_trading_recommendation(result_content)
            
        except ImportError:
            logger.error("Claude Code SDK not available, using mock trading decision")
            return self._mock_trading_decision(sports_analysis, market_data)
        except Exception as e:
            logger.error(f"Trading analysis error: {e}")
            return self._mock_trading_decision(sports_analysis, market_data)
    
    def _create_trading_prompt(self, sports_analysis: SportsAnalysis, 
                              market_data: MarketData, game_input: GameInput) -> str:
        """Create comprehensive trading analysis prompt"""
        
        return f"""
SPORTS ANALYST RECOMMENDATION:
- Predicted Winner: {sports_analysis.predicted_winner}
- Win Probability {game_input.team_a}: {sports_analysis.win_probability_team_a:.1%}
- Win Probability {game_input.team_b}: {sports_analysis.win_probability_team_b:.1%}
- Confidence: {sports_analysis.confidence:.1%}
- Key Factors: {', '.join(sports_analysis.key_factors)}
- Momentum: {sports_analysis.momentum_assessment}
- Reasoning: {sports_analysis.reasoning}

MARKET DATA:
- Event: {market_data.event_ticker}
- {game_input.team_a} Win Prices: Yes ${market_data.team_a_yes_price} | No ${market_data.team_a_no_price}
- {game_input.team_b} Win Prices: Yes ${market_data.team_b_yes_price} | No ${market_data.team_b_no_price}
- Volume (24h): {market_data.volume_24h:,}
- Bid-Ask Spread: ${market_data.bid_ask_spread}
- Last Trade: ${market_data.last_trade_price or 'N/A'}

GAME CONTEXT:
- Quarter {game_input.quarter}, {game_input.time_remaining} remaining
- Score: {game_input.team_a} {game_input.current_score['team_a']} - {game_input.team_b} {game_input.current_score['team_b']}

TRADING ANALYSIS FRAMEWORK:
1. MARKET EFFICIENCY: Do market prices reflect the sports analysis probabilities?
2. EDGE IDENTIFICATION: Where is the biggest discrepancy between sports and market?
3. LIQUIDITY CHECK: Is there sufficient volume to execute trades?
4. RISK ASSESSMENT: What's the potential downside vs upside?
5. POSITION SIZING: How much should we risk on this opportunity?

Provide your trading recommendation in this exact JSON format:
{{
    "action": "BUY|SELL|PASS",
    "target": "team_a_yes|team_a_no|team_b_yes|team_b_no",
    "quantity": 1-1000,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation combining sports analysis with market assessment",
    "sports_signal_strength": 0.0-1.0,
    "market_edge_detected": true/false,
    "risk_assessment": "low|medium|high"
}}

Make your decision based on finding market inefficiencies where the sports analysis suggests different probabilities than market prices.
"""
    
    def _parse_trading_recommendation(self, content: str) -> TradingRecommendation:
        """Parse Claude's trading recommendation"""
        try:
            # Extract JSON from response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                # Try to find JSON in the content
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                else:
                    raise ValueError("No JSON found in response")
            
            data = json.loads(json_str)
            
            return TradingRecommendation(
                action=data.get("action", "PASS"),
                target=data.get("target", ""),
                quantity=data.get("quantity", 0),
                confidence=data.get("confidence", 0.0),
                reasoning=data.get("reasoning", "Trading analysis completed"),
                sports_signal_strength=data.get("sports_signal_strength", 0.0),
                market_edge_detected=data.get("market_edge_detected", False),
                risk_assessment=data.get("risk_assessment", "medium")
            )
            
        except Exception as e:
            logger.error(f"Error parsing trading recommendation: {e}")
            # Return safe fallback
            return TradingRecommendation(
                action="PASS",
                target="",
                quantity=0,
                confidence=0.0,
                reasoning=f"Error parsing recommendation: {str(e)}",
                sports_signal_strength=0.0,
                market_edge_detected=False,
                risk_assessment="high"
            )
    
    def _mock_trading_decision(self, sports_analysis: SportsAnalysis, 
                              market_data: MarketData) -> TradingRecommendation:
        """Mock trading decision for testing"""
        if sports_analysis.confidence > 0.7:
            action = "BUY"
            target = "team_a_yes" if sports_analysis.predicted_winner == "team_a" else "team_b_yes"
            quantity = 100
            confidence = 0.6
        else:
            action = "PASS"
            target = ""
            quantity = 0
            confidence = 0.3
        
        return TradingRecommendation(
            action=action,
            target=target,
            quantity=quantity,
            confidence=confidence,
            reasoning="Mock trading decision based on sports confidence",
            sports_signal_strength=sports_analysis.confidence,
            market_edge_detected=sports_analysis.confidence > 0.7,
            risk_assessment="medium"
        )

class KalshiNFLTradingSystem:
    """Main orchestrator for the 2-agent Kalshi NFL trading system"""
    
    def __init__(self, kalshi_api_key: str = None):
        self.kalshi_client = KalshiAPIClient(kalshi_api_key)
        self.sports_agent = SportsAgent()
        self.trade_agent = TradeAgent()
        
        logger.info("Kalshi NFL Trading System MVP initialized")
    
    async def analyze_game(self, game_input: GameInput) -> Dict[str, Any]:
        """Main analysis workflow"""
        
        start_time = datetime.now()
        logger.info(f"Starting analysis for {game_input.team_a} vs {game_input.team_b}")
        
        try:
            # Step 1: Get market data
            event_ticker = f"NFL-{game_input.team_a}-{game_input.team_b}-{datetime.now().strftime('%Y-%m-%d')}"
            market_data = await self.kalshi_client.get_market_data(event_ticker)
            
            # Step 2: Sports Agent analysis
            logger.info("Running sports analysis...")
            sports_analysis = await self.sports_agent.analyze_game(game_input)
            
            # Step 3: Trade Agent decision
            logger.info("Running trading analysis...")
            trading_recommendation = await self.trade_agent.make_trading_decision(
                sports_analysis, market_data, game_input
            )
            
            # Step 4: Compile results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "timestamp": start_time.isoformat(),
                "analysis_duration_seconds": duration,
                "game_input": asdict(game_input),
                "market_data": asdict(market_data),
                "sports_analysis": asdict(sports_analysis),
                "trading_recommendation": asdict(trading_recommendation),
                "system_info": {
                    "version": "1.0.0-mvp",
                    "agents_used": ["SportsAgent", "TradeAgent"],
                    "api_integration": "kalshi" if self.kalshi_client.api_key else "mock"
                }
            }
            
            logger.info(f"Analysis completed in {duration:.2f} seconds")
            logger.info(f"Recommendation: {trading_recommendation.action} {trading_recommendation.target}")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "timestamp": start_time.isoformat(),
                "error": str(e),
                "game_input": asdict(game_input)
            }

# Example usage function
async def example_usage():
    """Example of how to use the system"""
    
    # Create sample game input
    game_input = GameInput(
        team_a="Buffalo Bills",
        team_b="Miami Dolphins",
        current_score={"team_a": 14, "team_b": 7},
        quarter=3,
        time_remaining="8:45",
        recent_events=["Bills touchdown", "Dolphins turnover"]
    )
    
    # Initialize system (no API key for MVP testing)
    system = KalshiNFLTradingSystem(kalshi_api_key=None)
    
    # Run analysis
    result = await system.analyze_game(game_input)
    
    # Display results
    print("\n" + "="*70)
    print("KALSHI NFL TRADING SYSTEM MVP - ANALYSIS RESULTS")
    print("="*70)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    sports = result["sports_analysis"]
    trading = result["trading_recommendation"]
    
    print(f"\nGAME: {game_input.team_a} vs {game_input.team_b}")
    print(f"Score: {game_input.current_score['team_a']} - {game_input.current_score['team_b']}")
    print(f"Time: Q{game_input.quarter}, {game_input.time_remaining}")
    
    print(f"\nSPORTS ANALYSIS:")
    print(f"  Predicted Winner: {sports['predicted_winner']}")
    print(f"  {game_input.team_a} Win Prob: {sports['win_probability_team_a']:.1%}")
    print(f"  {game_input.team_b} Win Prob: {sports['win_probability_team_b']:.1%}")
    print(f"  Confidence: {sports['confidence']:.1%}")
    print(f"  Key Factors: {', '.join(sports['key_factors'])}")
    
    print(f"\nTRADING RECOMMENDATION:")
    print(f"  Action: {trading['action']}")
    print(f"  Target: {trading['target']}")
    print(f"  Quantity: {trading['quantity']}")
    print(f"  Confidence: {trading['confidence']:.1%}")
    print(f"  Market Edge: {trading['market_edge_detected']}")
    print(f"  Risk: {trading['risk_assessment']}")
    
    print(f"\nREASONING:")
    print(f"  Sports: {sports['reasoning'][:150]}...")
    print(f"  Trading: {trading['reasoning'][:150]}...")
    
    print(f"\nDuration: {result['analysis_duration_seconds']:.2f} seconds")
    
    # Save full results
    output_file = f"kalshi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Full analysis saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(example_usage())