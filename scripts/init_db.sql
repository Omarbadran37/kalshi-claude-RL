-- Database initialization script for NFL Trading System

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create games table
CREATE TABLE IF NOT EXISTS games (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id VARCHAR(50) UNIQUE NOT NULL,
    home_team VARCHAR(10) NOT NULL,
    away_team VARCHAR(10) NOT NULL,
    game_date TIMESTAMP WITH TIME ZONE NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create plays table
CREATE TABLE IF NOT EXISTS plays (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    play_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    play_type VARCHAR(20) NOT NULL,
    down INTEGER,
    distance INTEGER,
    field_position INTEGER,
    score_home INTEGER NOT NULL DEFAULT 0,
    score_away INTEGER NOT NULL DEFAULT 0,
    time_remaining INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    possession_team VARCHAR(10) NOT NULL,
    description TEXT,
    result VARCHAR(20),
    yards_gained INTEGER,
    formation VARCHAR(50),
    personnel VARCHAR(20),
    momentum_score NUMERIC(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(play_id, game_id)
);

-- Create markets table
CREATE TABLE IF NOT EXISTS markets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    market_type VARCHAR(50) NOT NULL,
    game_id UUID REFERENCES games(id) ON DELETE SET NULL,
    close_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create price_data table
CREATE TABLE IF NOT EXISTS price_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID REFERENCES markets(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price NUMERIC(6,5) NOT NULL CHECK (open_price >= 0 AND open_price <= 1),
    high_price NUMERIC(6,5) NOT NULL CHECK (high_price >= 0 AND high_price <= 1),
    low_price NUMERIC(6,5) NOT NULL CHECK (low_price >= 0 AND low_price <= 1),
    close_price NUMERIC(6,5) NOT NULL CHECK (close_price >= 0 AND close_price <= 1),
    volume INTEGER NOT NULL DEFAULT 0 CHECK (volume >= 0),
    bid_price NUMERIC(6,5) CHECK (bid_price >= 0 AND bid_price <= 1),
    ask_price NUMERIC(6,5) CHECK (ask_price >= 0 AND ask_price <= 1),
    bid_size INTEGER CHECK (bid_size >= 0),
    ask_size INTEGER CHECK (ask_size >= 0),
    num_trades INTEGER CHECK (num_trades >= 0),
    vwap NUMERIC(6,5) CHECK (vwap >= 0 AND vwap <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(market_id, timestamp)
);

-- Create aligned_data table (for ML features)
CREATE TABLE IF NOT EXISTS aligned_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    play_id UUID REFERENCES plays(id) ON DELETE CASCADE,
    market_id UUID REFERENCES markets(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    time_diff_seconds INTEGER NOT NULL,
    price_close NUMERIC(6,5),
    price_volume INTEGER,
    momentum_score NUMERIC(5,4),
    features JSONB,
    prediction NUMERIC(5,4),
    actual_outcome NUMERIC(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(play_id, market_id)
);

-- Create model_predictions table
CREATE TABLE IF NOT EXISTS model_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    aligned_data_id UUID REFERENCES aligned_data(id) ON DELETE CASCADE,
    prediction NUMERIC(5,4) NOT NULL,
    confidence NUMERIC(5,4),
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create trading_signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID REFERENCES markets(id) ON DELETE CASCADE,
    signal_type VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    strength NUMERIC(3,2) NOT NULL CHECK (strength >= 0 AND strength <= 1),
    price_target NUMERIC(6,5) CHECK (price_target >= 0 AND price_target <= 1),
    confidence NUMERIC(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    reasoning TEXT,
    model_predictions JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'ACTIVE', -- 'ACTIVE', 'EXECUTED', 'EXPIRED', 'CANCELLED'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_plays_timestamp ON plays(timestamp);
CREATE INDEX IF NOT EXISTS idx_plays_game_id ON plays(game_id);
CREATE INDEX IF NOT EXISTS idx_price_data_timestamp ON price_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_price_data_market_id ON price_data(market_id);
CREATE INDEX IF NOT EXISTS idx_aligned_data_timestamp ON aligned_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_aligned_data_play_id ON aligned_data(play_id);
CREATE INDEX IF NOT EXISTS idx_aligned_data_market_id ON aligned_data(market_id);
CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_trading_signals_market_id ON trading_signals(market_id);
CREATE INDEX IF NOT EXISTS idx_trading_signals_status ON trading_signals(status);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_games_updated_at BEFORE UPDATE ON games
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_markets_updated_at BEFORE UPDATE ON markets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create a view for recent trading activity
CREATE OR REPLACE VIEW recent_trading_activity AS
SELECT
    ts.id,
    m.market_id,
    m.title as market_title,
    ts.signal_type,
    ts.strength,
    ts.price_target,
    ts.confidence,
    ts.timestamp,
    ts.status,
    g.home_team,
    g.away_team,
    g.game_date
FROM trading_signals ts
JOIN markets m ON ts.market_id = m.id
LEFT JOIN games g ON m.game_id = g.id
WHERE ts.timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY ts.timestamp DESC;

-- Grant permissions to nfltrader user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nfltrader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nfltrader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO nfltrader;