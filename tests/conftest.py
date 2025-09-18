"""Pytest configuration and fixtures for NFL Trading System tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Any

from src.nfl_trading.config import Config


@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration."""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    config_data = {
        'environment': 'test',
        'data': {
            'raw_path': str(temp_path / 'raw'),
            'processed_path': str(temp_path / 'processed'),
            'external_path': str(temp_path / 'external'),
            'nfl_data_path': str(temp_path / 'raw' / 'nfl'),
            'kalshi_data_path': str(temp_path / 'raw' / 'kalshi'),
        },
        'processing': {
            'batch_size': 100,
            'max_workers': 2,
            'chunk_size': 1000,
            'time_alignment_tolerance': 60
        },
        'logging': {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': str(temp_path / 'logs' / 'test.log'),
            'max_file_size': '10MB',
            'backup_count': 3,
            'console_output': False
        },
        'database': {
            'type': 'sqlite',
            'path': str(temp_path / 'test.db'),
            'echo': False,
            'pool_size': 1,
            'max_overflow': 1
        },
        'cache': {
            'type': 'memory',
            'ttl': 3600,
            'max_size': 100
        },
        'security': {
            'api_key_env_var': 'TEST_API_KEY',
            'secret_key_env_var': 'TEST_SECRET_KEY'
        },
        'api': {
            'kalshi': {'base_url': 'https://api.test.com', 'timeout': 30},
            'nfl': {'base_url': 'https://api.test.com', 'timeout': 30}
        },
        'model': {
            'hyperparameters': {'learning_rate': 0.001},
            'reinforcement_learning': {'algorithm': 'PPO'}
        },
        'features': {
            'nfl_features': ['down', 'distance', 'field_position'],
            'market_features': ['price', 'volume']
        },
        'monitoring': {
            'enable_metrics': False,
            'metrics_port': 8080,
            'health_check_port': 8081
        }
    }

    # Create temporary config file
    config_file = temp_path / 'test_config.yaml'
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    # Create directories
    for path_key, path_value in config_data['data'].items():
        Path(path_value).mkdir(parents=True, exist_ok=True)
    Path(config_data['logging']['file']).parent.mkdir(parents=True, exist_ok=True)

    return Config(config_path=str(config_file))


@pytest.fixture
def sample_nfl_data():
    """Create sample NFL play-by-play data."""
    base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

    plays = []
    for i in range(10):
        play = {
            'id': f'play_{i}',
            'timestamp': (base_time + timedelta(minutes=i*2)).isoformat(),
            'type': 'pass' if i % 2 == 0 else 'run',
            'down': (i % 4) + 1,
            'distance': np.random.randint(1, 21),
            'field_position': np.random.randint(20, 80),
            'score_home': np.random.randint(0, 35),
            'score_away': np.random.randint(0, 35),
            'time_remaining': 3600 - (i * 120),
            'quarter': min(4, (i // 3) + 1),
            'possession_team': 'HOME' if i % 2 == 0 else 'AWAY',
            'description': f'Sample play {i}',
            'result': 'success' if i % 3 != 0 else 'failure',
            'yards_gained': np.random.randint(-5, 25),
            'formation': 'shotgun' if i % 2 == 0 else 'singleback',
            'personnel': '11' if i % 2 == 0 else '21'
        }
        plays.append(play)

    return {
        'game': {
            'id': 'game_123',
            'home_team': 'HOME',
            'away_team': 'AWAY'
        },
        'plays': plays
    }


@pytest.fixture
def sample_kalshi_data():
    """Create sample Kalshi price data."""
    base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

    candlesticks = []
    price = 0.5

    for i in range(20):
        # Simulate price movement
        price_change = np.random.normal(0, 0.02)
        price = max(0.01, min(0.99, price + price_change))

        candlestick = {
            'timestamp': (base_time + timedelta(minutes=i)).isoformat(),
            'open': round(price, 4),
            'high': round(price + abs(np.random.normal(0, 0.01)), 4),
            'low': round(max(0.01, price - abs(np.random.normal(0, 0.01))), 4),
            'close': round(price, 4),
            'volume': np.random.randint(100, 1000),
            'bid': round(price - 0.01, 4),
            'ask': round(price + 0.01, 4),
            'bid_size': np.random.randint(10, 100),
            'ask_size': np.random.randint(10, 100),
            'trades': np.random.randint(1, 20),
            'vwap': round(price + np.random.normal(0, 0.005), 4)
        }
        candlesticks.append(candlestick)

    return {
        'market_id': 'market_123',
        'candlesticks': candlesticks
    }


@pytest.fixture
def sample_nfl_dataframe():
    """Create sample NFL DataFrame."""
    base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

    data = []
    for i in range(10):
        record = {
            'timestamp': base_time + timedelta(minutes=i*2),
            'game_id': 'game_123',
            'play_id': f'play_{i}',
            'play_type': 'pass' if i % 2 == 0 else 'run',
            'down': (i % 4) + 1,
            'distance': np.random.randint(1, 21),
            'field_position': np.random.randint(20, 80),
            'score_home': np.random.randint(0, 35),
            'score_away': np.random.randint(0, 35),
            'score_differential': np.random.randint(-21, 21),
            'time_remaining': 3600 - (i * 120),
            'quarter': min(4, (i // 3) + 1),
            'possession_team': 'HOME' if i % 2 == 0 else 'AWAY',
            'home_team': 'HOME',
            'away_team': 'AWAY',
            'description': f'Sample play {i}',
            'result': 'success' if i % 3 != 0 else 'failure',
            'yards_gained': np.random.randint(-5, 25),
            'formation': 'shotgun' if i % 2 == 0 else 'singleback',
            'personnel': '11' if i % 2 == 0 else '21',
            'momentum_score': np.random.uniform(-1, 1)
        }
        data.append(record)

    return pd.DataFrame(data)


@pytest.fixture
def sample_kalshi_dataframe():
    """Create sample Kalshi DataFrame."""
    base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

    data = []
    price = 0.5

    for i in range(20):
        # Simulate price movement
        price_change = np.random.normal(0, 0.02)
        price = max(0.01, min(0.99, price + price_change))

        record = {
            'timestamp': base_time + timedelta(minutes=i),
            'market_id': 'market_123',
            'open_price': round(price, 4),
            'high_price': round(price + abs(np.random.normal(0, 0.01)), 4),
            'low_price': round(max(0.01, price - abs(np.random.normal(0, 0.01))), 4),
            'close_price': round(price, 4),
            'volume': np.random.randint(100, 1000),
            'bid_price': round(price - 0.01, 4),
            'ask_price': round(price + 0.01, 4),
            'bid_size': np.random.randint(10, 100),
            'ask_size': np.random.randint(10, 100),
            'num_trades': np.random.randint(1, 20),
            'vwap': round(price + np.random.normal(0, 0.005), 4)
        }
        data.append(record)

    return pd.DataFrame(data)


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file."""
    def _create_temp_json(data: Dict[str, Any], filename: str = "test_data.json") -> Path:
        file_path = tmp_path / filename
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return file_path

    return _create_temp_json


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory with subdirectories."""
    def _create_temp_dir(subdirs: List[str] = None) -> Path:
        base_dir = tmp_path / "test_dir"
        base_dir.mkdir(exist_ok=True)

        if subdirs:
            for subdir in subdirs:
                (base_dir / subdir).mkdir(parents=True, exist_ok=True)

        return base_dir

    return _create_temp_dir