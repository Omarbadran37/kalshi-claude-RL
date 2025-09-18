"""Configuration management for NFL Trading System."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, validator
from dotenv import load_dotenv


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str
    host: Optional[str] = None
    port: Optional[int] = None
    name: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    path: Optional[str] = None  # for SQLite
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


class APIConfig(BaseModel):
    """API configuration."""
    kalshi: Dict[str, Any]
    nfl: Dict[str, Any]


class DataConfig(BaseModel):
    """Data paths configuration."""
    raw_path: str
    processed_path: str
    external_path: str
    nfl_data_path: str
    kalshi_data_path: str


class ProcessingConfig(BaseModel):
    """Data processing configuration."""
    batch_size: int
    max_workers: int
    chunk_size: int
    time_alignment_tolerance: int


class ModelConfig(BaseModel):
    """Model configuration."""
    hyperparameters: Dict[str, Any]
    reinforcement_learning: Dict[str, Any]


class FeaturesConfig(BaseModel):
    """Features configuration."""
    nfl_features: list
    market_features: list


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str
    format: str
    file: str
    max_file_size: str
    backup_count: int
    console_output: bool


class CacheConfig(BaseModel):
    """Cache configuration."""
    type: str
    ttl: int
    max_size: Optional[int] = None
    host: Optional[str] = None
    port: Optional[int] = None
    db: Optional[int] = None
    max_connections: Optional[int] = None


class SecurityConfig(BaseModel):
    """Security configuration."""
    api_key_env_var: str
    secret_key_env_var: str
    ssl_verify: bool = True
    encryption_key_env_var: Optional[str] = None


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enable_metrics: bool
    metrics_port: int
    health_check_port: int
    prometheus_enabled: bool = False
    grafana_enabled: bool = False


class Config:
    """Main configuration class for NFL Trading System."""

    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file
            environment: Environment name (dev, prod)
        """
        # Load environment variables
        load_dotenv()

        # Determine environment
        self.environment = environment or os.getenv('ENVIRONMENT', 'dev')

        # Determine config path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / 'configs' / f'{self.environment}.yaml'

        # Load configuration
        self._config = self._load_config(config_path)

        # Initialize configuration objects
        self.database = DatabaseConfig(**self._config['database'])
        self.api = APIConfig(**self._config['api'])
        self.data = DataConfig(**self._config['data'])
        self.processing = ProcessingConfig(**self._config['processing'])
        self.model = ModelConfig(**self._config['model'])
        self.features = FeaturesConfig(**self._config['features'])
        self.logging = LoggingConfig(**self._config['logging'])
        self.cache = CacheConfig(**self._config['cache'])
        self.security = SecurityConfig(**self._config['security'])
        self.monitoring = MonitoringConfig(**self._config['monitoring'])

        # Create data directories
        self._create_directories()

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Replace environment variables in config
        config = self._substitute_env_vars(config)

        return config

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config."""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config

    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data.raw_path,
            self.data.processed_path,
            self.data.external_path,
            self.data.nfl_data_path,
            self.data.kalshi_data_path,
            Path(self.logging.file).parent
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_api_key(self) -> str:
        """Get API key from environment."""
        api_key = os.getenv(self.security.api_key_env_var)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.security.api_key_env_var}")
        return api_key

    def get_secret_key(self) -> str:
        """Get secret key from environment."""
        secret_key = os.getenv(self.security.secret_key_env_var)
        if not secret_key:
            raise ValueError(f"Secret key not found in environment variable: {self.security.secret_key_env_var}")
        return secret_key

    def get_database_url(self) -> str:
        """Get database URL based on configuration."""
        if self.database.type == 'sqlite':
            return f"sqlite:///{self.database.path}"
        elif self.database.type == 'postgresql':
            return (f"postgresql://{self.database.user}:{self.database.password}"
                   f"@{self.database.host}:{self.database.port}/{self.database.name}")
        else:
            raise ValueError(f"Unsupported database type: {self.database.type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config

    def save(self, output_path: str):
        """Save current configuration to file."""
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)


# Global configuration instance
config = None

def get_config(config_path: Optional[str] = None, environment: Optional[str] = None) -> Config:
    """Get global configuration instance."""
    global config
    if config is None:
        config = Config(config_path, environment)
    return config


def reload_config(config_path: Optional[str] = None, environment: Optional[str] = None):
    """Reload global configuration."""
    global config
    config = Config(config_path, environment)
    return config