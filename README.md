# NFL Momentum Trading System

A reinforcement learning system that predicts price movements on Kalshi prediction markets using NFL play-by-play data and advanced machine learning techniques.

## 🏈 Overview

This system analyzes NFL game events in real-time and predicts momentum shifts that can be traded on Kalshi prediction markets. It combines play-by-play data analysis, technical indicators, and reinforcement learning to generate profitable trading signals.

### Key Features

- **Real-time Data Processing**: Ingests and processes NFL play-by-play data and Kalshi price feeds
- **Advanced Analytics**: Calculates momentum scores and technical indicators
- **Machine Learning**: Uses reinforcement learning for predictive modeling
- **Scalable Architecture**: Docker-containerized with microservices design
- **Production Ready**: Comprehensive logging, monitoring, and testing

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL (for production)
- Redis (for caching)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/nfl-trading-system.git
   cd nfl-trading-system
   ```

2. **Set up development environment**
   ```bash
   make setup-env
   source venv/bin/activate
   make install-dev
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run tests**
   ```bash
   make test
   ```

5. **Start the system with Docker**
   ```bash
   docker-compose up -d
   ```

## 📊 Data Pipeline

### NFL Data Processing
- Parses play-by-play JSON data from NFL APIs
- Calculates momentum scores based on game situation
- Extracts features like field position, score differential, time remaining

### Kalshi Data Processing
- Processes price candlestick data at 60-second intervals
- Calculates technical indicators (RSI, moving averages, Bollinger bands)
- Detects price anomalies and volume spikes

### Data Alignment
- Synchronizes NFL plays with market prices by timestamp
- Multiple alignment methods (nearest, forward-fill, interpolation)
- Configurable time tolerance for matching events

## 🧠 Machine Learning

The system uses several ML approaches:

1. **Feature Engineering**: Creates features from play data and market indicators
2. **Reinforcement Learning**: PPO algorithm for trading decision optimization
3. **Technical Analysis**: Traditional indicators adapted for prediction markets
4. **Momentum Scoring**: Proprietary algorithm for measuring game momentum

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NFL Data      │    │  Kalshi Data     │    │  Data Aligner   │
│   Processor     │───▶│  Processor       │───▶│                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ML Models     │    │  Trading         │    │  Feature        │
│                 │◀───│  Signals         │◀───│  Engineering    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components

- **Data Processors**: Handle ingestion and transformation of raw data
- **Data Aligner**: Synchronizes multi-source data by timestamp
- **Feature Engineering**: Creates ML-ready datasets
- **Model Training**: Reinforcement learning pipeline
- **Signal Generation**: Produces trading recommendations
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 📁 Project Structure

```
nfl-trading-system/
├── src/nfl_trading/          # Main application code
│   ├── data/                 # Data processing modules
│   ├── models/               # ML models and training
│   ├── config/               # Configuration management
│   └── utils/                # Utility functions
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── configs/                  # Configuration files
├── data/                     # Data storage
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed datasets
│   └── external/             # External data sources
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Utility scripts
└── monitoring/               # Monitoring configurations
```

## 🐳 Docker Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Services
- **nfl-trading-app**: Main application
- **postgres**: Database
- **redis**: Caching layer
- **jupyter**: Analysis notebooks
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards

## 🔧 Configuration

Configuration is managed through YAML files and environment variables:

- `configs/dev.yaml`: Development settings
- `configs/prod.yaml`: Production settings
- `.env`: Environment variables (API keys, secrets)

Key configuration sections:
- **Data paths**: Input/output directories
- **API settings**: Kalshi and NFL API configuration
- **Model parameters**: ML hyperparameters
- **Processing**: Batch sizes, worker counts
- **Monitoring**: Metrics and logging

## 🧪 Testing

```bash
# Run all tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# With coverage
make test-coverage
```

Test categories:
- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end workflows
- **Performance tests**: Benchmarking and load testing

## 📈 Monitoring

The system includes comprehensive monitoring:

- **Application Metrics**: Custom business metrics via Prometheus
- **System Metrics**: CPU, memory, disk usage
- **Database Metrics**: Query performance, connection pools
- **Trading Metrics**: Signal accuracy, P&L tracking
- **Grafana Dashboards**: Visual monitoring interface

Access monitoring at:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## 🔐 Security

Security measures implemented:
- Environment variable management for secrets
- Input validation and sanitization
- Rate limiting for API calls
- Database access controls
- Docker security best practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves risk and past performance does not guarantee future results. Users are responsible for complying with applicable laws and regulations.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/nfl-trading-system/issues)
- **Documentation**: [Wiki](https://github.com/your-org/nfl-trading-system/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/nfl-trading-system/discussions)

## 🙏 Acknowledgments

- NFL for providing play-by-play data
- Kalshi for market data APIs
- OpenAI for development assistance
- The open-source community