"""NFL data processor for play-by-play JSON data."""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pydantic import BaseModel, validator

from ..config import get_config, get_logger


logger = get_logger(__name__)


@dataclass
class PlayEvent:
    """Represents a single NFL play event."""
    timestamp: datetime
    game_id: str
    play_id: str
    play_type: str
    down: Optional[int]
    distance: Optional[int]
    field_position: Optional[int]
    score_home: int
    score_away: int
    time_remaining: int
    quarter: int
    possession_team: str
    home_team: str
    away_team: str
    description: str
    result: Optional[str]
    yards_gained: Optional[int]
    formation: Optional[str]
    personnel: Optional[str]
    weather_conditions: Optional[Dict[str, Any]]
    momentum_score: Optional[float] = None


class PlayEventValidator(BaseModel):
    """Pydantic validator for play events."""
    timestamp: datetime
    game_id: str
    play_id: str
    play_type: str
    down: Optional[int] = None
    distance: Optional[int] = None
    field_position: Optional[int] = None
    score_home: int
    score_away: int
    time_remaining: int
    quarter: int
    possession_team: str
    home_team: str
    away_team: str
    description: str
    result: Optional[str] = None
    yards_gained: Optional[int] = None
    formation: Optional[str] = None
    personnel: Optional[str] = None
    weather_conditions: Optional[Dict[str, Any]] = None

    @validator('timestamp')
    def validate_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    @validator('quarter')
    def validate_quarter(cls, v):
        if v < 1 or v > 5:  # Including overtime
            raise ValueError(f"Invalid quarter: {v}")
        return v

    @validator('down')
    def validate_down(cls, v):
        if v is not None and (v < 1 or v > 4):
            raise ValueError(f"Invalid down: {v}")
        return v

    @validator('field_position')
    def validate_field_position(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError(f"Invalid field position: {v}")
        return v


class NFLDataProcessor:
    """Processes NFL play-by-play data from JSON format."""

    def __init__(self, config=None):
        """Initialize the NFL data processor.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.NFLDataProcessor")

        # Feature extractors
        self.momentum_features = [
            'score_differential', 'field_position_advantage',
            'recent_plays_success', 'time_pressure'
        ]

    def load_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load NFL play-by-play data from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"NFL data file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"Successfully loaded NFL data from {file_path}")
            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading NFL data file {file_path}: {e}")
            raise

    def parse_play_events(self, raw_data: Dict[str, Any]) -> List[PlayEvent]:
        """Parse raw JSON data into structured play events.

        Args:
            raw_data: Raw JSON data from NFL API

        Returns:
            List of PlayEvent objects

        Raises:
            ValueError: If data structure is invalid
        """
        try:
            plays = []

            # Extract game metadata
            game_info = raw_data.get('game', {})
            game_id = game_info.get('id', 'unknown')
            home_team = game_info.get('home_team', 'UNK')
            away_team = game_info.get('away_team', 'UNK')

            # Process each play
            play_data = raw_data.get('plays', [])

            for play_json in play_data:
                try:
                    play_event = self._parse_single_play(
                        play_json, game_id, home_team, away_team
                    )
                    if play_event:
                        plays.append(play_event)

                except Exception as e:
                    self.logger.warning(f"Failed to parse play {play_json.get('id', 'unknown')}: {e}")
                    continue

            self.logger.info(f"Successfully parsed {len(plays)} play events from game {game_id}")
            return plays

        except Exception as e:
            self.logger.error(f"Error parsing play events: {e}")
            raise ValueError(f"Invalid play data structure: {e}")

    def _parse_single_play(
        self,
        play_json: Dict[str, Any],
        game_id: str,
        home_team: str,
        away_team: str
    ) -> Optional[PlayEvent]:
        """Parse a single play from JSON data.

        Args:
            play_json: Single play JSON data
            game_id: Game identifier
            home_team: Home team code
            away_team: Away team code

        Returns:
            PlayEvent object or None if invalid
        """
        try:
            # Validate required fields
            validator_data = PlayEventValidator(
                timestamp=play_json['timestamp'],
                game_id=game_id,
                play_id=play_json['id'],
                play_type=play_json['type'],
                down=play_json.get('down'),
                distance=play_json.get('distance'),
                field_position=play_json.get('field_position'),
                score_home=play_json.get('score_home', 0),
                score_away=play_json.get('score_away', 0),
                time_remaining=play_json.get('time_remaining', 0),
                quarter=play_json['quarter'],
                possession_team=play_json['possession_team'],
                home_team=home_team,
                away_team=away_team,
                description=play_json.get('description', ''),
                result=play_json.get('result'),
                yards_gained=play_json.get('yards_gained'),
                formation=play_json.get('formation'),
                personnel=play_json.get('personnel'),
                weather_conditions=play_json.get('weather')
            )

            return PlayEvent(**validator_data.dict())

        except Exception as e:
            self.logger.debug(f"Failed to parse play: {e}")
            return None

    def calculate_momentum_features(self, plays: List[PlayEvent]) -> List[PlayEvent]:
        """Calculate momentum-based features for each play.

        Args:
            plays: List of PlayEvent objects

        Returns:
            List of PlayEvent objects with momentum scores
        """
        for i, play in enumerate(plays):
            try:
                momentum_score = self._calculate_momentum_score(plays, i)
                play.momentum_score = momentum_score

            except Exception as e:
                self.logger.warning(f"Failed to calculate momentum for play {play.play_id}: {e}")
                play.momentum_score = 0.0

        return plays

    def _calculate_momentum_score(self, plays: List[PlayEvent], current_index: int) -> float:
        """Calculate momentum score for a specific play.

        Args:
            plays: List of all plays
            current_index: Index of current play

        Returns:
            Momentum score (-1.0 to 1.0)
        """
        current_play = plays[current_index]

        # Look at last 5 plays for momentum calculation
        window_size = min(5, current_index)
        recent_plays = plays[max(0, current_index - window_size):current_index]

        if not recent_plays:
            return 0.0

        # Calculate various momentum factors
        score_diff = current_play.score_home - current_play.score_away
        field_position_advantage = (current_play.field_position or 50) - 50

        # Recent success rate
        successful_plays = sum(
            1 for play in recent_plays
            if play.yards_gained and play.yards_gained > 0
        )
        success_rate = successful_plays / len(recent_plays) if recent_plays else 0.5

        # Time pressure (higher in 4th quarter or overtime)
        time_pressure = 1.0 if current_play.quarter >= 4 else 0.5

        # Combine factors
        momentum_score = (
            (score_diff / 21.0) * 0.3 +  # Normalize score difference
            (field_position_advantage / 50.0) * 0.2 +  # Field position
            (success_rate - 0.5) * 2.0 * 0.3 +  # Recent success
            (time_pressure - 0.5) * 0.2  # Time pressure
        )

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, momentum_score))

    def to_dataframe(self, plays: List[PlayEvent]) -> pd.DataFrame:
        """Convert play events to pandas DataFrame.

        Args:
            plays: List of PlayEvent objects

        Returns:
            Pandas DataFrame
        """
        if not plays:
            return pd.DataFrame()

        # Convert to dictionaries
        play_dicts = []
        for play in plays:
            play_dict = {
                'timestamp': play.timestamp,
                'game_id': play.game_id,
                'play_id': play.play_id,
                'play_type': play.play_type,
                'down': play.down,
                'distance': play.distance,
                'field_position': play.field_position,
                'score_home': play.score_home,
                'score_away': play.score_away,
                'score_differential': play.score_home - play.score_away,
                'time_remaining': play.time_remaining,
                'quarter': play.quarter,
                'possession_team': play.possession_team,
                'home_team': play.home_team,
                'away_team': play.away_team,
                'description': play.description,
                'result': play.result,
                'yards_gained': play.yards_gained,
                'formation': play.formation,
                'personnel': play.personnel,
                'momentum_score': play.momentum_score
            }

            # Add weather conditions as separate columns if present
            if play.weather_conditions:
                for key, value in play.weather_conditions.items():
                    play_dict[f'weather_{key}'] = value

            play_dicts.append(play_dict)

        df = pd.DataFrame(play_dicts)

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.logger.info(f"Created DataFrame with {len(df)} play events")
        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: Union[str, Path]):
        """Save processed data to file.

        Args:
            df: Processed DataFrame
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            df.to_json(output_path, orient='records', date_format='iso', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        self.logger.info(f"Saved processed NFL data to {output_path}")

    def process_file(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
        """Process a single NFL data file end-to-end.

        Args:
            input_path: Input JSON file path
            output_path: Output file path

        Returns:
            Processed DataFrame
        """
        try:
            # Load and parse data
            raw_data = self.load_json_file(input_path)
            plays = self.parse_play_events(raw_data)

            # Calculate momentum features
            plays = self.calculate_momentum_features(plays)

            # Convert to DataFrame
            df = self.to_dataframe(plays)

            # Save processed data
            self.save_processed_data(df, output_path)

            return df

        except Exception as e:
            self.logger.error(f"Failed to process NFL file {input_path}: {e}")
            raise

    def process_directory(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> List[pd.DataFrame]:
        """Process all JSON files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path

        Returns:
            List of processed DataFrames
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(input_dir.glob('*.json'))

        if not json_files:
            self.logger.warning(f"No JSON files found in {input_dir}")
            return []

        dataframes = []

        for json_file in json_files:
            try:
                output_file = output_dir / f"{json_file.stem}_processed.parquet"
                df = self.process_file(json_file, output_file)
                dataframes.append(df)

            except Exception as e:
                self.logger.error(f"Failed to process {json_file}: {e}")
                continue

        self.logger.info(f"Successfully processed {len(dataframes)}/{len(json_files)} NFL data files")
        return dataframes