"""Game state feature extractor for NFL play-by-play data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import get_config, get_logger


logger = get_logger(__name__)


class GameSituation(Enum):
    """Game situation categories."""
    BLOWOUT_AHEAD = "blowout_ahead"      # >14 point lead
    COMFORTABLE_AHEAD = "comfortable_ahead"  # 8-14 point lead
    SLIGHT_AHEAD = "slight_ahead"        # 1-7 point lead
    TIED = "tied"                        # 0 point difference
    SLIGHT_BEHIND = "slight_behind"      # 1-7 point deficit
    COMFORTABLE_BEHIND = "comfortable_behind"  # 8-14 point deficit
    BLOWOUT_BEHIND = "blowout_behind"    # >14 point deficit


class FieldZone(Enum):
    """Field position zones."""
    OWN_ENDZONE = "own_endzone"          # 0-10 yard line
    OWN_TERRITORY = "own_territory"      # 11-49 yard line
    MIDFIELD = "midfield"                # 50 yard line
    OPPONENT_TERRITORY = "opponent_territory"  # 51-79 yard line
    RED_ZONE = "red_zone"                # 80-100 yard line


@dataclass
class DriveContext:
    """Context information for a drive."""
    drive_id: str
    plays: List[Dict[str, Any]]
    starting_field_position: int
    starting_score_diff: int
    ending_score_diff: int
    total_yards: int
    total_plays: int
    time_of_possession: float
    result: str  # touchdown, field_goal, punt, turnover, etc.


class GameStateExtractor:
    """Extracts game state features from NFL play-by-play data."""

    def __init__(self, config=None):
        """Initialize the game state extractor.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.GameStateExtractor")

        # Feature configuration
        self.lookback_window = 5  # Number of plays to look back for momentum
        self.drive_lookback = 3   # Number of drives to look back
        
        # Thresholds
        self.red_zone_threshold = 20  # Yards from endzone
        self.long_play_threshold = 20  # Yards for "big play"
        self.short_yardage_threshold = 3  # Yards for short yardage situation

    def extract_features(self, plays_df: pd.DataFrame, team_focus: Optional[str] = None) -> pd.DataFrame:
        """Extract comprehensive game state features from play-by-play data.

        Args:
            plays_df: DataFrame with play-by-play data
            team_focus: Team to focus analysis on (optional)

        Returns:
            DataFrame with extracted features
        """
        try:
            self.logger.info(f"Extracting game state features from {len(plays_df)} plays")
            
            # Ensure required columns exist
            plays_df = self._validate_plays_data(plays_df)
            
            # Sort by timestamp
            plays_df = plays_df.sort_values('timestamp').reset_index(drop=True)
            
            # Extract base features
            features_df = self._extract_base_features(plays_df)
            
            # Add drive-level features
            features_df = self._add_drive_features(features_df, plays_df)
            
            # Add momentum features
            features_df = self._add_momentum_features(features_df, plays_df)
            
            # Add situational features
            features_df = self._add_situational_features(features_df, plays_df)
            
            # Add time-based features
            features_df = self._add_time_features(features_df, plays_df)
            
            # Add team-specific features if focus team specified
            if team_focus:
                features_df = self._add_team_specific_features(features_df, plays_df, team_focus)
            
            self.logger.info(f"Extracted {len(features_df.columns)} features from game state data")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error extracting game state features: {e}")
            raise

    def _validate_plays_data(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean plays data."""
        required_cols = ['timestamp', 'play_type', 'possession_team', 'field_position', 
                        'score_home', 'score_away', 'quarter', 'time_remaining']
        
        missing_cols = [col for col in required_cols if col not in plays_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        plays_df = plays_df.copy()
        
        # Fill missing values
        plays_df['down'] = plays_df['down'].fillna(0)
        plays_df['distance'] = plays_df['distance'].fillna(0)
        plays_df['yards_gained'] = plays_df['yards_gained'].fillna(0)
        plays_df['field_position'] = plays_df['field_position'].fillna(50)
        
        return plays_df

    def _extract_base_features(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Extract base features for each play."""
        features = []
        
        for idx, play in plays_df.iterrows():
            base_features = {
                'timestamp': play['timestamp'],
                'play_id': play.get('play_id', idx),
                
                # Score situation
                'score_differential': play['score_home'] - play['score_away'],
                'absolute_score_diff': abs(play['score_home'] - play['score_away']),
                'home_score': play['score_home'],
                'away_score': play['score_away'],
                'total_score': play['score_home'] + play['score_away'],
                
                # Field position
                'field_position': play['field_position'],
                'distance_to_endzone': 100 - play['field_position'],
                'distance_to_own_endzone': play['field_position'],
                
                # Down and distance
                'down': play['down'],
                'distance': play['distance'],
                'yards_to_go': play['distance'],
                
                # Time context
                'quarter': play['quarter'],
                'time_remaining': play['time_remaining'],
                'game_time_elapsed': (4 * 15 * 60) - play['time_remaining'],  # Assuming 15min quarters
                
                # Play characteristics
                'play_type': play['play_type'],
                'yards_gained': play['yards_gained'],
                'possession_team': play['possession_team']
            }
            
            # Add derived categorical features
            base_features.update(self._get_categorical_features(play))
            
            features.append(base_features)
        
        return pd.DataFrame(features)

    def _get_categorical_features(self, play: pd.Series) -> Dict[str, Any]:
        """Get categorical features for a single play."""
        features = {}
        
        # Game situation based on score
        score_diff = play['score_home'] - play['score_away']
        if score_diff > 14:
            features['game_situation'] = GameSituation.BLOWOUT_AHEAD.value
        elif score_diff > 7:
            features['game_situation'] = GameSituation.COMFORTABLE_AHEAD.value
        elif score_diff > 0:
            features['game_situation'] = GameSituation.SLIGHT_AHEAD.value
        elif score_diff == 0:
            features['game_situation'] = GameSituation.TIED.value
        elif score_diff > -8:
            features['game_situation'] = GameSituation.SLIGHT_BEHIND.value
        elif score_diff > -15:
            features['game_situation'] = GameSituation.COMFORTABLE_BEHIND.value
        else:
            features['game_situation'] = GameSituation.BLOWOUT_BEHIND.value
        
        # Field zone
        field_pos = play['field_position']
        if field_pos <= 10:
            features['field_zone'] = FieldZone.OWN_ENDZONE.value
        elif field_pos < 50:
            features['field_zone'] = FieldZone.OWN_TERRITORY.value
        elif field_pos == 50:
            features['field_zone'] = FieldZone.MIDFIELD.value
        elif field_pos < 80:
            features['field_zone'] = FieldZone.OPPONENT_TERRITORY.value
        else:
            features['field_zone'] = FieldZone.RED_ZONE.value
        
        # Down situation
        down = play['down']
        distance = play['distance']
        
        if down == 1:
            features['down_situation'] = 'first_down'
        elif down == 2:
            if distance <= 3:
                features['down_situation'] = 'second_short'
            elif distance <= 7:
                features['down_situation'] = 'second_medium'
            else:
                features['down_situation'] = 'second_long'
        elif down == 3:
            if distance <= 3:
                features['down_situation'] = 'third_short'
            elif distance <= 7:
                features['down_situation'] = 'third_medium'
            else:
                features['down_situation'] = 'third_long'
        elif down == 4:
            features['down_situation'] = 'fourth_down'
        else:
            features['down_situation'] = 'other'
        
        # Red zone flag
        features['in_red_zone'] = (100 - field_pos) <= self.red_zone_threshold
        
        # Two minute warning situations
        features['two_minute_warning'] = play['time_remaining'] <= 120
        features['fourth_quarter'] = play['quarter'] == 4
        features['overtime'] = play['quarter'] > 4
        
        # Critical situations
        features['critical_situation'] = (
            (play['quarter'] >= 4 and play['time_remaining'] <= 300) or  # Last 5 min
            (play['quarter'] > 4)  # Overtime
        )
        
        return features

    def _add_drive_features(self, features_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Add drive-level features."""
        # Group plays by drive (simplified - using possession changes)
        drives = self._identify_drives(plays_df)
        
        # Calculate drive metrics
        for idx, row in features_df.iterrows():
            play_timestamp = row['timestamp']
            
            # Find current drive
            current_drive = self._find_current_drive(drives, play_timestamp)
            
            if current_drive:
                drive_features = self._calculate_drive_features(current_drive, plays_df)
                
                # Add drive features to main dataframe
                for key, value in drive_features.items():
                    features_df.loc[idx, f'drive_{key}'] = value
        
        return features_df

    def _identify_drives(self, plays_df: pd.DataFrame) -> List[DriveContext]:
        """Identify drives from play sequence."""
        drives = []
        current_drive_plays = []
        current_possession = None
        drive_id = 0
        
        for idx, play in plays_df.iterrows():
            possession = play['possession_team']
            
            # Check for possession change
            if current_possession is None:
                current_possession = possession
                current_drive_plays = [play.to_dict()]
            elif possession != current_possession:
                # Drive ended, create drive context
                if current_drive_plays:
                    drive = self._create_drive_context(drive_id, current_drive_plays)
                    drives.append(drive)
                    drive_id += 1
                
                # Start new drive
                current_possession = possession
                current_drive_plays = [play.to_dict()]
            else:
                current_drive_plays.append(play.to_dict())
        
        # Add final drive
        if current_drive_plays:
            drive = self._create_drive_context(drive_id, current_drive_plays)
            drives.append(drive)
        
        return drives

    def _create_drive_context(self, drive_id: int, plays: List[Dict[str, Any]]) -> DriveContext:
        """Create drive context from list of plays."""
        if not plays:
            return None
        
        first_play = plays[0]
        last_play = plays[-1]
        
        total_yards = sum(p.get('yards_gained', 0) for p in plays)
        
        # Calculate time of possession (simplified)
        time_diff = 0
        if len(plays) > 1:
            first_time = pd.to_datetime(first_play['timestamp'])
            last_time = pd.to_datetime(last_play['timestamp'])
            time_diff = (last_time - first_time).total_seconds()
        
        # Determine drive result (simplified)
        last_play_type = last_play.get('play_type', 'unknown')
        if 'touchdown' in last_play_type.lower():
            result = 'touchdown'
        elif 'field_goal' in last_play_type.lower():
            result = 'field_goal'
        elif 'punt' in last_play_type.lower():
            result = 'punt'
        elif 'turnover' in last_play_type.lower() or 'interception' in last_play_type.lower():
            result = 'turnover'
        else:
            result = 'other'
        
        return DriveContext(
            drive_id=f"drive_{drive_id}",
            plays=plays,
            starting_field_position=first_play.get('field_position', 50),
            starting_score_diff=first_play.get('score_home', 0) - first_play.get('score_away', 0),
            ending_score_diff=last_play.get('score_home', 0) - last_play.get('score_away', 0),
            total_yards=total_yards,
            total_plays=len(plays),
            time_of_possession=time_diff,
            result=result
        )

    def _find_current_drive(self, drives: List[DriveContext], timestamp: datetime) -> Optional[DriveContext]:
        """Find the drive that contains the given timestamp."""
        timestamp = pd.to_datetime(timestamp)
        
        for drive in drives:
            if not drive.plays:
                continue
            
            first_play_time = pd.to_datetime(drive.plays[0]['timestamp'])
            last_play_time = pd.to_datetime(drive.plays[-1]['timestamp'])
            
            if first_play_time <= timestamp <= last_play_time:
                return drive
        
        return None

    def _calculate_drive_features(self, drive: DriveContext, plays_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate features for a specific drive."""
        features = {
            'plays_in_drive': drive.total_plays,
            'yards_in_drive': drive.total_yards,
            'avg_yards_per_play': drive.total_yards / max(drive.total_plays, 1),
            'starting_field_position': drive.starting_field_position,
            'drive_result': drive.result,
            'score_change_in_drive': drive.ending_score_diff - drive.starting_score_diff,
            'time_of_possession': drive.time_of_possession
        }
        
        # First down conversions in drive
        first_downs = sum(1 for p in drive.plays if p.get('yards_gained', 0) >= p.get('distance', 0))
        features['first_downs_in_drive'] = first_downs
        features['first_down_conversion_rate'] = first_downs / max(drive.total_plays, 1)
        
        # Big plays in drive
        big_plays = sum(1 for p in drive.plays if p.get('yards_gained', 0) >= self.long_play_threshold)
        features['big_plays_in_drive'] = big_plays
        
        return features

    def _add_momentum_features(self, features_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features using lookback window."""
        for idx in range(len(features_df)):
            start_idx = max(0, idx - self.lookback_window)
            recent_plays = plays_df.iloc[start_idx:idx+1]
            
            momentum_features = self._calculate_momentum_features(recent_plays, plays_df.iloc[idx])
            
            # Add momentum features
            for key, value in momentum_features.items():
                features_df.loc[idx, f'momentum_{key}'] = value
        
        return features_df

    def _calculate_momentum_features(self, recent_plays: pd.DataFrame, current_play: pd.Series) -> Dict[str, Any]:
        """Calculate momentum features from recent plays."""
        if len(recent_plays) == 0:
            return {key: 0.0 for key in ['yards_trend', 'success_rate', 'big_play_rate', 
                                       'score_trend', 'field_position_trend']}
        
        features = {}
        
        # Yards gained trend
        yards = recent_plays['yards_gained'].fillna(0)
        features['yards_trend'] = yards.mean()
        features['yards_per_play_recent'] = yards.sum() / len(recent_plays)
        
        # Success rate (simplified: positive yards gained)
        successful_plays = (yards > 0).sum()
        features['success_rate'] = successful_plays / len(recent_plays)
        
        # Big play rate
        big_plays = (yards >= self.long_play_threshold).sum()
        features['big_play_rate'] = big_plays / len(recent_plays)
        
        # Score differential trend
        if len(recent_plays) > 1:
            score_diffs = recent_plays['score_home'] - recent_plays['score_away']
            features['score_trend'] = score_diffs.iloc[-1] - score_diffs.iloc[0]
        else:
            features['score_trend'] = 0.0
        
        # Field position trend (positive means moving toward opponent endzone)
        if len(recent_plays) > 1:
            field_positions = recent_plays['field_position']
            features['field_position_trend'] = field_positions.iloc[-1] - field_positions.iloc[0]
        else:
            features['field_position_trend'] = 0.0
        
        # Consecutive successful plays
        consecutive_success = 0
        for i in range(len(recent_plays) - 1, -1, -1):
            if recent_plays.iloc[i]['yards_gained'] > 0:
                consecutive_success += 1
            else:
                break
        features['consecutive_success'] = consecutive_success
        
        return features

    def _add_situational_features(self, features_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Add situational awareness features."""
        for idx, row in features_df.iterrows():
            situational_features = self._calculate_situational_features(plays_df.iloc[idx], plays_df)
            
            # Add situational features
            for key, value in situational_features.items():
                features_df.loc[idx, f'situation_{key}'] = value
        
        return features_df

    def _calculate_situational_features(self, current_play: pd.Series, all_plays: pd.DataFrame) -> Dict[str, Any]:
        """Calculate situational awareness features."""
        features = {}
        
        # Goal line situation
        distance_to_endzone = 100 - current_play['field_position']
        features['goal_line_situation'] = distance_to_endzone <= 5
        features['red_zone_situation'] = distance_to_endzone <= self.red_zone_threshold
        
        # Short yardage situation
        features['short_yardage'] = current_play['distance'] <= self.short_yardage_threshold
        
        # Passing vs rushing situation
        down = current_play['down']
        distance = current_play['distance']
        
        # Typical passing situations
        features['passing_situation'] = (
            (down == 2 and distance > 7) or
            (down == 3 and distance > 3) or
            (current_play['time_remaining'] < 120 and current_play['quarter'] >= 4)
        )
        
        # Two minute drill
        features['two_minute_drill'] = (
            current_play['time_remaining'] <= 120 and 
            current_play['quarter'] >= 4
        )
        
        # Garbage time (large score differential late in game)
        features['garbage_time'] = (
            abs(current_play['score_home'] - current_play['score_away']) > 21 and
            current_play['quarter'] == 4 and
            current_play['time_remaining'] < 600  # Less than 10 minutes
        )
        
        # Comeback situation
        score_diff = current_play['score_home'] - current_play['score_away']
        features['comeback_situation'] = (
            score_diff < -7 and
            current_play['quarter'] >= 3
        )
        
        # Clock management situation
        features['clock_management'] = (
            current_play['quarter'] == 4 and
            current_play['time_remaining'] < 300  # Last 5 minutes
        )
        
        return features

    def _add_time_features(self, features_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based contextual features."""
        for idx, row in features_df.iterrows():
            time_features = self._calculate_time_features(plays_df.iloc[idx])
            
            # Add time features
            for key, value in time_features.items():
                features_df.loc[idx, f'time_{key}'] = value
        
        return features_df

    def _calculate_time_features(self, play: pd.Series) -> Dict[str, Any]:
        """Calculate time-based features."""
        features = {}
        
        quarter = play['quarter']
        time_remaining = play['time_remaining']
        
        # Game phase
        total_game_time = 4 * 15 * 60  # 60 minutes in seconds
        elapsed_time = total_game_time - time_remaining
        
        features['game_progress'] = elapsed_time / total_game_time
        features['time_remaining_pct'] = time_remaining / total_game_time
        
        # Quarter-specific features
        features['first_half'] = quarter <= 2
        features['second_half'] = quarter > 2
        features['final_quarter'] = quarter == 4
        features['overtime'] = quarter > 4
        
        # Time pressure indicators
        features['hurry_up_situation'] = (
            quarter >= 4 and time_remaining <= 120
        ) or (
            quarter == 2 and time_remaining <= 120
        )
        
        # End of quarter/half situations
        features['end_of_quarter'] = time_remaining % (15 * 60) <= 120  # Last 2 minutes of quarter
        features['end_of_half'] = (quarter == 2 or quarter == 4) and time_remaining <= 120
        
        # Time advantage/disadvantage
        if quarter >= 4:
            features['time_advantage'] = time_remaining > 300  # More than 5 minutes
            features['time_disadvantage'] = time_remaining <= 120  # Less than 2 minutes
        else:
            features['time_advantage'] = False
            features['time_disadvantage'] = False
        
        return features

    def _add_team_specific_features(self, features_df: pd.DataFrame, plays_df: pd.DataFrame, 
                                  team_focus: str) -> pd.DataFrame:
        """Add team-specific features focusing on a particular team."""
        for idx, row in features_df.iterrows():
            team_features = self._calculate_team_features(plays_df.iloc[idx], team_focus, plays_df)
            
            # Add team features
            for key, value in team_features.items():
                features_df.loc[idx, f'team_{key}'] = value
        
        return features_df

    def _calculate_team_features(self, current_play: pd.Series, focus_team: str, 
                               all_plays: pd.DataFrame) -> Dict[str, Any]:
        """Calculate team-specific features."""
        features = {}
        
        # Team possession
        features['has_possession'] = current_play['possession_team'] == focus_team
        
        # Score from team perspective
        if focus_team == current_play.get('home_team'):
            team_score = current_play['score_home']
            opponent_score = current_play['score_away']
        else:
            team_score = current_play['score_away']
            opponent_score = current_play['score_home']
        
        features['team_score'] = team_score
        features['opponent_score'] = opponent_score
        features['team_score_differential'] = team_score - opponent_score
        features['team_winning'] = team_score > opponent_score
        features['team_losing'] = team_score < opponent_score
        
        # Field position from team perspective
        if features['has_possession']:
            features['team_field_position'] = current_play['field_position']
            features['team_distance_to_endzone'] = 100 - current_play['field_position']
        else:
            features['team_field_position'] = 100 - current_play['field_position']
            features['team_distance_to_endzone'] = current_play['field_position']
        
        return features

    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """Get mapping of feature names to their descriptions."""
        return {
            # Base features
            'score_differential': 'Home team score minus away team score',
            'absolute_score_diff': 'Absolute difference in scores',
            'field_position': 'Current field position (0-100 yard line)',
            'distance_to_endzone': 'Yards to opponent endzone',
            'down': 'Current down (1-4)',
            'distance': 'Yards needed for first down',
            'quarter': 'Current quarter',
            'time_remaining': 'Seconds remaining in game',
            
            # Categorical features
            'game_situation': 'Game situation based on score differential',
            'field_zone': 'Field zone (own territory, red zone, etc.)',
            'down_situation': 'Down and distance situation',
            'in_red_zone': 'Whether team is in red zone',
            'critical_situation': 'Late game critical situation',
            
            # Drive features
            'drive_plays_in_drive': 'Number of plays in current drive',
            'drive_yards_in_drive': 'Total yards gained in current drive',
            'drive_avg_yards_per_play': 'Average yards per play in drive',
            'drive_first_down_conversion_rate': 'First down conversion rate in drive',
            
            # Momentum features
            'momentum_yards_trend': 'Average yards gained in recent plays',
            'momentum_success_rate': 'Success rate of recent plays',
            'momentum_big_play_rate': 'Rate of big plays in recent plays',
            'momentum_consecutive_success': 'Number of consecutive successful plays',
            
            # Situational features
            'situation_goal_line_situation': 'Team is near goal line',
            'situation_short_yardage': 'Short yardage situation',
            'situation_passing_situation': 'Typical passing down/distance',
            'situation_two_minute_drill': 'Two minute warning situation',
            'situation_comeback_situation': 'Team is in comeback situation',
            
            # Time features
            'time_game_progress': 'Percentage of game completed',
            'time_final_quarter': 'Currently in final quarter',
            'time_hurry_up_situation': 'Time pressure situation',
            'time_end_of_half': 'End of half situation'
        }