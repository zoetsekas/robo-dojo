"""
Ray RLLib Callbacks for Curriculum Learning and Self-Play.

These callbacks integrate the curriculum and self-play systems with
Ray RLLib's training loop, handling:
- Curriculum progression checks
- Policy snapshot creation
- Opponent selection for self-play
- Metrics logging

Updated for Ray 2.x+ RLlibCallback API.
"""
import os
import logging
from typing import Dict, Any, Optional

from ray.rllib.callbacks.callbacks import RLlibCallback

from .curriculum import TrainingCurriculum, TrainingPhase
from .self_play import PolicyLeague, OpponentSampler

# Setup logging
logger = logging.getLogger(__name__)


class CurriculumCallback(RLlibCallback):
    """
    RLLib callback for curriculum-based training.
    
    Tracks win rates and progresses through curriculum stages based on
    performance milestones.
    """
    
    def __init__(self):
        super().__init__()
        self.curriculum: Optional[TrainingCurriculum] = None
        self.episode_outcomes: list = []  # Track recent outcomes for win rate
        self.win_rate_window = 100  # Rolling window for win rate calculation
    
    def on_algorithm_init(self, *, algorithm, metrics_logger, **kwargs) -> None:
        """Initialize curriculum when algorithm starts."""
        # Check for existing checkpoint
        checkpoint_path = "artifacts/curriculum/curriculum_state.json"
        if os.path.exists(checkpoint_path):
            logger.info("[CurriculumCallback] Loading existing curriculum checkpoint")
            self.curriculum = TrainingCurriculum.load_checkpoint(checkpoint_path)
        else:
            logger.info("[CurriculumCallback] Initializing new curriculum")
            self.curriculum = TrainingCurriculum.default()
        
        logger.info(f"[CurriculumCallback] Starting at stage: {self.curriculum.current_stage.name}")
    
    def on_episode_end(
        self,
        *,
        episode,
        metrics_logger=None,
        **kwargs
    ) -> None:
        """Track episode outcomes and detailed game metrics for TensorBoard."""
        # Handle both old (total_reward) and new (get_return()) APIs
        if hasattr(episode, 'get_return'):
            episode_return = episode.get_return()
        else:
            episode_return = getattr(episode, 'total_reward', 0)
        
        if episode_return > 0.5:
            self.episode_outcomes.append(1)
        elif episode_return < -0.5:
            self.episode_outcomes.append(0)
        else:
            # Draw or unclear - count as half
            self.episode_outcomes.append(0.5)
        
        # Keep only recent outcomes
        if len(self.episode_outcomes) > self.win_rate_window:
            self.episode_outcomes = self.episode_outcomes[-self.win_rate_window:]
        
        # Extract detailed game metrics from episode info (if available)
        if metrics_logger:
            # Get last info dict from episode
            last_info = {}
            if hasattr(episode, 'get_infos'):
                infos = episode.get_infos()
                if infos:
                    last_info = infos[-1] if isinstance(infos, list) else infos
            elif hasattr(episode, 'last_info_for'):
                try:
                    last_info = episode.last_info_for() or {}
                except:
                    pass
            
            # Log combat metrics if available
            if "damage_dealt" in last_info:
                metrics_logger.log_value("game/damage_dealt", float(last_info["damage_dealt"]), reduce="mean")
            if "damage_taken" in last_info:
                metrics_logger.log_value("game/damage_taken", float(last_info["damage_taken"]), reduce="mean")
            if "bullets_fired" in last_info:
                metrics_logger.log_value("game/bullets_fired", float(last_info["bullets_fired"]), reduce="mean")
            if "hits_dealt" in last_info:
                metrics_logger.log_value("game/hits_dealt", float(last_info["hits_dealt"]), reduce="mean")
            if "energy_remaining" in last_info:
                metrics_logger.log_value("game/energy_remaining", float(last_info["energy_remaining"]), reduce="mean")
            if "episode_length" in last_info:
                metrics_logger.log_value("game/episode_length", float(last_info["episode_length"]), reduce="mean")
            
            # Calculate and log accuracy if we have both bullets and hits
            if last_info.get("bullets_fired", 0) > 0:
                accuracy = last_info.get("hits_dealt", 0) / last_info["bullets_fired"]
                metrics_logger.log_value("game/accuracy", accuracy, reduce="mean")
    
    def on_train_result(self, *, algorithm, metrics_logger, result: Dict[str, Any], **kwargs) -> None:
        """Check curriculum progression after each training iteration."""
        if not self.curriculum:
            return
        
        # Calculate current win rate
        if self.episode_outcomes:
            win_rate = sum(self.episode_outcomes) / len(self.episode_outcomes)
        else:
            win_rate = 0.0
        
        # Get reward from new API location
        reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0.0)
        if reward_mean == 0.0:
            reward_mean = result.get("episode_reward_mean", 0.0)
        
        # Prepare metrics
        metrics = {
            "win_rate": win_rate,
            "reward_mean": reward_mean,
            "episodes_total": result.get("episodes_total", 0),
        }
        
        # Check for progression
        if self.curriculum.check_progression(metrics):
            logger.info(f"[CurriculumCallback] Advanced to: {self.curriculum.current_stage.name}")
            # Reset outcome tracking for new stage
            self.episode_outcomes = []
        
        # Log curriculum metrics
        if metrics_logger:
            curriculum_metrics = self.curriculum.get_metrics_for_logging()
            curriculum_metrics["curriculum/win_rate"] = win_rate
            for key, value in curriculum_metrics.items():
                if isinstance(value, (int, float, bool)):
                    metrics_logger.log_value(key, float(value), reduce="mean")
                else:
                    # For non-numeric values (like stage_name), log as info or skip
                    # RLlib's MetricsLogger.log_value with mean reduction requires scalars.
                    pass
    
    def get_current_opponent_config(self) -> Dict[str, Any]:
        """Get the current opponent configuration from curriculum."""
        if self.curriculum:
            return self.curriculum.get_opponent_config()
        return {"type": "sample_bot", "script": "simple_target.py"}


class SelfPlayCallback(RLlibCallback):
    """
    RLLib callback for league-based self-play training.
    
    Manages policy snapshots and opponent selection during self-play phases.
    """
    
    def __init__(
        self,
        snapshot_interval: int = 500,
        league_max_size: int = 20,
    ):
        super().__init__()
        self.snapshot_interval = snapshot_interval
        self.league: Optional[PolicyLeague] = None
        self.sampler: Optional[OpponentSampler] = None
        self.league_max_size = league_max_size
        self.current_main_iteration = 0
    
    def on_algorithm_init(self, *, algorithm, metrics_logger, **kwargs) -> None:
        """Initialize or load the policy league."""
        checkpoint_path = "artifacts/policy_league/league_state.pkl"
        
        if os.path.exists(checkpoint_path):
            logger.info("[SelfPlayCallback] Loading existing policy league")
            self.league = PolicyLeague.load_checkpoint(checkpoint_path)
        else:
            logger.info("[SelfPlayCallback] Initializing new policy league")
            self.league = PolicyLeague(max_size=self.league_max_size)
        
        self.sampler = OpponentSampler(league=self.league)
    
    def on_train_result(self, *, algorithm, metrics_logger, result: Dict[str, Any], **kwargs) -> None:
        """Create policy snapshots periodically."""
        if not self.league:
            return
        
        iteration = result.get("training_iteration", 0)
        self.current_main_iteration = iteration
        
        # Check if we should create a snapshot
        if iteration > 0 and iteration % self.snapshot_interval == 0:
            self._create_snapshot(algorithm, result)
        
        # Log league metrics
        if metrics_logger:
            league_metrics = self.league.get_metrics_for_logging()
            for key, value in league_metrics.items():
                metrics_logger.log_value(key, value, reduce="mean")
    
    def _create_snapshot(self, algorithm, result: Dict[str, Any]) -> None:
        """Create and store a policy snapshot."""
        iteration = result.get("training_iteration", 0)
        
        # Get weights from main policy
        try:
            policy = algorithm.get_policy("default_policy")
            if policy is None:
                policy = algorithm.get_policy("main")
            if policy is None:
                # Single-agent case
                policy = algorithm.get_policy()
            
            if policy:
                weights = policy.get_weights()
                reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0.0)
                metrics = {
                    "reward_mean": reward_mean,
                    "win_rate": 0.5,  # Default, will be updated from curriculum
                }
                
                self.league.add_snapshot(iteration, weights, metrics)
                
                # Periodically save league checkpoint
                if len(self.league) % 5 == 0:
                    self.league.save_checkpoint()
        except Exception as e:
            logger.error(f"[SelfPlayCallback] Failed to create snapshot: {e}")
    
    def get_opponent_weights(self, opponent_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get opponent weights for self-play."""
        if self.sampler and opponent_config.get("type") == "self_play":
            return self.sampler.get_opponent(opponent_config)
        return None
    
    def report_episode_result(self, main_won: bool) -> None:
        """Report episode result for ELO updates."""
        if self.sampler:
            self.sampler.report_match_result(main_won, self.current_main_iteration)


class CombinedTrainingCallback(RLlibCallback):
    """
    Combined callback that handles both curriculum and self-play.
    
    This is the recommended callback to use - it automatically delegates
    to curriculum or self-play callbacks based on the current training phase.
    """
    
    def __init__(
        self,
        snapshot_interval: int = 500,
        league_max_size: int = 20,
    ):
        super().__init__()
        self.curriculum_cb = CurriculumCallback()
        self.self_play_cb = SelfPlayCallback(
            snapshot_interval=snapshot_interval,
            league_max_size=league_max_size,
        )
    
    def on_algorithm_init(self, *, algorithm, metrics_logger, **kwargs) -> None:
        """Initialize both callbacks."""
        self.curriculum_cb.on_algorithm_init(
            algorithm=algorithm, 
            metrics_logger=metrics_logger, 
            **kwargs
        )
        self.self_play_cb.on_algorithm_init(
            algorithm=algorithm, 
            metrics_logger=metrics_logger, 
            **kwargs
        )
    
    def on_episode_created(self, *, episode, **kwargs) -> None:
        """Initialize episode custom data."""
        # Handle both old (user_data) and new (custom_data) APIs
        opponent_config = self.curriculum_cb.get_current_opponent_config()
        if hasattr(episode, 'custom_data'):
            episode.custom_data["opponent_config"] = opponent_config
        elif hasattr(episode, 'user_data'):
            episode.user_data["opponent_config"] = opponent_config
    
    def on_episode_end(
        self,
        *,
        episode,
        metrics_logger=None,
        **kwargs
    ) -> None:
        """Handle episode end for curriculum and self-play."""
        self.curriculum_cb.on_episode_end(
            episode=episode,
            metrics_logger=metrics_logger,
            **kwargs
        )
        
        # Report match result for ELO if self-play
        # Handle both old (user_data) and new (custom_data) APIs
        if hasattr(episode, 'custom_data'):
            opponent_config = episode.custom_data.get("opponent_config", {})
        elif hasattr(episode, 'user_data'):
            opponent_config = episode.user_data.get("opponent_config", {})
        else:
            opponent_config = {}
            
        if opponent_config.get("type") == "self_play":
            # Handle both get_return() and total_reward
            if hasattr(episode, 'get_return'):
                main_won = episode.get_return() > 0.5
            else:
                main_won = getattr(episode, 'total_reward', 0) > 0.5
            self.self_play_cb.report_episode_result(main_won)
    
    def on_train_result(self, *, algorithm, metrics_logger, result: Dict[str, Any], **kwargs) -> None:
        """Handle training result for curriculum and self-play."""
        self.curriculum_cb.on_train_result(
            algorithm=algorithm, 
            metrics_logger=metrics_logger, 
            result=result, 
            **kwargs
        )
        
        # Only create snapshots during self-play phase
        if self.curriculum_cb.curriculum:
            phase = self.curriculum_cb.curriculum.current_phase
            if phase in (TrainingPhase.SELF_PLAY, TrainingPhase.ADVANCED):
                self.self_play_cb.on_train_result(
                    algorithm=algorithm, 
                    metrics_logger=metrics_logger, 
                    result=result, 
                    **kwargs
                )
