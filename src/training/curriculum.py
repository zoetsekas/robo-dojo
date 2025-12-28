"""
Curriculum Learning Manager for RoboDojo.

Manages progression through training phases:
- Phase 1: Sample bot training (foundation skills)
- Phase 2: League-based self-play (emergent strategies)
- Phase 3: Advanced self-play with exploiters (robustness)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import os


class TrainingPhase(Enum):
    """Training phases in the curriculum."""
    FOUNDATION = 1      # Sample bot training
    SELF_PLAY = 2       # League-based self-play
    ADVANCED = 3        # Exploiter training


@dataclass
class CurriculumStage:
    """A single stage within a training phase."""
    name: str
    opponent: str           # Bot script name or "self_play"
    min_iterations: int     # Minimum iterations before checking milestone
    win_rate_milestone: float  # Win rate required to advance
    max_iterations: int = 0    # 0 = no max limit
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "opponent": self.opponent,
            "min_iterations": self.min_iterations,
            "win_rate_milestone": self.win_rate_milestone,
            "max_iterations": self.max_iterations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumStage":
        return cls(**data)


@dataclass
class TrainingCurriculum:
    """
    Manages curriculum progression through training phases.
    
    The curriculum defines a sequence of stages, each with:
    - An opponent (sample bot or self-play)
    - A minimum number of iterations
    - A win rate milestone to advance
    
    Example usage:
        curriculum = TrainingCurriculum.default()
        
        # In training loop:
        if curriculum.check_progression(metrics):
            print(f"Advancing to stage: {curriculum.current_stage.name}")
            
        opponent_config = curriculum.get_opponent_config()
    """
    stages: List[CurriculumStage] = field(default_factory=list)
    current_stage_idx: int = 0
    current_iteration: int = 0
    stage_start_iteration: int = 0
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    checkpoint_dir: str = "artifacts/curriculum"
    
    @classmethod
    def default(cls) -> "TrainingCurriculum":
        """Create curriculum with default Phase 1 stages."""
        stages = [
            # Phase 1: Foundation Skills
            CurriculumStage(
                name="1.1_stationary_target",
                opponent="noop_bot.py",
                min_iterations=500,
                win_rate_milestone=0.9,  # 90% threshold per user preference
                max_iterations=2000,
            ),
            CurriculumStage(
                name="1.2_simple_target",
                opponent="simple_target.py",
                min_iterations=2000,
                win_rate_milestone=0.9,  # 90% threshold
                max_iterations=10000,
            ),
            CurriculumStage(
                name="1.3_spinning_target",
                opponent="simple_spin.py",
                min_iterations=5000,
                win_rate_milestone=0.9,  # 90% threshold
                max_iterations=20000,
            ),
            # Phase 2: Self-Play
            CurriculumStage(
                name="2.1_self_play_basic",
                opponent="self_play",
                min_iterations=10000,
                win_rate_milestone=0.55,  # Against league, 50% is baseline
                max_iterations=0,  # No limit - continuous improvement
            ),
        ]
        return cls(stages=stages)
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get the current training stage."""
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return self.stages[-1]  # Stay on last stage
    
    @property
    def current_phase(self) -> TrainingPhase:
        """Determine current phase from stage name."""
        stage_name = self.current_stage.name
        if stage_name.startswith("1."):
            return TrainingPhase.FOUNDATION
        elif stage_name.startswith("2."):
            return TrainingPhase.SELF_PLAY
        else:
            return TrainingPhase.ADVANCED
    
    @property
    def iterations_in_stage(self) -> int:
        """Number of iterations completed in current stage."""
        return self.current_iteration - self.stage_start_iteration
    
    def check_progression(self, metrics: Dict[str, float]) -> bool:
        """
        Check if we should advance to the next curriculum stage.
        
        Args:
            metrics: Dictionary with at least 'win_rate' key
            
        Returns:
            True if stage advanced, False otherwise
        """
        self.current_iteration += 1
        self.metrics_history.append(metrics)
        
        stage = self.current_stage
        iters_in_stage = self.iterations_in_stage
        
        # Not enough iterations yet
        if iters_in_stage < stage.min_iterations:
            return False
        
        # Check win rate milestone
        win_rate = metrics.get("win_rate", 0.0)
        
        # Use rolling average of last 100 iterations for stability
        recent_wins = [m.get("win_rate", 0.0) for m in self.metrics_history[-100:]]
        avg_win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
        
        milestone_reached = avg_win_rate >= stage.win_rate_milestone
        max_reached = stage.max_iterations > 0 and iters_in_stage >= stage.max_iterations
        
        if milestone_reached or max_reached:
            return self._advance_stage(avg_win_rate, milestone_reached)
        
        return False
    
    def _advance_stage(self, final_win_rate: float, milestone_reached: bool) -> bool:
        """Advance to the next stage."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Already at final stage
        
        reason = "milestone" if milestone_reached else "max_iterations"
        print(f"[Curriculum] Stage '{self.current_stage.name}' complete ({reason})")
        print(f"[Curriculum] Final win rate: {final_win_rate:.2%}")
        
        self.current_stage_idx += 1
        self.stage_start_iteration = self.current_iteration
        
        print(f"[Curriculum] Advancing to stage: {self.current_stage.name}")
        
        # Save checkpoint
        self.save_checkpoint()
        
        return True
    
    def get_opponent_config(self) -> Dict[str, Any]:
        """
        Get configuration for the current opponent.
        
        Returns:
            Dictionary with opponent type and settings
        """
        stage = self.current_stage
        
        if stage.opponent == "self_play":
            return {
                "type": "self_play",
                "phase": self.current_phase.value,
                "stage": stage.name,
            }
        else:
            return {
                "type": "sample_bot",
                "script": stage.opponent,
                "path": f"/app/src/bots/{stage.opponent}",
                "stage": stage.name,
            }
    
    def get_metrics_for_logging(self) -> Dict[str, Any]:
        """Get curriculum state as metrics for MLflow/logging."""
        return {
            "curriculum/phase": self.current_phase.value,
            "curriculum/stage_idx": self.current_stage_idx,
            "curriculum/stage_name": self.current_stage.name,
            "curriculum/iterations_in_stage": self.iterations_in_stage,
            "curriculum/total_iterations": self.current_iteration,
        }
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save curriculum state to file."""
        path = path or os.path.join(self.checkpoint_dir, "curriculum_state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            "stages": [s.to_dict() for s in self.stages],
            "current_stage_idx": self.current_stage_idx,
            "current_iteration": self.current_iteration,
            "stage_start_iteration": self.stage_start_iteration,
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"[Curriculum] Checkpoint saved to {path}")
        return path
    
    @classmethod
    def load_checkpoint(cls, path: str) -> "TrainingCurriculum":
        """Load curriculum state from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        stages = [CurriculumStage.from_dict(s) for s in state["stages"]]
        
        return cls(
            stages=stages,
            current_stage_idx=state["current_stage_idx"],
            current_iteration=state["current_iteration"],
            stage_start_iteration=state["stage_start_iteration"],
        )
