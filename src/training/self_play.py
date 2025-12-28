"""
Self-Play Training Module for RoboDojo.

Implements league-based self-play where the main policy plays against
a pool of its frozen past versions. This creates a continuously
evolving curriculum that drives the policy to become more robust.

Key components:
- PolicyLeague: Manages a pool of frozen policy snapshots
- OpponentSampler: Selects opponents with configurable strategies
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os
import json
import pickle
from datetime import datetime


@dataclass
class PolicySnapshot:
    """A frozen snapshot of a policy's weights."""
    iteration: int
    weights: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    elo_rating: float = 1000.0  # Initial ELO rating
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize snapshot metadata (not weights)."""
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "elo_rating": self.elo_rating,
        }


class PolicyLeague:
    """
    Manages a pool of frozen policy snapshots for self-play.
    
    The league maintains a fixed-size pool of past policy versions.
    When the pool is full, the oldest policies are removed (FIFO),
    but exceptionally strong policies may be preserved.
    
    Example usage:
        league = PolicyLeague(max_size=20)
        
        # After each training checkpoint
        league.add_snapshot(
            iteration=1000,
            weights=policy.get_weights(),
            metrics={"win_rate": 0.65, "reward_mean": 150.0}
        )
        
        # Get opponent for next episode
        opponent_weights = league.sample_opponent()
    """
    
    def __init__(
        self,
        max_size: int = 20,
        preserve_top_k: int = 3,
        checkpoint_dir: str = "artifacts/policy_league"
    ):
        """
        Args:
            max_size: Maximum number of policies in the league
            preserve_top_k: Number of top-performing policies to always keep
            checkpoint_dir: Directory for saving/loading league state
        """
        self.max_size = max_size
        self.preserve_top_k = preserve_top_k
        self.checkpoint_dir = checkpoint_dir
        self.snapshots: List[PolicySnapshot] = []
        
    def __len__(self) -> int:
        return len(self.snapshots)
    
    def add_snapshot(
        self,
        iteration: int,
        weights: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """
        Add a new policy snapshot to the league.
        
        Args:
            iteration: Training iteration when snapshot was taken
            weights: Policy weights from policy.get_weights()
            metrics: Performance metrics (win_rate, reward_mean, etc.)
        """
        snapshot = PolicySnapshot(
            iteration=iteration,
            weights=weights,
            metrics=metrics,
        )
        
        self.snapshots.append(snapshot)
        
        # Prune if over capacity
        if len(self.snapshots) > self.max_size:
            self._prune_league()
        
        print(f"[League] Added snapshot at iteration {iteration} "
              f"(league size: {len(self.snapshots)})")
    
    def _prune_league(self) -> None:
        """Remove policies to stay within max_size, preserving top performers."""
        if len(self.snapshots) <= self.max_size:
            return
        
        # Sort by ELO rating (or win_rate if no ELO games played)
        sorted_by_strength = sorted(
            enumerate(self.snapshots),
            key=lambda x: x[1].elo_rating,
            reverse=True
        )
        
        # Indices to preserve
        preserve_indices = set()
        
        # Always keep the most recent policy
        preserve_indices.add(len(self.snapshots) - 1)
        
        # Keep top K by strength
        for idx, _ in sorted_by_strength[:self.preserve_top_k]:
            preserve_indices.add(idx)
        
        # Fill remaining slots with most recent policies
        remaining = self.max_size - len(preserve_indices)
        for i in range(len(self.snapshots) - 1, -1, -1):
            if len(preserve_indices) >= self.max_size:
                break
            preserve_indices.add(i)
        
        # Filter to only preserved policies
        self.snapshots = [s for i, s in enumerate(self.snapshots) if i in preserve_indices]
    
    def sample_opponent(
        self,
        strategy: str = "prioritized",
        current_elo: Optional[float] = None
    ) -> Tuple[Dict[str, Any], int]:
        """
        Sample an opponent from the league.
        
        Args:
            strategy: Sampling strategy
                - "prioritized": 70% recent, 20% historical, 10% random
                - "uniform": Equal probability for all
                - "elo_matched": Prefer similar ELO ratings
            current_elo: Current policy's ELO (for elo_matched strategy)
            
        Returns:
            Tuple of (opponent_weights, snapshot_iteration)
        """
        if not self.snapshots:
            raise ValueError("League is empty, cannot sample opponent")
        
        if len(self.snapshots) == 1:
            return self.snapshots[0].weights, self.snapshots[0].iteration
        
        if strategy == "uniform":
            idx = np.random.randint(len(self.snapshots))
            
        elif strategy == "prioritized":
            idx = self._prioritized_sample()
            
        elif strategy == "elo_matched" and current_elo is not None:
            idx = self._elo_matched_sample(current_elo)
            
        else:
            idx = self._prioritized_sample()
        
        snapshot = self.snapshots[idx]
        return snapshot.weights, snapshot.iteration
    
    def _prioritized_sample(self) -> int:
        """Sample with priority on recent policies."""
        n = len(self.snapshots)
        
        # Define probability weights
        weights = np.zeros(n)
        
        # Last 3 policies get 70% total
        recent_start = max(0, n - 3)
        weights[recent_start:] = 0.7 / min(3, n)
        
        # Historical policies get 20%
        if recent_start > 0:
            weights[:recent_start] = 0.2 / recent_start
        
        # Add 10% uniform noise
        weights += 0.1 / n
        
        # Normalize
        weights /= weights.sum()
        
        return np.random.choice(n, p=weights)
    
    def _elo_matched_sample(self, current_elo: float, temperature: float = 200.0) -> int:
        """Sample opponents with similar ELO ratings."""
        elos = np.array([s.elo_rating for s in self.snapshots])
        
        # Probability inversely proportional to ELO difference
        diff = np.abs(elos - current_elo)
        weights = np.exp(-diff / temperature)
        weights /= weights.sum()
        
        return np.random.choice(len(self.snapshots), p=weights)
    
    def update_elo(
        self,
        winner_iteration: int,
        loser_iteration: int,
        k_factor: float = 32.0
    ) -> None:
        """
        Update ELO ratings after a match.
        
        Args:
            winner_iteration: Iteration of the winning policy
            loser_iteration: Iteration of the losing policy
            k_factor: ELO update magnitude
        """
        winner = next((s for s in self.snapshots if s.iteration == winner_iteration), None)
        loser = next((s for s in self.snapshots if s.iteration == loser_iteration), None)
        
        if not winner or not loser:
            return
        
        # Expected scores
        exp_winner = 1 / (1 + 10 ** ((loser.elo_rating - winner.elo_rating) / 400))
        exp_loser = 1 - exp_winner
        
        # Update ratings
        winner.elo_rating += k_factor * (1 - exp_winner)
        loser.elo_rating += k_factor * (0 - exp_loser)
    
    def get_metrics_for_logging(self) -> Dict[str, Any]:
        """Get league state as metrics for MLflow/logging."""
        if not self.snapshots:
            return {"league/size": 0}
        
        elos = [s.elo_rating for s in self.snapshots]
        return {
            "league/size": len(self.snapshots),
            "league/elo_mean": np.mean(elos),
            "league/elo_max": np.max(elos),
            "league/elo_min": np.min(elos),
            "league/latest_iteration": self.snapshots[-1].iteration,
        }
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save league state to disk."""
        path = path or os.path.join(self.checkpoint_dir, "league_state.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                "snapshots": self.snapshots,
                "max_size": self.max_size,
                "preserve_top_k": self.preserve_top_k,
            }, f)
        
        # Also save metadata as JSON for inspection
        meta_path = path.replace(".pkl", "_meta.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "num_snapshots": len(self.snapshots),
                "snapshots": [s.to_dict() for s in self.snapshots],
            }, f, indent=2)
        
        print(f"[League] Saved checkpoint to {path}")
        return path
    
    @classmethod
    def load_checkpoint(cls, path: str) -> "PolicyLeague":
        """Load league state from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        league = cls(
            max_size=data["max_size"],
            preserve_top_k=data["preserve_top_k"],
        )
        league.snapshots = data["snapshots"]
        
        print(f"[League] Loaded {len(league.snapshots)} snapshots from {path}")
        return league


class OpponentSampler:
    """
    High-level opponent sampler that works with both sample bots and self-play.
    
    This class abstracts the opponent selection logic so the training loop
    doesn't need to know about the underlying curriculum or league mechanics.
    """
    
    def __init__(
        self,
        league: Optional[PolicyLeague] = None,
        default_strategy: str = "prioritized"
    ):
        self.league = league
        self.default_strategy = default_strategy
        self.current_opponent_iteration: Optional[int] = None
    
    def get_opponent(
        self,
        opponent_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get opponent weights based on config.
        
        Args:
            opponent_config: From TrainingCurriculum.get_opponent_config()
            
        Returns:
            Opponent weights if self-play, None if sample bot (bot is separate process)
        """
        if opponent_config["type"] == "sample_bot":
            # Sample bots run as separate processes, no weights needed
            self.current_opponent_iteration = None
            return None
        
        if opponent_config["type"] == "self_play" and self.league:
            weights, iteration = self.league.sample_opponent(self.default_strategy)
            self.current_opponent_iteration = iteration
            return weights
        
        return None
    
    def report_match_result(self, main_won: bool, main_iteration: int) -> None:
        """Report match result for ELO updates."""
        if self.league and self.current_opponent_iteration is not None:
            if main_won:
                self.league.update_elo(main_iteration, self.current_opponent_iteration)
            else:
                self.league.update_elo(self.current_opponent_iteration, main_iteration)
