import hydra
from omegaconf import DictConfig, OmegaConf
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import cv2
import os
import numpy as np

from src.env.robocode_env import RobocodeGymEnv
from src.models.multimodal_net import MultimodalRoboModel
from src.training.callbacks import CombinedTrainingCallback, CurriculumCallback, SelfPlayCallback
from ray.tune.logger import Logger
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Train] %(message)s')
logger = logging.getLogger(__name__)


class MinimalLogger(Logger):
    """A logger that does nothing to avoid deprecation warnings from UnifiedLogger."""
    def _init(self): pass
    def on_result(self, result): pass
    def close(self): pass
    def flush(self): pass


def custom_logger_creator(config):
    return MinimalLogger(config, tempfile.mkdtemp(prefix="robodojo_log_"))


def create_callbacks(cfg: DictConfig):
    """Create the appropriate callback configuration based on config."""
    if cfg.curriculum_enabled or cfg.self_play_only:
        # Use CombinedTrainingCallback for curriculum/self-play
        return CombinedTrainingCallback(
            snapshot_interval=100 if cfg.smoke_test else 500,
            league_max_size=5 if cfg.smoke_test else 20,
        )
    else:
        # For smoke test without curriculum, use default callbacks
        return DefaultCallbacks()


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    # Register environment and model
    register_env("robocode_multimodal", lambda env_cfg: RobocodeGymEnv(env_cfg))
    ModelCatalog.register_custom_model("multimodal_robo_model", MultimodalRoboModel)

    # Determine batch sizes based on mode
    if cfg.smoke_test:
        train_batch_size = 200
        minibatch_size = 64
        num_epochs = 1
        max_iterations = 5
    else:
        train_batch_size = cfg.training.train_batch_size
        minibatch_size = cfg.training.minibatch_size
        num_epochs = cfg.training.num_epochs
        max_iterations = cfg.max_iterations

    # Prepare env_config
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_config.update({
        "num_workers": cfg.hardware.num_workers,
        "smoke_test": cfg.smoke_test,
    })

    # Build configuration
    config = (
        PPOConfig()
        .environment("robocode_multimodal", env_config=env_config)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=cfg.hardware.num_workers, 
            num_envs_per_env_runner=cfg.env.num_envs_per_env_runner,
            create_env_on_local_worker=False,
            sample_timeout_s=cfg.env.sample_timeout_s
        )
        .training(
            model={"custom_model": "multimodal_robo_model"},
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            # PPO hyperparameters from Hydra
            lr=cfg.training.lr,
            gamma=cfg.training.gamma,
            lambda_=cfg.training.lambda_,
            clip_param=cfg.training.clip_param,
            entropy_coeff=cfg.training.entropy_coeff,
            vf_loss_coeff=cfg.training.vf_loss_coeff,
        )
        .resources(num_gpus=cfg.hardware.num_gpus)
    )
    
    # Disable config validation for experimental features
    config.experimental(_validate_config=False)

    # Set up callbacks
    callbacks = create_callbacks(cfg)
    config.callbacks(type(callbacks))

    logger.info("=" * 60)
    logger.info("RoboDojo Training Configuration (Hydra)")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 60)

    # Initialize Ray
    logger.info("Connecting to Ray cluster...")
    ray.init(address="auto")

    # Build algorithm
    logger.info("Building PPO algorithm...")
    algo = config.build_algo(logger_creator=custom_logger_creator)

    # Resume from checkpoint if specified
    if cfg.resume:
        logger.info(f"Resuming from checkpoint: {cfg.resume}")
        algo.restore(cfg.resume)

    # Training loop
    logger.info("\nStarting training loop...")
    logger.info("-" * 60)
    
    best_reward = float('-inf')
    
    for i in range(max_iterations):
        results = algo.train()
        
        # Extract metrics (handle both old and new API)
        reward = (
            results.get("episode_reward_mean") or 
            results.get("env_runners", {}).get("episode_reward_mean") or
            0.0
        )
        
        # Get curriculum metrics if available
        custom_metrics = results.get("custom_metrics", {})
        stage_name = custom_metrics.get("curriculum/stage_name", "N/A")
        win_rate = custom_metrics.get("curriculum/win_rate", 0.0)
        league_size = custom_metrics.get("league/size", 0)
        
        # Print progress
        logger.info(f"Iter {i:4d} | Reward: {reward:7.2f} | "
                    f"Win Rate: {win_rate:5.1%} | "
                    f"Stage: {stage_name} | "
                    f"League: {league_size}")
        
        # Save best checkpoint
        if reward > best_reward:
            best_reward = reward
            checkpoint_dir = algo.save("artifacts/checkpoints/best")
            logger.info(f"  → New best! Saved to {checkpoint_dir}")
        
        # Periodic checkpoints
        if i > 0 and i % 100 == 0:
            checkpoint_dir = algo.save(f"artifacts/checkpoints/iter_{i}")
            logger.info(f"  → Checkpoint saved to {checkpoint_dir}")
    
    # Final save
    logger.info("-" * 60)
    final_checkpoint = algo.save("artifacts/checkpoints/final")
    logger.info(f"Training complete! Final checkpoint: {final_checkpoint}")
    logger.info(f"Best reward achieved: {best_reward:.2f}")
    
    ray.shutdown()


if __name__ == "__main__":
    main()

