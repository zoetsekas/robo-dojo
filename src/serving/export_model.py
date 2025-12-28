import torch
import os
import argparse
import logging
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from src.models.multimodal_net import MultimodalRoboModel, VectorOnlyRoboModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Export] %(message)s')
logger = logging.getLogger(__name__)

def export_checkpoint(checkpoint_path, output_path):
    """Load an RLlib checkpoint and save the model weights for standalone use."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Register models so Ray can deserialize the policy
    ModelCatalog.register_custom_model("multimodal_robo_model", MultimodalRoboModel)
    ModelCatalog.register_custom_model("vector_only_robo_model", VectorOnlyRoboModel)
    
    # Load just the policy to avoid starting the whole algorithm/env
    policy_path = os.path.abspath(os.path.join(checkpoint_path, "policies", "default_policy"))
    logger.info(f"Loading policy from: {policy_path}")
    policy = Policy.from_checkpoint(policy_path)
    
    # Extract the torch model
    model = policy.model
    
    # Save the state dict
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Successfully exported model weights to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RLlib checkpoint")
    parser.add_argument("--output", type=str, default="artifacts/serving/bot_weights.pt", help="Output path for .pt file")
    parser.add_argument("--vector-only", action="store_true", help="Hint that this is a vector-only model (not strictly needed but good for docs)")
    
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)

