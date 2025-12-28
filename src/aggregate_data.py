"""
Aggregate expert data collected from multiple parallel collectors.
Combines all JSON game state files into a single training-ready dataset.
"""
import os
import json
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Aggregate] %(message)s')
logger = logging.getLogger(__name__)

def aggregate_expert_data(data_dir="artifacts/expert_data", output_file="artifacts/aggregated_expert_data.npz"):
    """
    Aggregate all expert game state JSON files into a single .npz file.
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist!")
        return
    
    # Find all JSON files
    json_files = list(data_dir.glob("expert_game_states_*.json"))
    
    if not json_files:
        logger.warning(f"No expert data files found in {data_dir}")
        return
    
    logger.info(f"Found {len(json_files)} data files to aggregate:")
    for f in json_files:
        logger.info(f"  - {f.name}")
    
    # Load and combine all game states
    all_states = []
    total_steps = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                states = json.load(f)
                all_states.extend(states)
                total_steps += len(states)
                logger.info(f"  Loaded {len(states)} steps from {json_file.name}")
        except Exception as e:
            logger.error(f"  Error loading {json_file.name}: {e}")
    
    if not all_states:
        logger.error("No valid game states found!")
        return
    
    logger.info(f"\nTotal steps collected: {total_steps}")
    
    # Convert to training format
    # For each step, extract features from bot_states
    observations = []
    actions = []
    
    for state in all_states:
        bot_states = state.get("bot_states", [])
        if not bot_states:
            continue
        
        # Use first bot's perspective (SimpleTarget)
        bot = bot_states[0]
        
        # Extract observation features
        obs = {
            "x": bot.get("x", 0),
            "y": bot.get("y", 0),
            "direction": bot.get("direction", 0),
            "gun_direction": bot.get("gunDirection", 0),
            "radar_direction": bot.get("radarDirection", 0),
            "speed": bot.get("speed", 0),
            "gun_heat": bot.get("gunHeat", 0),
            "energy": bot.get("energy", 100),
        }
        observations.append(obs)
        
        # Extract actions (what the bot did this turn)
        # Note: This is reconstructed from state changes - not perfect but good enough
        action = {
            "target_speed": bot.get("targetSpeed", 0),
            "turn_rate": bot.get("turnRate", 0),
            "gun_turn_rate": bot.get("gunTurnRate", 0),
            "radar_turn_rate": bot.get("radarTurnRate", 0),
            "fire_power": 0  # We'd need event data for this
        }
        actions.append(action)
    
    # Convert to numpy arrays
    obs_array = np.array([[o["x"], o["y"], o["direction"], o["gun_direction"], 
                          o["radar_direction"], o["speed"], o["gun_heat"], o["energy"]] 
                         for o in observations], dtype=np.float32)
    
    action_array = np.array([[a["target_speed"], a["turn_rate"], a["gun_turn_rate"],
                             a["radar_turn_rate"], a["fire_power"]]
                            for a in actions], dtype=np.float32)
    
    # Save aggregated data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(
        output_file,
        observations=obs_array,
        actions=action_array,
        metadata={
            "total_steps": total_steps,
            "num_collectors": len(json_files),
            "description": "Aggregated expert demonstrations from sample bot battles"
        }
    )
    
    logger.info(f"\n✅ Aggregated data saved to {output_file}")
    logger.info(f"   - Observations shape: {obs_array.shape}")
    logger.info(f"   - Actions shape: {action_array.shape}")

if __name__ == "__main__":
    aggregate_expert_data()
