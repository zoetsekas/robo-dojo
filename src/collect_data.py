"""
Expert data collection by observing battles between sample bots.
We act as a passive observer, recording frame data from their battles.
"""
import asyncio
import websockets
import json
import numpy as np
import time
import os
import subprocess
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Collect] %(message)s')
logger = logging.getLogger(__name__)

async def collect_expert_data(num_rounds=10, output_dir="artifacts/expert_data"):
    """
    Collect expert demonstrations by observing sample bot battles.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for collected data
    visual_frames = []
    game_states = []
    
    server_url = "ws://127.0.0.1:7654"
    
    logger.info("Connecting as observer to collect data...")
    
    try:
        async with websockets.connect(server_url) as ws:
            # 1. Perform observer handshake
            msg = await ws.recv()
            handshake = json.loads(msg)
            session_id = handshake["sessionId"]
            
            await ws.send(json.dumps({
                "type": "ObserverHandshake",
                "sessionId": session_id,
                "name": "DataCollector",
                "version": "1.0",
                "author": "RoboDojo"
            }))
            
            logger.info("Observer connected. Waiting for game to start...")
            
            # 2. Listen for game events and collect data
            round_count = 0
            step_count = 0
            
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "GameStartedEventForObserver":
                    logger.info(f"Game started! Collecting data for round {round_count + 1}...")
                    
                elif msg_type == "TickEventForObserver":
                    # Extract relevant game state
                    turn_number = data.get("turnNumber", 0)
                    round_number = data.get("roundNumber", 0)
                    bot_states = data.get("botStates", [])
                    
                    # Store game state from first bot's perspective
                    if bot_states:
                        state = {
                            "turn": turn_number,
                            "round": round_number,
                            "bot_states": bot_states,
                            "bullet_states": data.get("bulletStates", [])
                        }
                        game_states.append(state)
                        step_count += 1
                        
                        if step_count % 100 == 0:
                            logger.info(f"  Collected {step_count} steps...")
                
                elif msg_type == "RoundEndedEventForObserver":
                    round_count += 1
                    logger.info(f"Round {round_count} ended. Total steps: {step_count}")
                    
                elif msg_type == "GameEndedEventForObserver":
                    logger.info(f"Game ended. Total rounds: {round_count}, Total steps: {step_count}")
                    if round_count >= num_rounds:
                        break
                        
    except Exception as e:
        logger.error(f"Observer error: {e}")
    
    # 3. Save collected data
    if game_states:
        timestamp = int(time.time())
        output_file = f"{output_dir}/expert_game_states_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(game_states, f)
        
        logger.info(f"Saved {len(game_states)} game states to {output_file}")
    else:
        logger.warning("No data collected!")

def main():
    logger.info("Starting Expert Data Collection...")
    logger.info("This will observe battles between SimpleTarget and SimpleSpin.")
    
    # The infrastructure (server, GUI, bots, controller) is already started by robocode_env.py
    # We just need to connect as an observer
    
    # Wait a bit for the infrastructure to be ready
    logger.info("Waiting for infrastructure to initialize...")
    time.sleep(30)
    
    # Run the data collection
    asyncio.run(collect_expert_data(num_rounds=3))
    
    logger.info("Data collection complete!")

if __name__ == "__main__":
    main()
