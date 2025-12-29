import asyncio
import websockets
import json
import sys
import logging

# Setup logging - output to stderr so it shows up in subprocess capture
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Controller] %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

# Default game setup values
DEFAULT_GAME_SETUP = {
    "game_type": "melee",
    "arena_width": 800,
    "arena_height": 600,
    "min_participants": 2,
    "max_participants": None,
    "num_rounds": 1000,
    "gun_cooling_rate": 0.1,
    "max_inactivity_turns": 450,
    "turn_timeout": 100000,
    "ready_timeout": 1000000,
    "default_turns_per_second": 30
}

async def trigger_start(server_url, expected_bots=2, game_setup_config=None):
    # Merge provided config with defaults
    cfg = {**DEFAULT_GAME_SETUP, **(game_setup_config or {})}
    
    try:
        async with websockets.connect(server_url) as ws:
            # 1. Wait for Server Handshake
            msg = await ws.recv()
            handshake = json.loads(msg)
            session_id = handshake["sessionId"]
            
            # 2. Send Controller Handshake
            await ws.send(json.dumps({
                "type": "ControllerHandshake",
                "sessionId": session_id,
                "name": "RoboDojoController",
                "version": "1.0",
                "author": "Antigravity"
            }))
            
            # 3. Wait for bots to join
            logger.info(f"Controller waiting for {expected_bots} bots...")
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data["type"] == "BotListUpdate":
                    bots = data["bots"]
                    bot_names = [b.get("name", "Unknown") for b in bots]
                    logger.info(f"Bots joined: {len(bots)} ({', '.join(bot_names)})")
                    if len(bots) >= expected_bots:
                        break
                await asyncio.sleep(0.5)
            
            # 4. Send StartGame with config-driven gameSetup
            bot_addresses = [{"host": b["host"], "port": b["port"]} for b in data["bots"]]
            
            game_setup = {
                "gameType": cfg["game_type"],
                "arenaWidth": cfg["arena_width"],
                "arenaHeight": cfg["arena_height"],
                "minNumberOfParticipants": cfg["min_participants"],
                "maxNumberOfParticipants": cfg["max_participants"],
                "numberOfRounds": cfg["num_rounds"],
                "gunCoolingRate": cfg["gun_cooling_rate"],
                "maxInactivityTurns": cfg["max_inactivity_turns"],
                "turnTimeout": cfg["turn_timeout"],
                "readyTimeout": cfg["ready_timeout"],
                "isArenaWidthLocked": False,
                "isArenaHeightLocked": False,
                "isMinNumberOfParticipantsLocked": False,
                "isMaxNumberOfParticipantsLocked": False,
                "isNumberOfRoundsLocked": False,
                "isGunCoolingRateLocked": False,
                "isMaxInactivityTurnsLocked": False,
                "isTurnTimeoutLocked": False,
                "isReadyTimeoutLocked": False,
                "defaultTurnsPerSecond": cfg["default_turns_per_second"]
            }
            
            await ws.send(json.dumps({
                "type": "StartGame",
                "gameSetup": game_setup,
                "botAddresses": bot_addresses
            }))
            logger.info(f"Match triggered with config: {cfg['game_type']}, arena {cfg['arena_width']}x{cfg['arena_height']}, min_participants={cfg['min_participants']}")
            
            # 5. Wait for GameStarted
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data["type"] in ["GameStartedEventForBot", "RoundStartedEvent"]:
                    logger.info(f"Server confirms game/round start ({data['type']}).")
                    break
                await asyncio.sleep(0.1)
            
            # Keep alive for a moment
            await asyncio.sleep(1)

            
    except Exception as e:
        logger.error(f"Controller error: {e}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:7654"
    num_bots = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    game_setup_json = sys.argv[3] if len(sys.argv) > 3 else "{}"
    
    try:
        game_setup_config = json.loads(game_setup_json)
    except json.JSONDecodeError:
        logger.warning(f"Invalid game_setup JSON, using defaults: {game_setup_json}")
        game_setup_config = {}
    
    asyncio.run(trigger_start(url, num_bots, game_setup_config))

