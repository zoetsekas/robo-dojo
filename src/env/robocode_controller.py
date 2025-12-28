import asyncio
import websockets
import json
import sys
import logging

# Setup logging - output to stderr so it shows up in subprocess capture
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Controller] %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

async def trigger_start(server_url, expected_bots=2):
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
            
            # 4. Send StartGame with proper gameSetup
            bot_addresses = [{"host": b["host"], "port": b["port"]} for b in data["bots"]]
            
            # Basic game setup for classic melee
            # Use many rounds to keep game running during training
            game_setup = {
                "gameType": "melee",
                "arenaWidth": 800,
                "arenaHeight": 600,
                "minNumberOfParticipants": 2,
                "maxNumberOfParticipants": None,
                "numberOfRounds": 1000,  # Many rounds for long training sessions
                "gunCoolingRate": 0.1,
                "maxInactivityTurns": 450,
                "turnTimeout": 100000,
                "readyTimeout": 1000000,
                "isArenaWidthLocked": False,
                "isArenaHeightLocked": False,
                "isMinNumberOfParticipantsLocked": False,
                "isMaxNumberOfParticipantsLocked": False,
                "isNumberOfRoundsLocked": False,
                "isGunCoolingRateLocked": False,
                "isMaxInactivityTurnsLocked": False,
                "isTurnTimeoutLocked": False,
                "isReadyTimeoutLocked": False,
                "defaultTurnsPerSecond": 30
            }
            
            await ws.send(json.dumps({
                "type": "StartGame",
                "gameSetup": game_setup,
                "botAddresses": bot_addresses
            }))
            logger.info(f"Match triggered successfully with {len(bots)} bots.")
            
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
    asyncio.run(trigger_start(url, num_bots))

