#!/usr/bin/env python3
"""
Minimal bot that does nothing - just keeps the game alive for observer to collect data.
Uses a simple workaround for the BotState issue by never processing ticks.
"""
import asyncio
from robocode_tank_royale.bot_api import Bot, BotInfo

class NoOpBot(Bot):
    def __init__(self, name="NoOpBot1"):
        bot_info = BotInfo(
            name=name,
            version="1.0",
            authors=["RoboDojo"],
            description="Minimal bot for observer-based data collection",
            game_types=["classic", "melee", "1v1"]
        )
        super().__init__(bot_info, "ws://127.0.0.1:7654")
    
    async def run(self):
        # Don't process ticks - just stay connected
        # The observer will collect the game data
        while self.is_running():
            try:
                await asyncio.sleep(0.1)
            except:
                break

if __name__ == "__main__":
    import sys
    bot_name = sys.argv[1] if len(sys.argv) > 1 else "NoOpBot1"
    bot = NoOpBot(bot_name)
    try:
        asyncio.run(bot.start())
    except Exception as e:
        print(f"Bot {bot_name} stopped: {e}")
