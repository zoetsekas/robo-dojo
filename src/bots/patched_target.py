#!/usr/bin/env python3
"""
Patched SimpleTarget bot with BotState fix.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply comprehensive BotState patch before any other imports
import bot_state_patch

# Now safe to import robocode modules
import asyncio
from robocode_tank_royale.bot_api import Bot, BotInfo

class SimpleTarget(Bot):
    def __init__(self):
        bot_info = BotInfo(
            name="SimpleTarget",
            version="1.0",
            authors=["RoboDojo"],
            description="Simple stationary target for training",
            game_types=["classic", "melee", "1v1"]
        )
        super().__init__(bot_info, "ws://127.0.0.1:7654")
    
    async def run(self):
        while self.is_running():
            await self.go()

if __name__ == "__main__":
    bot = SimpleTarget()
    asyncio.run(bot.start())
