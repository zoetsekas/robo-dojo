from robocode_tank_royale.bot_api import Bot, BotInfo
import asyncio
import sys

class WallsBot(Bot):
    async def run(self):
        # Move to a corner first or just hug the walls
        move_amount = max(self.arena_width, self.arena_height)
        
        while self.is_running():
            # Hug the walls
            await self.forward(move_amount)
            await self.turn_left(90)
            # Scan while moving
            self.turn_gun_left(360)
            await self.go()

if __name__ == "__main__":
    bot_info = BotInfo(
        name="WallsBot",
        version="1.0",
        authors=["RoboDojo"],
        game_types=["melee", "1v1"],
        description="A bot that hugs the walls"
    )
    server_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:7654"
    bot = WallsBot(bot_info, server_url)
    asyncio.run(bot.start())
