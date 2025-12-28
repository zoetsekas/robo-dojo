from robocode_tank_royale.bot_api import Bot, BotInfo
import asyncio
import sys

class SpinBot(Bot):
    async def run(self):
        while self.is_running():
            # Spin and fire continuously
            self.turn_gun_left(10)
            self.set_fire(1.0)
            self.set_max_speed(8)
            self.set_turn_left(10)
            await self.go()

if __name__ == "__main__":
    bot_info = BotInfo(
        name="SpinBot",
        version="1.0",
        authors=["RoboDojo"],
        game_types=["melee", "1v1"],
        description="A bot that spins and fires"
    )
    server_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:7654"
    bot = SpinBot(bot_info, server_url)
    asyncio.run(bot.start())
