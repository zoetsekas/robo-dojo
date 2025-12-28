import asyncio
from robocode_tank_royale.bot_api import Bot, BotInfo

class SimpleTarget(Bot):
    def __init__(self, server_url):
        bot_info = BotInfo(
            name="SimpleTarget",
            version="1.0",
            authors=["RoboDojo"],
            description="Simple stationary target for training",
            game_types=["classic", "melee", "1v1"]
        )
        super().__init__(bot_info, server_url)
    
    async def run(self):
        while self.is_running():
            await self.go()

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:7654"
    bot = SimpleTarget() if len(sys.argv) <= 1 else SimpleTarget(url)
    asyncio.run(bot.start())
