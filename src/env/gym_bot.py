"""
GymBot - Async bridge between Gymnasium environment and Robocode Tank Royale.

This bot uses the Tank Royale async Python API correctly:
- Event handlers are async functions
- Actions like fire() are awaited
- Communication with Gym env uses thread-safe queues

Tank Royale Event Reference (v0.34.2):
======================================
Base Events:
- BotEvent: turn_number (int)
- EventABC: base class with no attributes

Bot Events (inherit from BotEvent, have turn_number):
- TickEvent: round_number, bot_state, bullet_states, events
- DeathEvent: is_critical() -> True
- WonRoundEvent: is_critical() -> True
- ScannedBotEvent: scanned_by_bot_id, scanned_bot_id, energy, x, y, direction, speed
- HitBotEvent: victim_id, energy, x, y, is_rammed
- HitWallEvent: (no additional attrs)
- HitByBulletEvent: bullet, damage, energy
- BulletFiredEvent: bullet (BulletState)
- BulletHitBotEvent: victim_id, bullet, damage, energy
- BulletHitWallEvent: bullet
- BulletHitBulletEvent: bullet, hit_bullet
- BotDeathEvent: victim_id (another bot died)
- SkippedTurnEvent: is_critical() -> True

Game Events (inherit from EventABC):
- RoundStartedEvent: round_number
- RoundEndedEvent: round_number, turn_number, results
- GameStartedEvent: my_id, initial_position, game_setup
- GameEndedEvent: number_of_rounds, results
"""
import asyncio
import queue
import logging
from robocode_tank_royale.bot_api import Bot, BotInfo
from robocode_tank_royale.bot_api.events import (
    TickEvent, ScannedBotEvent, HitBotEvent, HitWallEvent, 
    DeathEvent, WonRoundEvent, RoundStartedEvent, RoundEndedEvent,
    HitByBulletEvent, BulletFiredEvent, BulletHitBotEvent, BulletHitWallEvent,
    BotDeathEvent, SkippedTurnEvent, GameStartedEvent, GameEndedEvent
)
from robocode_tank_royale.bot_api.internal.thread_interrupted_exception import ThreadInterruptedException

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [GymBot] %(message)s')
logger = logging.getLogger(__name__)


class GymBot(Bot):
    """Bot that bridges between async Tank Royale API and sync Gym environment."""
    
    def __init__(self, bot_info: BotInfo, server_url: str):
        super().__init__(bot_info, server_url)
        self.event_queue = queue.Queue()  # Thread-safe queue for events
        self.action_queue = queue.Queue()  # Thread-safe queue for actions
        self.current_tick = 0
        self.last_obs = {}
        self._pending_action = None
        
        # Statistics
        self.rounds_played = 0
        self.wins = 0
        self.deaths = 0
        self.bullets_fired = 0
        self.hits_dealt = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        
        logger.info(f"GymBot initialized, connecting to {server_url}")
    
    async def run(self):
        """Main bot loop - called by Tank Royale when game starts."""
        logger.info("GymBot run() started - bot is active")
        try:
            while self.is_running():
                # Check if there's an action from the environment
                try:
                    action = self.action_queue.get_nowait()
                    await self._apply_action(action)
                except queue.Empty:
                    pass
                
                # Must call go() to advance to next turn
                await self.go()
        except asyncio.CancelledError:
            logger.info("GymBot run() cancelled")
            raise
        except ThreadInterruptedException:
            logger.info("GymBot lifecycle interrupted (normal during shutdown)")
        except Exception as e:
            logger.error(f"Error in GymBot run loop: {e}")
        finally:
            logger.info("GymBot run() ended")

    async def stop_bot(self):
        """Explicitly stop the bot and its connection."""
        logger.info("GymBot.stop_bot() requested")
        try:
            if self.is_running():
                await self.stop()
        except ThreadInterruptedException:
            logger.debug("Caught expected ThreadInterruptedException during stop_bot")
        except Exception as e:
            logger.warning(f"Error during bot stop: {e}")
    
    # === Game Lifecycle Events ===
    
    async def on_game_started(self, event: GameStartedEvent):
        """Called when game starts. Not a BotEvent (no turn_number)."""
        logger.info(f"Game started! Bot ID: {event.my_id}, Arena: {event.game_setup.arena_width}x{event.game_setup.arena_height}")
        self.event_queue.put({"type": "game_start", "bot_id": event.my_id})
    
    async def on_game_ended(self, event: GameEndedEvent):
        """Called when game ends. Not a BotEvent (no turn_number)."""
        logger.info(f"Game ended after {event.number_of_rounds} rounds")
        self.event_queue.put({"type": "game_end", "rounds": event.number_of_rounds})
    
    async def on_round_started(self, event: RoundStartedEvent):
        """Called when a new round starts. Not a BotEvent (no turn_number)."""
        self.rounds_played += 1
        logger.info(f"Round {event.round_number} started (total: {self.rounds_played})")
        self.event_queue.put({"type": "round_start", "round": event.round_number})
    
    async def on_round_ended(self, event: RoundEndedEvent):
        """Called when a round ends. Not a BotEvent but HAS turn_number."""
        logger.info(f"Round {event.round_number} ended at turn {event.turn_number}")
        self.event_queue.put({"type": "round_end", "round": event.round_number, "turn": event.turn_number})
    
    # === Turn Events ===
    
    async def on_tick(self, event: TickEvent):
        """Called every turn. Send observation to environment."""
        self.current_tick = event.turn_number
        
        # Log every 100 ticks for development
        if event.turn_number % 100 == 0:
            logger.info(f"Tick {event.turn_number}: pos=({self.get_x():.1f}, {self.get_y():.1f}), energy={self.get_energy():.1f}")
        
        # Construct observation from bot state (all fields from bot-state.schema.yaml)
        obs = {
            "x": self.get_x(),
            "y": self.get_y(),
            "speed": self.get_speed(),
            "energy": self.get_energy(),
            "heading": self.get_direction(),
            "gun_heading": self.get_gun_direction(),
            "radar_heading": self.get_radar_direction(),
            "radar_sweep": self.get_radar_sweep(),  # Angle swept by radar this turn
            "turn_rate": self.get_turn_rate(),      # Current body turn rate
            "gun_turn_rate": self.get_gun_turn_rate(),  # Current gun turn rate
            "radar_turn_rate": self.get_radar_turn_rate(),  # Current radar turn rate
            "gun_heat": self.get_gun_heat(),
            "enemy_count": self.get_enemy_count(),
            "scanned": []
        }
        self.event_queue.put({"type": "tick", "obs": obs, "turn": event.turn_number})
    
    async def on_skipped_turn(self, event: SkippedTurnEvent):
        """Called when bot skipped a turn (too slow)."""
        logger.warning(f"SKIPPED TURN at {event.turn_number}!")
        self.event_queue.put({"type": "skipped_turn", "turn": event.turn_number})
    
    # === Scanning Events ===
    
    async def on_scanned_bot(self, event: ScannedBotEvent):
        """Bot detected by radar scan."""
        # Attributes: scanned_by_bot_id, scanned_bot_id, energy, x, y, direction, speed
        logger.debug(f"Scanned bot id={event.scanned_bot_id} at ({event.x:.1f}, {event.y:.1f})")
        self.event_queue.put({
            "type": "scanned", 
            "enemy_id": event.scanned_bot_id,
            "x": event.x, 
            "y": event.y,
            "energy": event.energy,
            "direction": event.direction,
            "speed": event.speed
        })
    
    # === Collision Events ===
    
    async def on_hit_wall(self, event: HitWallEvent):
        """Bot hit a wall."""
        logger.debug(f"Hit wall at turn {event.turn_number}")
        self.event_queue.put({"type": "hit_wall", "turn": event.turn_number})
    
    async def on_hit_bot(self, event: HitBotEvent):
        """Bot collided with another bot."""
        # Attributes: victim_id, energy, x, y, is_rammed
        logger.debug(f"Hit bot id={event.victim_id}, rammed={event.is_rammed}")
        self.event_queue.put({
            "type": "hit_bot", 
            "victim_id": event.victim_id,
            "is_rammed": event.is_rammed,
            "energy": event.energy,
            "x": event.x,  # Position of collision
            "y": event.y
        })
    
    # === Bullet Events ===
    
    async def on_bullet_fired(self, event: BulletFiredEvent):
        """Our bot fired a bullet."""
        self.bullets_fired += 1
        logger.debug(f"Fired bullet at turn {event.turn_number}")
    
    async def on_bullet_hit_bot(self, event: BulletHitBotEvent):
        """Our bullet hit an enemy bot."""
        # Attributes: victim_id, bullet, damage, energy
        self.hits_dealt += 1
        self.damage_dealt += event.damage
        logger.debug(f"Bullet hit enemy id={event.victim_id}, damage={event.damage:.1f}")
        self.event_queue.put({
            "type": "bullet_hit",
            "victim_id": event.victim_id,
            "damage": event.damage,
            "bullet_power": event.bullet.power if event.bullet else None
        })
    
    async def on_bullet_hit_wall(self, event: BulletHitWallEvent):
        """Our bullet hit a wall (missed enemy)."""
        # Attributes: bullet
        logger.debug(f"Bullet hit wall at turn {event.turn_number}")
        self.event_queue.put({
            "type": "bullet_miss",
            "turn": event.turn_number,
            "bullet_id": event.bullet.bullet_id if event.bullet else None
        })
    
    async def on_hit_by_bullet(self, event: HitByBulletEvent):
        """Our bot was hit by enemy bullet."""
        # Attributes: bullet, damage, energy
        self.damage_taken += event.damage
        logger.debug(f"Hit by bullet, damage={event.damage:.1f}, energy left={event.energy:.1f}")
        self.event_queue.put({
            "type": "hit_by_bullet",
            "damage": event.damage,
            "energy": event.energy,
            "bullet_owner": event.bullet.owner_id if event.bullet else None,
            "bullet_power": event.bullet.power if event.bullet else None
        })
    
    # === Death Events ===
    
    async def on_death(self, event: DeathEvent):
        """This bot died."""
        self.deaths += 1
        logger.info(f"Bot DIED at turn {event.turn_number} (total deaths: {self.deaths})")
        self.event_queue.put({"type": "death", "turn": event.turn_number})
    
    async def on_won_round(self, event: WonRoundEvent):
        """This bot won the round."""
        self.wins += 1
        logger.info(f"Bot WON round at turn {event.turn_number} (total wins: {self.wins})")
        self.event_queue.put({"type": "win", "turn": event.turn_number})
    
    async def on_bot_death(self, event: BotDeathEvent):
        """Another bot died."""
        # Attributes: victim_id
        logger.debug(f"Enemy bot id={event.victim_id} died")
        self.event_queue.put({"type": "enemy_death", "victim_id": event.victim_id})
    
    # === Action Handling ===
    
    async def _apply_action(self, action):
        """Apply action from environment to bot.
        
        Action space: [target_speed, turn_rate, gun_turn_rate, radar_turn_rate, fire]
        """
        # Set movement properties (these are properties, not coroutines)
        self.target_speed = float(action[0])
        self.turn_rate = float(action[1])
        self.gun_turn_rate = float(action[2])
        self.radar_turn_rate = float(action[3])
        
        # Fire if power > threshold and gun is cool
        if action[4] > 0.1 and self.get_gun_heat() == 0:
            fire_power = max(0.1, min(3.0, float(action[4])))
            await self.fire(fire_power)
    
    # === Gym Interface (thread-safe) ===
    
    def send_action(self, action):
        """Queue an action to be applied (thread-safe)."""
        self.action_queue.put(action)
    
    def get_events(self):
        """Get pending events (thread-safe)."""
        events = []
        while not self.event_queue.empty():
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events
    
    def get_stats(self):
        """Get bot statistics."""
        return {
            "rounds_played": self.rounds_played,
            "wins": self.wins,
            "deaths": self.deaths,
            "win_rate": self.wins / max(1, self.rounds_played),
            "bullets_fired": self.bullets_fired,
            "hits_dealt": self.hits_dealt,
            "accuracy": self.hits_dealt / max(1, self.bullets_fired),
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken
        }
