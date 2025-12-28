"""
OpponentManager - Robust lifecycle management for opponent bots.

Handles:
- Bot registry with both internal and sample bots
- Process lifecycle (start, stop, restart)
- Per-episode random opponent selection
- Cleanup of temporary launcher scripts
"""
import os
import sys
import signal
import subprocess
import tempfile
import threading
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class BotSpec:
    """Specification for a bot that can be launched as an opponent."""
    name: str
    bot_type: str  # "internal" or "sample"
    path: str      # Module path for internal, directory path for sample
    
    def __repr__(self):
        return f"BotSpec({self.name}, {self.bot_type})"


def get_bot_registry() -> Dict[str, BotSpec]:
    """
    Returns the complete registry of available opponent bots.
    
    Internal bots: Python modules in src/bots/
    Sample bots: Robocode sample bots in sample_bots/
    """
    registry = {}
    
    # Internal bots (modules)
    internal_bots = {
        "simple_target": "src.bots.simple_target",
        "spin_bot": "src.bots.spin_bot",
        "walls_bot": "src.bots.walls_bot",
        "noop_bot": "src.bots.noop_bot",
    }
    for name, module in internal_bots.items():
        registry[name] = BotSpec(name=name, bot_type="internal", path=module)
    
    # Sample bots (directories)
    sample_bots = [
        "Crazy", "Fire", "Corners", "RamFire", "SpinBot", 
        "Target", "TrackFire", "VelocityBot", "Walls"
    ]
    for name in sample_bots:
        registry[name] = BotSpec(
            name=name, 
            bot_type="sample", 
            path=f"sample_bots/{name}"
        )
    
    return registry


class OpponentManager:
    """
    Manages opponent bot processes with robust lifecycle handling.
    
    Features:
    - Centralized bot registry
    - Per-episode opponent selection (random or fixed)
    - Process health monitoring and restart
    - Clean shutdown and temp file cleanup
    
    Usage:
        manager = OpponentManager(server_url="ws://127.0.0.1:8000")
        
        # Start opponents for an episode
        manager.start_opponents(["Crazy"], count=1)
        
        # Check if opponents are healthy
        if manager.get_crashed_count() > 0:
            manager.restart_crashed()
        
        # Clean up at episode end
        manager.stop_all()
    """
    
    def __init__(
        self, 
        server_url: str,
        opponent_pool: Optional[List[str]] = None,
        registry: Optional[Dict[str, BotSpec]] = None
    ):
        """
        Initialize the opponent manager.
        
        Args:
            server_url: WebSocket URL of the Robocode server
            opponent_pool: List of bot names to select from (None = all bots)
            registry: Bot registry (None = use default)
        """
        self.server_url = server_url
        self.registry = registry or get_bot_registry()
        self.opponent_pool = opponent_pool or list(self.registry.keys())
        
        self.processes: List[subprocess.Popen] = []
        self.launcher_scripts: List[str] = []  # Track temp files for cleanup
        self.current_opponents: List[str] = []  # Names of active opponents
        self._log_threads: List[threading.Thread] = []
        
        logger.info(f"OpponentManager initialized with {len(self.opponent_pool)} bots in pool")
    
    @property
    def available_bots(self) -> List[str]:
        """List of bot names available for selection."""
        return self.opponent_pool
    
    def select_random_opponent(self) -> str:
        """Select a random opponent from the pool."""
        return random.choice(self.opponent_pool)
    
    def start_opponents(
        self, 
        opponent_types: Optional[List[str]] = None,
        count: int = 1,
        randomize: bool = False
    ) -> List[subprocess.Popen]:
        """
        Start opponent bot processes.
        
        Args:
            opponent_types: Specific bot types to start (None for random)
            count: Number of opponents to start
            randomize: If True and opponent_types is None, select randomly
            
        Returns:
            List of started processes
        """
        # Determine which opponents to start
        if opponent_types is None or randomize:
            selected = [self.select_random_opponent() for _ in range(count)]
        else:
            # Repeat the provided types to match count
            selected = (opponent_types * count)[:count]
        
        logger.info(f"Starting {count} opponent(s): {selected}")
        
        started_procs = []
        for i, bot_name in enumerate(selected):
            spec = self.registry.get(bot_name)
            if spec is None:
                logger.warning(f"Unknown bot '{bot_name}', skipping")
                continue
            
            proc = self._start_single_opponent(spec, index=i)
            if proc:
                started_procs.append(proc)
                self.processes.append(proc)
                self.current_opponents.append(bot_name)
        
        logger.info(f"{len(started_procs)} opponent(s) started successfully")
        return started_procs
    
    def _start_single_opponent(self, spec: BotSpec, index: int) -> Optional[subprocess.Popen]:
        """Start a single opponent bot process."""
        try:
            if spec.bot_type == "internal":
                return self._start_internal_bot(spec, index)
            else:
                return self._start_sample_bot(spec, index)
        except Exception as e:
            logger.error(f"Failed to start opponent '{spec.name}': {e}")
            return None
    
    def _start_internal_bot(self, spec: BotSpec, index: int) -> subprocess.Popen:
        """Start an internal (module-based) bot."""
        cmd = [sys.executable, "-m", spec.path, self.server_url]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        self._start_log_thread(proc, f"{spec.name}-{index}")
        return proc
    
    def _start_sample_bot(self, spec: BotSpec, index: int) -> subprocess.Popen:
        """Start a sample bot using a launcher script."""
        bot_dir = os.path.abspath(spec.path)
        bot_name = spec.name
        
        # Create launcher script
        launcher_content = f'''
import sys
import os
import asyncio
import json

# Add bot directory to path
sys.path.insert(0, '{bot_dir}')

# Read JSON info for bot metadata
json_path = os.path.join('{bot_dir}', '{bot_name}.json')
info = {{}}
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        info = json.load(f)

# Map JSON keys to environment variables
key_map = {{
    'name': 'BOT_NAME', 
    'version': 'BOT_VERSION', 
    'authors': 'BOT_AUTHORS',
    'description': 'BOT_DESCRIPTION', 
    'homepage': 'BOT_HOMEPAGE',
    'countryCodes': 'BOT_COUNTRY_CODES', 
    'gameTypes': 'BOT_GAME_TYPES',
    'platform': 'BOT_PLATFORM', 
    'programmingLang': 'BOT_PROGRAMMING_LANG'
}}

# Merge with defaults
data = {{'gameTypes': 'classic,melee,1v1', **info}}

for k, v in data.items():
    if k in key_map:
        val = ','.join(v) if isinstance(v, list) else str(v)
        os.environ[key_map[k]] = val

# Set server URL
os.environ['SERVER_URL'] = '{self.server_url}'
os.environ['ROBOCODE_SERVER_URL'] = '{self.server_url}'

# Import and run the bot
import {bot_name}
bot = {bot_name}.{bot_name}()
asyncio.run(bot.start())
'''
        
        # Write launcher to temp file
        fd, launcher_path = tempfile.mkstemp(suffix='.py', prefix=f'launch_{bot_name}_')
        with os.fdopen(fd, 'w') as f:
            f.write(launcher_content)
        self.launcher_scripts.append(launcher_path)
        
        # Start the process
        proc = subprocess.Popen(
            [sys.executable, launcher_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        self._start_log_thread(proc, f"{spec.name}-{index}")
        return proc
    
    def _start_log_thread(self, proc: subprocess.Popen, name: str):
        """Start a background thread to log process output."""
        def _log_output():
            for line in iter(proc.stdout.readline, b''):
                logger.debug(f"[Opponent-{name}] {line.decode().strip()}")
        
        thread = threading.Thread(target=_log_output, daemon=True)
        thread.start()
        self._log_threads.append(thread)
    
    def stop_all(self) -> int:
        """
        Stop all opponent processes and clean up.
        
        Returns:
            Number of processes terminated
        """
        count = 0
        for proc in self.processes:
            try:
                if hasattr(os, 'getpgid') and hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=2)
                count += 1
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
                try:
                    proc.kill()
                except:
                    pass
        
        self.processes.clear()
        self.current_opponents.clear()
        
        # Clean up launcher scripts
        self._cleanup_launcher_scripts()
        
        logger.info(f"Stopped {count} opponent(s)")
        return count
    
    def _cleanup_launcher_scripts(self):
        """Remove temporary launcher script files."""
        for path in self.launcher_scripts:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.debug(f"Failed to remove launcher script {path}: {e}")
        self.launcher_scripts.clear()
    
    def get_active_count(self) -> int:
        """Get number of currently active (running) opponent processes."""
        return sum(1 for p in self.processes if p.poll() is None)
    
    def get_crashed_count(self) -> int:
        """Get number of crashed opponent processes."""
        return sum(1 for p in self.processes if p.poll() is not None)
    
    def restart_crashed(self) -> int:
        """
        Restart any crashed opponent processes.
        
        Returns:
            Number of processes restarted
        """
        crashed_indices = []
        for i, proc in enumerate(self.processes):
            if proc.poll() is not None:
                crashed_indices.append(i)
        
        if not crashed_indices:
            return 0
        
        logger.warning(f"Found {len(crashed_indices)} crashed opponent(s), restarting...")
        
        # Restart each crashed opponent
        restarted = 0
        for i in crashed_indices:
            if i < len(self.current_opponents):
                bot_name = self.current_opponents[i]
                spec = self.registry.get(bot_name)
                if spec:
                    new_proc = self._start_single_opponent(spec, i)
                    if new_proc:
                        self.processes[i] = new_proc
                        restarted += 1
        
        return restarted
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all opponent processes."""
        return {
            "active": self.get_active_count(),
            "crashed": self.get_crashed_count(),
            "total": len(self.processes),
            "opponents": self.current_opponents.copy(),
            "pool_size": len(self.opponent_pool),
        }
