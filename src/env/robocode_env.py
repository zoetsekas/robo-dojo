import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import time
import os
import subprocess
import socket
import atexit
import signal
import logging
import sys
from .gym_bot import GymBot
from .video_capture import VideoCapture, EpisodeRecorder, ensure_window_manager
from .opponent_manager import OpponentManager, get_bot_registry
from robocode_tank_royale.bot_api import BotInfo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [RoboEnv] %(message)s')
logger = logging.getLogger(__name__)


class RobocodeGymEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, env_config=None):
        env_config = env_config or {}
        
        # Robust extraction for different RLlib versions
        worker_idx = getattr(env_config, "worker_index", env_config.get("worker_index", 0))
        vector_idx = getattr(env_config, "vector_index", env_config.get("vector_index", 0))
        pid = os.getpid()
        
        self.opponent_type = env_config.get("opponent_type", "random")
        self.num_opponents = env_config.get("num_opponents", 1)
        self.opponent_pool = env_config.get("opponent_pool", None)  # None = all bots
        
        # 1. Port/Display allocation based on worker and vector index
        # Using a wider range to avoid collisions
        self.port = 8000 + (worker_idx * 10) + (vector_idx * 2)
        
        # 3. Combine for Display
        self.display_num = 100 + (worker_idx % 100) + vector_idx
        self.display_str = f":{self.display_num}"
        
        hostname = socket.gethostname()
        logger.info(f"Initializing Environment (Worker {worker_idx}, Vector {vector_idx}, PID {pid})")
        logger.info(f"  Hostname: {hostname}, Port: {self.port}, Display: {self.display_str}")
        
        self.server_url = f"ws://127.0.0.1:{self.port}"
        # Unique name to identify this specific worker/vector's bot
        self.bot_name = f"RoboBot_W{worker_idx}_V{vector_idx}"

        # KILL STALE PROCESSES on this worker
        # This is critical if the container/worker restarted without cleaning up.
        self._cleanup_stale_processes()
        
        # GUI and Display configuration
        self.use_xvfb = env_config.get("use_xvfb", True)
        self.use_gui = env_config.get("use_gui", False)
        self.use_visual_obs = env_config.get("use_visual_obs", True)  # Vector-only mode if False
        self.export_tick_data = env_config.get("export_tick_data", False)  # Export tick data to JSON
        self.external_display = os.environ.get("ROBODOJO_DISPLAY")
        
        # Tick data collection (for debugging/analysis)
        self._tick_data = []

        
        if self.external_display:
            logger.info(f"Using external display: {self.external_display}")
            self.display_str = self.external_display
            self.use_xvfb = False
        else:
            # display_num and display_str already set via PID above
            logger.info(f"Using internal Xvfb: {self.display_str}")

        self.processes = []
        self.bot = None
        self.bot_thread = None
        self.stop_event = threading.Event()

        self.video_capture = None
        self.last_img = np.zeros((84, 84, 3), dtype=np.uint8)
        self.last_event = None
        
        # Multi-enemy tracking (up to 3 enemies, sorted by distance)
        self.max_tracked_enemies = 3
        self.tracked_enemies = [
            {"id": -1, "x": 0.0, "y": 0.0, "speed": 0.0, "heading": 0.0, "energy": 0.0, "distance": 9999.0}
            for _ in range(self.max_tracked_enemies)
        ]
        
        # Combat statistics (reset each episode)
        self.combat_stats = {
            "bullets_fired": 0, "hits_dealt": 0,
            "damage_dealt": 0.0, "damage_taken": 0.0
        }
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0
        
        # Rollout robustness tracking
        self._needs_force_reset = False  # Set when timeout occurs to force clean reset
        self._needs_rebuild_infra = False # Set when server is likely crashed and needs restart
        self._consecutive_timeouts = 0    # Track consecutive timeouts for health monitoring
        self._max_consecutive_timeouts = 3  # After this many, force full environment restart
        self._step_timeout_s = env_config.get("step_timeout_s", 5.0)  # Configurable timeout
        self._last_tick_time = time.time()  # Track bot responsiveness
        
        # Episode recorder (record every N episodes)
        record_every = env_config.get("record_every_n_episodes", 100)
        self.episode_recorder = EpisodeRecorder(record_every_n=record_every)
        
        # Tick data export (for debugging - saves game state to JSON)
        self._tick_data = []  # List of tick data for current episode
        self._export_tick_data = env_config.get("smoke_test", False)  # Only during smoke test
        
        # Initialize opponent manager (opponents started per-episode in reset())
        self.opponent_manager = None  # Created after server starts
        
        # Start headless infrastructure (server only, not opponents)
        self._start_infrastructure()

        # Observation Space: conditional based on use_visual_obs
        # Vector: 37 dims = 13 self-state + 18 enemies (3×6) + 4 combat + 2 reserved
        if self.use_visual_obs:
            # Multimodal: visual + vector
            self.observation_space = spaces.Dict({
                "visual_obs": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
                "vector_obs": spaces.Box(low=-float('inf'), high=float('inf'), shape=(37,), dtype=np.float32)
            })
            logger.info("Observation mode: MULTIMODAL (visual + vector)")
        else:
            # Vector-only: simple Box space
            self.observation_space = spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(37,), dtype=np.float32
            )
            logger.info("Observation mode: VECTOR-ONLY (37 dims)")
        
        # Action Space: [target_speed, turn_rate, gun_turn_rate, radar_turn_rate, fire]
        self.action_space = spaces.Box(
            low=np.array([-8, -10, -20, -45, 0], dtype=np.float32),
            high=np.array([8, 10, 20, 45, 3], dtype=np.float32),
            dtype=np.float32
        )
        
        # Register cleanup
        atexit.register(self.close)

    def _start_infrastructure(self):
        """Starts Xvfb, Server, and GUI for this isolated env."""
        logger.info(f"Infrastructure: Port={self.port}, DISPLAY={self.display_str}")
        
        # 0. Cleanup EXISTING infrastructure if this is a rebuild
        if self.processes:
            logger.info("Cleaning up existing infrastructure before rebuild...")
            for proc in self.processes:
                try:
                    # Kill the whole process group
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    try: proc.kill()
                    except: pass
            self.processes = []
            time.sleep(2) # Give OS time to release ports/locks

        if self.use_xvfb:
            # 1. Cleanup any stale lock files
            lock_file = f"/tmp/.X{self.display_num}-lock"
            if os.path.exists(lock_file):
                logger.info(f"Force-removing stale lock {lock_file}")
                try: os.remove(lock_file)
                except: pass

            # 2. Xvfb - Run in background (1024x768 is more robust for Robocode)
            # Disable MIT-SHM as it often causes issues in Docker
            # -ac: Disable access control checks
            # -nolisten tcp: Don't listen for TCP connections
            xvfb_cmd = ["Xvfb", self.display_str, "-ac", "-screen", "0", "1024x768x24", "-nolisten", "tcp", "-extension", "MIT-SHM"]
            self.xvfb_proc = subprocess.Popen(xvfb_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            self.processes.append(self.xvfb_proc)
            
            # Create a dummy Xauthority for this worker to avoid permission issues
            xauth_file = f"/tmp/.Xauthority-{self.display_num}"
            os.environ["XAUTHORITY"] = xauth_file
            
            # Start Xvfb logging thread
            def _log_xvfb(proc, disp):
                try:
                    for line in iter(proc.stdout.readline, b''):
                        msg = line.decode('utf-8', errors='replace').strip()
                        if msg: logger.debug(f"[Xvfb-{disp}] {msg}")
                except: pass
            threading.Thread(target=_log_xvfb, args=(self.xvfb_proc, self.display_str), daemon=True).start()
            
            # Wait until Xvfb is actually responding
            logger.info(f"Waiting for Xvfb on {self.display_str}...")
            # Use xdpyinfo to wait for X server to be ready
            xvfb_ready = False
            for _ in range(30):
                if subprocess.run(f"xdpyinfo -display {self.display_str}", shell=True, capture_output=True).returncode == 0:
                    xvfb_ready = True
                    break
                time.sleep(0.5)
            
            if not xvfb_ready:
                logger.error(f"Xvfb failed to start on {self.display_str}")
            else:
                logger.info(f"X server is ready on {self.display_str}")
        
        # Set DISPLAY for the current worker process and its children
        os.environ["DISPLAY"] = self.display_str
        
        # 1.5 Window Manager (Fluxbox) - Required for Java AWT to map windows correctly in Xvfb
        if self.use_xvfb:
            # We set DBUS and AT_BRIDGE to /dev/null to prevent Java hangs
            wm_env = os.environ.copy()
            wm_env["DISPLAY"] = self.display_str
            wm_env["DBUS_SESSION_BUS_ADDRESS"] = "/dev/null"
            wm_env["NO_AT_BRIDGE"] = "1"
            
            # Use fluxbox as it's often more stable for headless Java than openbox
            wm_cmd = ["fluxbox", "-display", self.display_str]
            self.wm_proc = subprocess.Popen(wm_cmd, env=wm_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            self.processes.append(self.wm_proc)
            
            # Start WM logging thread
            def _log_wm(proc, disp):
                try:
                    for line in iter(proc.stdout.readline, b''):
                        msg = line.decode('utf-8', errors='replace').strip()
                        if msg: logger.debug(f"[WM-{disp}] {msg}")
                except: pass
            threading.Thread(target=_log_wm, args=(self.wm_proc, self.display_str), daemon=True).start()
            
            logger.info(f"Window manager (fluxbox) started on {self.display_str}")
            time.sleep(2)
            
            # OPTIONAL: Start x11vnc for remote debugging (mapped via docker if port published)
            # Use a unique port for each display if possible, but for smoke-test Display 100/101 is fine
            try:
                vnc_port = 5900 + (self.display_num - 100)
                vnc_cmd = ["x11vnc", "-display", self.display_str, "-forever", "-nopw", "-rfbport", str(vnc_port), "-shared"]
                vnc_proc = subprocess.Popen(vnc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
                self.processes.append(vnc_proc)
                logger.debug(f"VNC server started on port {vnc_port} for {self.display_str}")
            except Exception as e:
                logger.warning(f"Failed to start VNC: {e}")
        
        # Essential for Java AWT to work correctly with non-reparenting WMs like Openbox/Fluxbox
        os.environ["_JAVA_AWT_WM_NONREPARENTING"] = "1"
        self.video_capture = None
        
        # Only initialize video capture if visual observations are enabled
        if self.use_visual_obs:
            for i in range(20):
                if self.use_xvfb and self.xvfb_proc.poll() is not None:
                    logger.error(f"Xvfb DIED early.")
                    break

                try:
                    from .video_capture import VideoCapture
                    self.video_capture = VideoCapture(display=self.display_str)
                    # Test a grab
                    logger.info("Xvfb is UP and VideoCapture initialized.")
                    break
                except Exception:
                    time.sleep(1)
            
            if self.video_capture is None:
                raise RuntimeError(f"Xvfb failed to respond on {self.display_str}")
        else:
            logger.info("Skipping VideoCapture initialization (vector-only mode)")

        # 3. Server (Retry with random port if needed)
        server_jar = "/robocode/server.jar"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Increase memory to 1GB for better stability at high throughput
                # FORCE bind to 127.0.0.1 to prevent cross-container discovery
                server_cmd = [
                    "java", "-Xmx1024m", 
                    "-Djava.net.preferIPv4Stack=true", 
                    "-Djava.security.egd=file:/dev/./urandom",
                    "-jar", server_jar, 
                    "--port", str(self.port)
                ]


                self.server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
                self.processes.append(self.server_proc)
                
                # Start logging immediately
                def _log_server(proc, port):
                    try:
                        for line in iter(proc.stdout.readline, b''):
                            logger.info(f"[Server-{port}] {line.decode().strip()}")
                    except: pass
                threading.Thread(target=_log_server, args=(self.server_proc, self.port), daemon=True).start()

                # Wait loop to verify server is actually listening
                server_up = False
                for _ in range(40): # Wait up to 20 seconds (increased from 15)
                    time.sleep(0.5)
                    if self.server_proc.poll() is not None:
                         raise RuntimeError("Server process died immediately")
                    
                    # Try to connect to port to verify it's open
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                         if s.connect_ex(('127.0.0.1', self.port)) == 0:
                             server_up = True
                             break
                
                if not server_up:
                     raise RuntimeError(f"Server started but port {self.port} never opened.")
                
                # If we got here, server is up and listening
                break
            except Exception as e:
                logger.warning(f"Failed to start server on port {self.port} (attempt {attempt}): {e}")
                # cleanup failed proc
                try: 
                    self.server_proc.terminate()
                    self.server_proc.wait()
                except: pass
                
                import random
                self.port = random.randint(10000, 20000)
                self.server_url = f"ws://127.0.0.1:{self.port}"
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to start Robocode server after {max_retries} attempts.")

        self.processes.append(self.server_proc)
        
        time.sleep(2)  # Wait for server to fully stabilize


        # 3. GUI (optional - for visual debugging)
        if self.use_gui:
            # Use wrapper script that handles all Java 2D software rendering options
            # This ensures options like opengl=false, xrender=false are definitely applied
            gui_wrapper = "/app/scripts/gui_wrapper.sh"
            gui_cmd = [
                gui_wrapper,
                "--server-url", self.server_url, 
                "--no-sound"
            ]
            # Inherit environment and add window manager hints
            env = os.environ.copy()
            env["_JAVA_AWT_WM_NONREPARENTING"] = "1"
            env["AWT_TOOLKIT"] = "XToolkit"  # Force X11 toolkit, not abstract
            env["DISPLAY"] = self.display_str
            env["XAUTHORITY"] = os.environ.get("XAUTHORITY", "")
            env["DBUS_SESSION_BUS_ADDRESS"] = "/dev/null"
            env["NO_AT_BRIDGE"] = "1"
            env["LC_ALL"] = "C"  # Consistency for window title matching

            self.gui_proc = subprocess.Popen(gui_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
            
            # Start GUI logging thread
            def _log_gui(proc, port):
                logger.debug(f"GUI logging thread for port {port} started.")
                try:
                    for line in iter(proc.stdout.readline, b''):
                        msg = line.decode('utf-8', errors='replace').strip()
                        if msg:
                            logger.info(f"[GUI-{port}] {msg}")
                except Exception as e:
                    logger.debug(f"GUI logging error on port {port}: {e}")
                finally:
                    logger.debug(f"GUI logging thread for port {port} stopped.")
            threading.Thread(target=_log_gui, args=(self.gui_proc, self.port), daemon=True).start()

            self.processes.append(self.gui_proc)
            logger.info(f"Server and GUI started on {self.port}.")
        else:
            logger.info(f"Server started on {self.port} (GUI disabled).")
        
        # INCREASED WAIT: GUI (Java Swing) takes a few seconds to initialize and draw
        logger.info("Waiting for GUI window to appear...")
        
        # 3.5 Wait for and Maximize GUI Window (ensure it fills 800x600)
        if self.use_gui and self.use_xvfb:
            found = False
            for _ in range(20):  # Wait up to 20 seconds for window
                time.sleep(1)
                
                # Check if process is still alive
                if self.gui_proc.poll() is not None:
                    logger.error(f"GUI process died with exit code {self.gui_proc.returncode}")
                    break

                # Check if 'robocode', 'tank', or 'royale' appears in window list
                try:
                    # Use a safer way to call wmctrl that won't crash the whole process
                    window_list_proc = subprocess.run(
                        f"DISPLAY={self.display_str} wmctrl -l", 
                        shell=True, 
                        capture_output=True, 
                        text=True
                    )
                    window_list = window_list_proc.stdout
                    # Match any window that looks like robocode or the tank royale app
                    if window_list_proc.returncode == 0:
                        if window_list.strip():
                            logger.debug(f"Detected windows: {window_list.strip()}")
                        
                        target_title = None
                        if 'Robocode' in window_list: target_title = 'Robocode'
                        elif 'Tank' in window_list: target_title = 'Tank'
                        elif 'Royale' in window_list: target_title = 'Royale'

                        if target_title:
                            logger.info(f"Detected GUI window: {target_title}")
                            # Try to maximize it
                            os.system(f"DISPLAY={self.display_str} wmctrl -r '{target_title}' -b add,maximized_vert,maximized_horz")
                            found = True
                            break
                except Exception as e:
                    # wmctrl can fail if the WM isn't ready yet, just log and retry
                    if _ == 0: # Only log once to avoid spam
                        logger.debug(f"wmctrl not ready yet: {e}")
                    pass
            
            if not found:
                # Check if the process even survived
                exit_code = self.gui_proc.poll()
                if exit_code is not None:
                    logger.error(f"GUI process crashed with exit code {exit_code} before window appeared!")
                else:
                    logger.warning("Could not detect Robocode GUI window via wmctrl. This often happens if the GUI takes too long to map. Continuing anyway...")
            else:
                logger.info("GUI window successfully localized and maximized.")
                # Give Java time to actually render pixels after maximize
                time.sleep(3)
                # Take a diagnostic screenshot to /app/artifacts (volume-mounted to host)
                try:
                    os.makedirs("/app/artifacts", exist_ok=True)
                    screenshot_path = f"/app/artifacts/gui_screenshot_{self.port}.png"
                    os.system(f"DISPLAY={self.display_str} import -window root {screenshot_path}")
                    logger.info(f"Saved diagnostic screenshot to {screenshot_path}")
                except:
                    pass
            
        # 4. Initialize OpponentManager (opponents started per-episode in reset())
        self.opponent_manager = OpponentManager(
            server_url=self.server_url,
            opponent_pool=self.opponent_pool,
            registry=get_bot_registry()
        )
        logger.info(f"OpponentManager initialized with pool: {self.opponent_manager.available_bots}")


    def _cleanup_stale_processes(self):
        """Forcibly kill any Robocode-related processes lingering on this node."""
        logger.info(f"Cleaning up stale processes for port {self.port}...")
        try:
            # Kill Java server processes using this port
            os.system(f"pkill -9 -f 'server.jar.*--port {self.port}'")
            # Kill any python bots or other processes connected to this server, 
            # but EXCLUDE the GUI and the server themselves if they happen to match.
            os.system(f"pgrep -f '{self.server_url}' | grep -v 'gui.jar' | grep -v 'server.jar' | xargs kill -9 2>/dev/null")
            # Kill any Xvfb on our display
            os.system(f"pkill -9 -f 'Xvfb {self.display_str}'")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error during stale process cleanup: {e}")

    def _hard_reset(self):
        """Forcibly stop all bot threads and clean up the loop."""
        logger.info("Executing HARD RESET of bot lifecycle...")
        self.stop_event.set()
        
        if self.bot:
            try:
                # We can't easily await from here, but we can signal the bot
                # The thread will handle the cleanup
                pass
            except Exception: pass
            
        if self.bot_thread and self.bot_thread.is_alive():
            logger.info("Joining bot thread...")
            # We don't want to block forever, but we need it to stop
            self.bot_thread.join(timeout=3)
            if self.bot_thread.is_alive():
                logger.warning("Bot thread still alive after join timeout!")

        self.bot = None
        self.bot_thread = None
        self.stop_event.clear()
        
        # Ensure ALL bot processes associated with this server are dead
        # EXCLUDE the GUI and the server themselves from this cleanup
        if hasattr(self, 'server_url'):
            os.system(f"pgrep -f '{self.server_url}' | grep -v 'gui.jar' | grep -v 'server.jar' | xargs kill -9 2>/dev/null")
            
        # Kill any lingering controller processes for this port
        os.system(f"pkill -9 -f 'src.env.robocode_controller {self.server_url}'")

    def close(self):
        """Full cleanup of all resources."""
        logger.info("Closing RobocodeGymEnv...")
        self._hard_reset()
        
        # Stop opponent processes via manager
        if self.opponent_manager:
            self.opponent_manager.stop_all()
        
        # Terminate all other processes (server, GUI, Xvfb)
        for proc in self.processes:
            try:
                # Kill the entire process group if possible
                if hasattr(os, 'getpgid'):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                try: proc.kill()
                except Exception: pass
        
        self.processes = []
        
        if self.video_capture:
            self.video_capture.stop_recording()
        
        logger.info("RobocodeGymEnv closed.")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Check if we need to rebuild the whole infrastructure (server + Xvfb)
        if self._needs_rebuild_infra:
            logger.warning("REBUILDING ENVIRONMENT INFRASTRUCTURE (Server/Xvfb)...")
            self._start_infrastructure()
            self._needs_rebuild_infra = False
            self.episode_count = 0 # Reset count since this is a fresh start
        
        # 2. Reset health state
        self._needs_force_reset = False
        self._consecutive_timeouts = 0
        self._last_tick_time = time.time()
        
        # Save tick data from previous episode (if any)
        if self._export_tick_data and self._tick_data and self.episode_count > 0:
            self._save_tick_data()
        self._tick_data = []  # Clear for new episode
        
        # 3. Hard reset bot lifecycle to ensure ZERO bot accumulation
        self._hard_reset()
        
        # Per-episode opponent selection and startup
        if self.opponent_manager:
            self.opponent_manager.stop_all()
            
            # Select opponent(s) for this episode
            if self.opponent_type == "random":
                selected = self.opponent_manager.select_random_opponent()
                logger.info(f"Episode {self.episode_count + 1}: Randomly selected opponent '{selected}'")
                self.opponent_manager.start_opponents([selected], self.num_opponents)
            else:
                self.opponent_manager.start_opponents([self.opponent_type], self.num_opponents)
            
            # Give opponents time to connect to server
            time.sleep(3)
        
        bot_info = BotInfo(
            name=self.bot_name,
            version="1.0",
            authors=["RoboDojo"],
            game_types=["melee", "1v1"],
            description="Multimodal RL Agent"
        )
        
        try:
            # Wait for port to be open before starting bot
            logger.info(f"Waiting for server port {self.port} to be open...")
            connected = False
            for i in range(30):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    res = s.connect_ex(('127.0.0.1', self.port))
                    if res == 0:
                        connected = True
                        break
                time.sleep(1)
            
            if not connected:
                raise RuntimeError(f"Server port {self.port} never opened.")

            # Start GymBot (our trainable agent)
            self.bot = GymBot(bot_info, self.server_url)
            
            def _start_bot_loop(stop_ev, bot):
                import asyncio
                import logging
                loop_logger = logging.getLogger("RoboBotLoop")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def _run_bot():
                    """Internal task to keep the bot connected."""
                    while not stop_ev.is_set():
                        try:
                            # start() blocks until disconnected or stop() is called
                            await bot.start()
                        except Exception as e:
                            if stop_ev.is_set():
                                break
                            loop_logger.error(f"Bot connection error: {e}")
                            await asyncio.sleep(1)
                
                bot_task = loop.create_task(_run_bot())
                
                # Monitor stop event to signal bot to stop
                async def _monitor_stop():
                    while not stop_ev.is_set():
                        await asyncio.sleep(0.2)
                    
                    loop_logger.info("Stop event detected, initiating bot shutdown...")
                    # Signal the bot to stop its internal loop and disconnect
                    try:
                        if bot.is_running():
                            await bot.stop_bot()
                    except Exception as e:
                        loop_logger.debug(f"Exception during bot.stop_bot(): {e}")
                    
                    # Cancel the main run task if it's still alive
                    if not bot_task.done():
                        bot_task.cancel()

                monitor_task = loop.create_task(_monitor_stop())
                
                try:
                    # Wait for the main bot task to complete or be cancelled
                    loop.run_until_complete(bot_task)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    loop_logger.error(f"Unexpected error in bot task: {e}")
                finally:
                    # CRITICAL: Clean up ALL pending tasks before closing loop
                    # This prevents "Task was destroyed but it is pending" and "Event loop is closed" errors
                    loop_logger.info("Cleaning up pending tasks...")
                    
                    # 1. Cancel monitor task if it hasn't finished
                    if not monitor_task.done():
                        monitor_task.cancel()
                    
                    # 2. Identify all remaining tasks
                    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                    
                    if pending:
                        loop_logger.debug(f"Cancelling {len(pending)} pending tasks...")
                        for task in pending:
                            task.cancel()
                        
                        # 3. Allow tasks a final chance to process cancellation/cleanup
                        try:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        except Exception as e:
                            loop_logger.debug(f"Error during task gather: {e}")
                    
                    # 4. Final closure
                    try:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()
                        loop_logger.info("Bot event loop closed successfully.")
                    except Exception as e:
                        loop_logger.warning(f"Error during loop closure: {e}")
            
            self.bot_thread = threading.Thread(
                target=_start_bot_loop, 
                args=(self.stop_event, self.bot), 
                daemon=True
            )
            self.bot_thread.start()
            time.sleep(3)  # Wait for GymBot to connect
            
            # Now trigger the game
            total_bots = self.num_opponents + 1
            logger.info(f"Starting match controller (Total Bots: {total_bots})")
            def _run_controller():
                try:
                    cmd = ["python", "-m", "src.env.robocode_controller", self.server_url, str(total_bots)]
                    subprocess.run(cmd, capture_output=True, text=True)
                except Exception as e:
                    logger.error(f"Controller error: {e}")
            
            threading.Thread(target=_run_controller, daemon=True).start()
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            raise
        
        # Episode start tracking
        self.episode_count += 1
        self.step_count = 0
        self.total_reward = 0
        self._tick_data = []  # Clear tick history for new episode
        
        # Reset enemy/combat tracking for new episode
        self.tracked_enemies = [
            {"id": -1, "x": 0.0, "y": 0.0, "speed": 0.0, "heading": 0.0, "energy": 0.0, "distance": 9999.0}
            for _ in range(self.max_tracked_enemies)
        ]
        self.combat_stats = {"bullets_fired": 0, "hits_dealt": 0, "damage_dealt": 0.0, "damage_taken": 0.0}
        
        # Start recording if applicable
        if self.video_capture:
            self.episode_recorder.on_episode_start(self.video_capture)
        
        # Clear any stale events from previous episode
        if self.bot:
            self.bot.get_events()  # Drain queue
        
        self.last_event = None
        logger.info(f"Episode {self.episode_count} started")
        return self._get_obs(), {}

    def step(self, action):
        # Check if we need a force reset due to previous timeout
        if self._needs_force_reset:
            logger.warning("Force reset required due to previous timeout, returning done=True")
            self._needs_force_reset = False
            return self._get_obs(), 0.0, True, False, {"force_reset": True}
        
        if self.bot:
             self.bot.send_action(action)
        
        # Check for crashed opponents and restart if needed
        if self.opponent_manager and self.opponent_manager.get_crashed_count() > 0:
            logger.warning("Detected crashed opponent(s), attempting restart...")
            self.opponent_manager.restart_crashed()
        
        done = False
        truncated = False
        accumulated_reward = 0.0  # Buffer rewards from multiple events
        info = {}
        
        start_wait = time.time()
        tick_event = None
        events_processed = 0
        
        # Event buffering loop - accumulate all events until we get a tick
        while True:
            elapsed = time.time() - start_wait
            
            # Timeout check with configurable duration
            if elapsed > self._step_timeout_s:
                self._consecutive_timeouts += 1
                logger.warning(f"Step timeout after {elapsed:.1f}s (consecutive: {self._consecutive_timeouts})")
                
                # Check if we've hit too many consecutive timeouts
                if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                    logger.error(f"Hit {self._consecutive_timeouts} consecutive timeouts, forcing server REBUILD")
                    self._needs_force_reset = True
                    self._needs_rebuild_infra = True
                    info["timeout_reset"] = True
                
                truncated = True
                info["timeout"] = True
                break
            
            # Bot connection check
            if not self.bot or not self.bot.is_running():
                done = True
                info["bot_disconnected"] = True
                logger.warning("Bot disconnected during step")
                break

            # Non-blocking event queue check
            if self.bot.event_queue.empty():
                time.sleep(0.001)
                continue
            
            event = self.bot.event_queue.get()
            events_processed += 1
            
            # Process event based on type
            event_type = event.get("type", "unknown")
            
            if event_type == "tick":
                tick_event = event
                self._last_tick_time = time.time()
                self._consecutive_timeouts = 0  # Reset timeout counter on successful tick
                
                # Capture frame for visual observation
                if self.video_capture:
                    full_frame = self.video_capture.grab_frame()
                    self.last_img = self.video_capture.process_for_model(full_frame)
                    
                    # DEBUG: Save captured frame periodically for smoke test verification
                    if self.step_count % 50 == 0 and self.step_count > 0:
                        try:
                            import cv2
                            debug_path = f"/app/artifacts/debug_frame_ep{self.episode_count}_step{self.step_count}.png"
                            # full_frame is BGR, save directly
                            cv2.imwrite(debug_path, full_frame)
                            logger.debug(f"Saved debug frame to {debug_path}")
                        except Exception as e:
                            logger.debug(f"Debug frame save failed: {e}")
                
                # Collect tick data for export (smoke test debugging)
                if self._export_tick_data and "obs" in tick_event:
                    tick_record = {
                        "step": self.step_count,
                        "turn": tick_event.get("turn", 0),
                        "obs": tick_event["obs"],
                        "action": list(action) if hasattr(action, '__iter__') else action
                    }
                    self._tick_data.append(tick_record)
                
                break
                
            elif event_type == "death":
                done = True
                accumulated_reward += -1.0
                logger.info(f"Episode {self.episode_count} ended: DEATH after {self.step_count} steps, reward={self.total_reward:.2f}")
                break
                
            elif event_type == "win":
                done = True
                accumulated_reward += 1.0
                logger.info(f"Episode {self.episode_count} ended: WIN after {self.step_count} steps, reward={self.total_reward:.2f}")
                break

            elif event_type in ["round_end", "game_end"]:
                done = True
                logger.debug(f"Episode {self.episode_count} ended: {event_type}")
                break
                
            # Accumulate intermediate rewards (these continue to tick)
            elif event_type == "hit_bot":
                accumulated_reward += 0.5
                logger.info(f"[Step {self.step_count}] COLLISION: Hit bot {event.get('victim_id')}!")
            elif event_type == "hit_wall":
                accumulated_reward -= 0.1
                logger.debug(f"[Step {self.step_count}] Wall hit")
            elif event_type == "bullet_hit":
                accumulated_reward += 0.3  # Our bullet hit enemy
                # Update combat stats
                self.combat_stats["hits_dealt"] += 1
                self.combat_stats["damage_dealt"] += event.get("damage", 0)
                logger.info(f"[Step {self.step_count}] ATTACK: Bullet hit enemy {event.get('victim_id')} (Damage: {event.get('damage', 0):.1f})")
            elif event_type == "hit_by_bullet":
                accumulated_reward -= 0.2  # We got hit
                # Update combat stats
                self.combat_stats["damage_taken"] += event.get("damage", 0)
                logger.info(f"[Step {self.step_count}] DEFENSE: Hit by bullet (Damage: {event.get('damage', 0):.1f})")
            elif event_type == "scanned":
                accumulated_reward += 0.05  # Radar found enemy
                # Update multi-enemy tracking with scanned data
                my_x = self.last_event.get("obs", {}).get("x", 0) if self.last_event else 0
                my_y = self.last_event.get("obs", {}).get("y", 0) if self.last_event else 0
                enemy_x = event.get("x", 0)
                enemy_y = event.get("y", 0)
                enemy_id = event.get("enemy_id", -1)
                distance = ((enemy_x - my_x)**2 + (enemy_y - my_y)**2)**0.5
                
                new_enemy = {
                    "id": enemy_id, "x": enemy_x, "y": enemy_y,
                    "speed": event.get("speed", 0),
                    "heading": event.get("direction", 0),
                    "energy": event.get("energy", 0),
                    "distance": distance
                }
                
                # Update existing entry or add new one
                updated = False
                for i, tracked in enumerate(self.tracked_enemies):
                    if tracked["id"] == enemy_id:
                        self.tracked_enemies[i] = new_enemy
                        updated = True
                        break
                
                if not updated:
                    # Add to list if there's a slot with default id (-1) or replace furthest
                    self.tracked_enemies.append(new_enemy)
                
                # Sort by distance and keep only top N
                self.tracked_enemies.sort(key=lambda e: e["distance"])
                self.tracked_enemies = self.tracked_enemies[:self.max_tracked_enemies]
            elif event_type == "skipped_turn":
                accumulated_reward -= 0.5  # Penalty for being too slow
                logger.warning(f"[Step {self.step_count}] SKIPPED TURN: CPU lag detected")
        
        # Update tracking
        self.step_count += 1
        self.total_reward += accumulated_reward
        
        # Periodic step log for progress tracking
        if self.step_count % 100 == 0:
            energy = 0
            if tick_event:
                energy = tick_event.get("obs", {}).get("energy", 0)
            logger.info(f"Episode {self.episode_count} Step {self.step_count}: Reward={self.total_reward:.2f}, Energy={energy:.1f}")
            
        info["events_processed"] = events_processed
        info["step_time"] = time.time() - start_wait
        
        # Collection and Export logic
        if self.export_tick_data and tick_event:
            tick_snapshot = {
                "step": self.step_count,
                "reward": accumulated_reward,
                "total_reward": self.total_reward,
                "obs": tick_event.get("obs", {}),
                "tracked_enemies": self.tracked_enemies.copy(),
                "combat_stats": self.combat_stats.copy()
            }
            self._tick_data.append(tick_snapshot)

        # Stop recording and save tick data if episode ended
        if done:
            if self.video_capture:
                self.episode_recorder.on_episode_end(self.video_capture)
            
            if self.export_tick_data:
                self._save_tick_data()
                
        self.last_event = tick_event
        return self._get_obs(), accumulated_reward, done, truncated, info

    def _get_obs(self):
        """Build 37-dimensional observation vector for multi-enemy support.
        
        Layout:
        - [0-12]  Self state (13 dims): x, y, speed, energy, heading, gun_heading, radar_heading,
                  gun_heat, enemy_count, radar_sweep, turn_rate, gun_turn_rate, radar_turn_rate
        - [13-30] Enemy states (18 dims): 3 enemies × 6 dims (x, y, speed, heading, energy, distance)
        - [31-34] Combat stats (4 dims): bullets_fired, hits_dealt, damage_dealt, damage_taken
        - [35-36] Reserved (2 dims): padding for future use
        """
        if self.last_event and "obs" in self.last_event:
            o = self.last_event["obs"]
            c = self.combat_stats
            
            # Self state (13 dims)
            self_state = [
                o["x"] / 800.0, 
                o["y"] / 600.0, 
                o["speed"] / 8.0, 
                o["energy"] / 100.0, 
                o["heading"] / 360.0,
                o["gun_heading"] / 360.0, 
                o["radar_heading"] / 360.0, 
                o["gun_heat"] / 3.0, 
                o["enemy_count"] / 10.0,
                o.get("radar_sweep", 0) / 360.0,
                o.get("turn_rate", 0) / 10.0,
                o.get("gun_turn_rate", 0) / 20.0,
                o.get("radar_turn_rate", 0) / 45.0,
            ]
            
            # Enemy states (18 dims = 3 enemies × 6 dims each)
            enemy_state = []
            for i in range(self.max_tracked_enemies):
                e = self.tracked_enemies[i] if i < len(self.tracked_enemies) else {
                    "x": 0.0, "y": 0.0, "speed": 0.0, "heading": 0.0, "energy": 0.0, "distance": 9999.0
                }
                enemy_state.extend([
                    e["x"] / 800.0,
                    e["y"] / 600.0,
                    e["speed"] / 8.0,
                    e["heading"] / 360.0,
                    e["energy"] / 100.0,
                    min(e["distance"] / 1000.0, 1.0),  # Normalized distance
                ])
            
            # Combat stats (4 dims)
            combat_state = [
                min(c["bullets_fired"] / 50.0, 1.0),
                min(c["hits_dealt"] / 20.0, 1.0),
                min(c["damage_dealt"] / 100.0, 1.0),
                min(c["damage_taken"] / 100.0, 1.0),
            ]
            
            # Reserved (2 dims)
            reserved = [0.0, 0.0]
            
            vec = np.array(self_state + enemy_state + combat_state + reserved, dtype=np.float32)

        else:
            vec = np.zeros((37,), dtype=np.float32)

        # Return format depends on observation mode
        if self.use_visual_obs:
            return {
                "visual_obs": self.last_img,
                "vector_obs": vec
            }
        else:
            return vec  # Vector-only: return raw numpy array

    def render(self):
        return self.last_img

    def get_health_status(self):
        """Get health status for monitoring rollout stability."""
        return {
            "consecutive_timeouts": self._consecutive_timeouts,
            "needs_force_reset": self._needs_force_reset,
            "last_tick_age_s": time.time() - self._last_tick_time,
            "bot_running": self.bot.is_running() if self.bot else False,
            "opponent_status": self.opponent_manager.get_status() if self.opponent_manager else None,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
        }

    def _save_tick_data(self):
        """Save tick-by-tick game state data to JSON file for debugging."""
        import json
        try:
            os.makedirs("artifacts/tick_data", exist_ok=True)
            filename = f"artifacts/tick_data/episode_{self.episode_count}_ticks.json"
            
            summary = {
                "episode": self.episode_count,
                "total_steps": len(self._tick_data),
                "port": self.port,
                "ticks": self._tick_data
            }
            
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Exported tick data to {filename} ({len(self._tick_data)} ticks)")
        except Exception as e:
            logger.error(f"Failed to save tick data: {e}")
