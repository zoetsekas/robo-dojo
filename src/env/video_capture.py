import os
import time
import numpy as np
import cv2
import subprocess
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)


def ensure_window_manager(display=":99"):
    """
    Checks if a window manager is running on the display.
    If not, attempts to start fluxbox.
    """
    try:
        env = os.environ.copy()
        env["DISPLAY"] = display
        result = subprocess.run(["wmctrl", "-m"], env=env, capture_output=True, text=True)
        if result.returncode != 0:
            logger.info(f"No Window Manager detected on {display}. Starting fluxbox...")
            subprocess.Popen(["fluxbox"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            return True
        return True
    except FileNotFoundError:
        logger.warning("wmctrl not found. Cannot verify Window Manager status.")
        return False


class VideoCapture:
    """
    Robust Video Capture for Robocode Tank Royale in Xvfb.
    
    Uses ImageMagick 'import' for frame capture (proven to work).
    Records frames to temp directory, then stitches with FFmpeg at episode end.
    """
    
    def __init__(self, display=":99", width=1024, height=768):
        self.display = display
        self.width = width
        self.height = height
        
        # Recording state
        self.recording = False
        self.recording_path = None
        self.frames_dir = None
        self.frames_recorded = 0
        self._recording_start_time = None
        self._last_frame_time = 0
        self._target_frame_interval = 1.0 / 15  # 15 FPS for recording (less disk I/O)
        
        # For RL model frame cache
        self._last_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_cache_time = 0
        self._frame_cache_ttl = 0.05  # 20fps cache for RL
        
        logger.info(f"VideoCapture initialized: display={display}, size={width}x{height}")

    def grab_frame(self):
        """
        Grabs a frame from the virtual display.
        Uses xdotool to force focus, then captures with xwd+convert.
        Returns BGR frame (OpenCV format).
        """
        current_time = time.time()
        
        # Return cached frame if recent enough (for RL model)
        if current_time - self._frame_cache_time < self._frame_cache_ttl:
            return self._last_frame.copy()
        
        try:
            env = os.environ.copy()
            env["DISPLAY"] = self.display
            
            # Force focus on Robocode window using xdotool
            window_id = self._get_robocode_window_id(env)
            if window_id:
                # Raise and focus the window to ensure it's visible
                subprocess.run(
                    ["xdotool", "windowactivate", "--sync", window_id],
                    env=env, capture_output=True, timeout=1.0
                )
                subprocess.run(
                    ["xdotool", "windowraise", window_id],
                    env=env, capture_output=True, timeout=1.0
                )
                
                # Capture specific window with xwd (better than import for some cases)
                xwd_proc = subprocess.run(
                    ["xwd", "-id", window_id, "-silent"],
                    env=env, capture_output=True, timeout=2.0
                )
                if xwd_proc.returncode == 0 and xwd_proc.stdout:
                    # Convert xwd to ppm
                    convert_proc = subprocess.run(
                        ["convert", "xwd:-", "ppm:-"],
                        input=xwd_proc.stdout,
                        capture_output=True, timeout=2.0
                    )
                    if convert_proc.returncode == 0 and convert_proc.stdout:
                        result_stdout = convert_proc.stdout
                    else:
                        result_stdout = None
                else:
                    result_stdout = None
            else:
                # Fallback to root window with import
                result = subprocess.run(
                    ["import", "-window", "root", "-depth", "8", "ppm:-"],
                    env=env,
                    capture_output=True,
                    timeout=2.0
                )
                result_stdout = result.stdout if result.returncode == 0 else None
            
            if result_stdout:
                frame = self._parse_ppm(result_stdout)
                if frame is not None:
                    # Resize if needed
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    self._last_frame = frame
                    self._frame_cache_time = current_time
                    
                    # Save frame if recording (with rate limiting)
                    if self.recording and self.frames_dir:
                        self._save_frame_if_needed(frame, current_time)
                    
                    return frame
            
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            logger.debug(f"Frame capture error: {e}")
        
        return self._last_frame.copy()

    def _get_robocode_window_id(self, env):
        """Find Robocode window ID using wmctrl."""
        try:
            result = subprocess.run(
                ["wmctrl", "-l"],
                env=env,
                capture_output=True,
                text=True,
                timeout=1.0
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if 'Robocode' in line or 'robocode' in line.lower():
                        # Format: "0x... workspace host title"
                        window_id = line.split()[0]
                        return window_id
        except:
            pass
        return None

    def _parse_ppm(self, ppm_data):
        """Parse PPM image data to numpy array (BGR)."""
        try:
            # PPM format: P6\n<width> <height>\n<maxval>\n<binary data>
            header_end = 0
            newline_count = 0
            for i, byte in enumerate(ppm_data):
                if byte == ord('\n'):
                    newline_count += 1
                    if newline_count == 3:
                        header_end = i + 1
                        break
            
            header = ppm_data[:header_end].decode('ascii')
            lines = [l for l in header.strip().split('\n') if not l.startswith('#')]
            
            if lines[0] != 'P6':
                return None
                
            dims = lines[1].split()
            width, height = int(dims[0]), int(dims[1])
            
            pixel_data = ppm_data[header_end:]
            img = np.frombuffer(pixel_data, dtype=np.uint8)
            img = img[:width * height * 3]
            img = img.reshape((height, width, 3))
            
            # Convert RGB to BGR for OpenCV
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        except Exception:
            return None

    def _save_frame_if_needed(self, frame, current_time):
        """Save frame to temp directory with rate limiting."""
        if current_time - self._last_frame_time < self._target_frame_interval:
            return
        
        try:
            frame_path = os.path.join(self.frames_dir, f"frame_{self.frames_recorded:06d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.frames_recorded += 1
            self._last_frame_time = current_time
        except Exception as e:
            logger.debug(f"Frame save error: {e}")

    def process_for_model(self, frame, target_dim=(84, 84)):
        """Resizes and converts frame for the RL model (returns RGB)."""
        if frame is None or frame.size == 0:
            return np.zeros((*target_dim, 3), dtype=np.uint8)
        resized = cv2.resize(frame, target_dim, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def start_recording(self, output_path, fps=15):
        """Start frame-by-frame recording to temp directory."""
        if self.recording:
            self.stop_recording()
        
        # Create temp directory for frames
        self.frames_dir = tempfile.mkdtemp(prefix="robo_frames_")
        
        # Prepare output path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        base_path = output_path.rsplit('.', 1)[0]
        self.recording_path = f"{base_path}.mp4"
        
        self.recording = True
        self.frames_recorded = 0
        self._recording_start_time = time.time()
        self._last_frame_time = 0
        self._target_frame_interval = 1.0 / fps
        
        logger.info(f"Started frame recording to {self.frames_dir}")
        return True

    def stop_recording(self):
        """Stop recording and stitch frames into video with FFmpeg."""
        frames = self.frames_recorded
        
        if not self.recording or not self.frames_dir:
            return frames
        
        self.recording = False
        
        try:
            if frames < 5:
                logger.warning(f"Too few frames ({frames}), skipping video creation")
                return frames
            
            # Stitch frames with FFmpeg
            frame_pattern = os.path.join(self.frames_dir, "frame_%06d.jpg")
            
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", "15",
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                "-preset", "fast",
                self.recording_path
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0 and os.path.exists(self.recording_path):
                file_size = os.path.getsize(self.recording_path)
                logger.info(f"Recording saved: {self.recording_path} ({frames} frames, {file_size/1024:.1f}KB)")
            else:
                logger.error(f"FFmpeg stitch failed: {result.stderr.decode()[:200]}")
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg stitch timed out")
        except Exception as e:
            logger.error(f"Video creation error: {e}")
        finally:
            # Cleanup temp frames
            try:
                shutil.rmtree(self.frames_dir)
            except:
                pass
            self.frames_dir = None
        
        return frames

    def save_screenshot(self, path):
        """Save a single screenshot using ImageMagick."""
        try:
            output_dir = os.path.dirname(path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            env = os.environ.copy()
            env["DISPLAY"] = self.display
            
            result = subprocess.run(
                ["import", "-window", "root", path],
                env=env,
                capture_output=True,
                timeout=5.0
            )
            
            if result.returncode == 0:
                logger.info(f"Screenshot saved to {path}")
                return True
            return False
                
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return False


class EpisodeRecorder:
    """Records episodes for training visualization."""
    
    def __init__(self, output_dir="/app/artifacts/recordings", record_every_n=100):
        self.output_dir = output_dir
        self.record_every_n = record_every_n
        self.episode_count = 0
        self.current_recording = None
        os.makedirs(output_dir, exist_ok=True)
    
    def on_episode_start(self, video_capture):
        """Called at episode start."""
        self.episode_count += 1
        if self.episode_count % self.record_every_n == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.output_dir}/episode_{self.episode_count}_{timestamp}.mp4"
            video_capture.start_recording(output_path)
            self.current_recording = output_path
            logger.info(f"Recording episode {self.episode_count}")
    
    def on_episode_end(self, video_capture):
        """Called at episode end."""
        if self.current_recording:
            frames = video_capture.stop_recording()
            logger.info(f"Episode {self.episode_count} recording done ({frames} frames)")
            self.current_recording = None
    
    def get_stats(self):
        return {
            "episodes_recorded": self.episode_count // self.record_every_n,
            "total_episodes": self.episode_count
        }
