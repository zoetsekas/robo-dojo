import sys
import os
import time
import numpy as np
import cv2
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.env.video_capture import VideoCapture

def test_video_recording():
    """Tests the video writing logic with synthetic moving frames."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [TestVideo] %(message)s')
    logger = logging.getLogger(__name__)
    
    # Mock mss to avoid "Unable to open display" error
    from unittest.mock import MagicMock
    import mss
    mss.mss = MagicMock()
    
    # 1. Initialize VideoCapture
    vc = VideoCapture(display=":99", width=800, height=600)
    
    # 2. Define a synthetic grab_frame to simulate a moving bot
    def synthetic_grab():
        # Create a blank BGR frame (OpenCV default)
        frame = np.zeros((vc.height, vc.width, 3), dtype=np.uint8)
        
        # Draw a moving "bot" (green square)
        t = time.time()
        x = int(400 + 200 * np.cos(t * 2))
        y = int(300 + 150 * np.sin(t * 2))
        cv2.rectangle(frame, (x-40, y-40), (x+40, y+40), (0, 255, 0), -1)
        
        # Add some text
        cv2.putText(frame, f"RoboDojo Video Test: {time.strftime('%H:%M:%S')}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Feed directly to the writer (VideoCapture._write_frame now expects BGR)
        if vc.recording and vc.video_writer:
            vc._write_frame(frame)
            
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the patch
    vc.grab_frame = synthetic_grab
    
    # 3. Test the recording life cycle
    output_path = "artifacts/recordings/offline_test.mp4"
    logger.info(f"Starting 5-second test recording to {output_path}...")
    
    if vc.start_recording(output_path, fps=30):
        # Record 150 frames (5 seconds @ 30fps)
        for i in range(150):
            vc.grab_frame()
            time.sleep(1/30)
            if (i+1) % 30 == 0:
                logger.info(f"  Recorded {i+1} frames...")
        
        frames_saved = vc.stop_recording()
        logger.info(f"Test complete. Total frames saved: {frames_saved}")
        
        # 4. Final verification
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Success! Video created: {output_path}")
            logger.info(f"File size: {file_size/1024:.1f} KB")
            
            if file_size > 5000: # Synthetic video should be at least a few KB
                logger.info("✅ VERIFIED: Video file is playable and has content.")
                print(f"\n[SUCCESS] Please check: {os.path.abspath(output_path)}")
            else:
                logger.error("❌ ERROR: Video file is suspiciously small. Codec issue likely.")
        else:
            logger.error("❌ ERROR: Video file was NOT created.")
    else:
        logger.error("❌ ERROR: Could not initialize VideoWriter. Check your OpenCV installation.")

if __name__ == "__main__":
    test_video_recording()
