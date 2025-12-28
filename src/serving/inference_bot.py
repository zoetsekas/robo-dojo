import asyncio
import numpy as np
import torch
import torch.nn as nn
import logging
import sys
from robocode_tank_royale.bot_api import Bot, BotInfo
from src.env.video_capture import VideoCapture

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Inference] %(message)s')
logger = logging.getLogger(__name__)

class InferenceNet(nn.Module):
    """Standalone version of the trained network without RLlib dependencies."""
    def __init__(self, vector_dim=15, num_actions=5):
        super().__init__()
        # num_actions*2 because RLlib outputs Mean and LogStd for Box spaces
        num_outputs = num_actions * 2
        
        # Visual Branch
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.visual_fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        
        # Vector Branch
        self.vector_fc = nn.Sequential(
            nn.Linear(vector_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        
        # Fusion
        self.fusion_fc = nn.Sequential(nn.Linear(512 + 128, 512), nn.ReLU())
        self.action_head = nn.Linear(512, num_outputs)

    def forward(self, visual_obs, vector_obs):
        # Visual: (B, H, W, C) -> (B, C, H, W) and normalize
        x_vis = visual_obs.float() / 255.0
        x_vis = x_vis.permute(0, 3, 1, 2)
        vis_feat = self.visual_fc(self.cnn(x_vis))
        
        vec_feat = self.vector_fc(vector_obs)
        fused = torch.cat([vis_feat, vec_feat], dim=1)
        context = self.fusion_fc(fused)
        return self.action_head(context)

class InferenceBot(Bot):
    def __init__(self, weights_path, server_url, display=":99"):
        bot_info = BotInfo(
            name="RoboDojoInference",
            version="1.0",
            authors=["Antigravity"],
            description="Trained RL Agent in Serving Mode",
            game_types=["classic", "melee", "1v1"]
        )
        super().__init__(bot_info, server_url)
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InferenceNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
        # Vision
        self.video_capture = VideoCapture(display=display)
        
        logger.info(f"InferenceBot initialized on {self.device}. Using display {display}")

    def _normalize_vec(self, o):
        # Must match RobocodeGymEnv._get_obs normalization
        return np.array([
            o['x'] / 800.0, o['y'] / 600.0, o['speed'] / 8.0, 
            o['energy'] / 100.0, o['heading'] / 360.0,
            o['gun_heading'] / 360.0, o['radar_heading'] / 360.0,
            o['gun_heat'] / 3.0, o['enemy_count'] / 10.0,
            0,0,0,0,0,0
        ][:15], dtype=np.float32)

    async def run(self):
        while self.is_running():
            # Create observations
            vector_obs = self._normalize_vec({
                "x": self.get_x(), "y": self.get_y(), "speed": self.get_speed(),
                "energy": self.get_energy(), "heading": self.get_direction(),
                "gun_heading": self.get_gun_direction(), "radar_heading": self.get_radar_direction(),
                "gun_heat": self.get_gun_heat(), "enemy_count": self.get_enemy_count()
            })
            
            # Grab visual obs
            frame = self.video_capture.grab_frame()
            visual_obs = self.video_capture.process_for_model(frame)
            
            # Inference
            with torch.no_grad():
                v_tensor = torch.from_numpy(visual_obs).unsqueeze(0).to(self.device)
                vec_tensor = torch.from_numpy(vector_obs).unsqueeze(0).to(self.device)
                outputs = self.model(v_tensor, vec_tensor).cpu().numpy()[0]
                
                # First 5 are Means for actions [target_speed, turn, gun, radar, fire]
                actions = outputs[:5]
            
            # Apply actions
            self.target_speed = float(actions[0])
            self.turn_rate = float(actions[1])
            self.gun_turn_rate = float(actions[2])
            self.radar_turn_rate = float(actions[3])
            
            if actions[4] > 0.1 and self.get_gun_heat() == 0:
                fire_power = max(0.1, min(3.0, float(actions[4])))
                await self.fire(fire_power)
            
            await self.go()


if __name__ == "__main__":
    import sys
    weights = sys.argv[1] if len(sys.argv) > 1 else "artifacts/serving/bot_weights.pt"
    url = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:7654"
    disp = sys.argv[3] if len(sys.argv) > 3 else ":99"
    
    bot = InferenceBot(weights, url, disp)
    asyncio.run(bot.start())
