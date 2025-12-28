import asyncio
import numpy as np
import torch
import torch.nn as nn
import logging
import sys
import argparse
from robocode_tank_royale.bot_api import Bot, BotInfo
from robocode_tank_royale.bot_api.events import ScannedBotEvent, BulletHitBotEvent, HitByBulletEvent, BulletFiredEvent
from src.env.video_capture import VideoCapture

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Inference] %(message)s')
logger = logging.getLogger(__name__)

class MultimodalInferenceNet(nn.Module):
    """Standalone version of the Multimodal network."""
    def __init__(self, vector_dim=37, num_actions=5):
        super().__init__()
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
        x_vis = visual_obs.float() / 255.0
        x_vis = x_vis.permute(0, 3, 1, 2)
        vis_feat = self.visual_fc(self.cnn(x_vis))
        vec_feat = self.vector_fc(vector_obs)
        fused = torch.cat([vis_feat, vec_feat], dim=1)
        context = self.fusion_fc(fused)
        return self.action_head(context)

class VectorOnlyInferenceNet(nn.Module):
    """Standalone version of the Vector-Only network."""
    def __init__(self, vector_dim=37, num_actions=5):
        super().__init__()
        num_outputs = num_actions * 2
        self.fc = nn.Sequential(
            nn.Linear(vector_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.action_head = nn.Linear(128, num_outputs)

    def forward(self, vector_obs):
        feat = self.fc(vector_obs)
        return self.action_head(feat)

class InferenceBot(Bot):
    def __init__(self, weights_path, server_url, display=":99", vector_only=False):
        bot_info = BotInfo(
            name="RoboDojoInference",
            version="2.0",
            authors=["Antigravity"],
            description="Trained RL Agent with Multi-Enemy Tracking",
            game_types=["classic", "melee", "1v1"]
        )
        super().__init__(bot_info, server_url)
        
        self.vector_only = vector_only
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model Selection
        if vector_only:
            self.model = VectorOnlyInferenceNet(vector_dim=37).to(self.device)
            self.video_capture = None
        else:
            self.model = MultimodalInferenceNet(vector_dim=37).to(self.device)
            self.video_capture = VideoCapture(display=display)
        
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
        # tracking state
        self.max_tracked_enemies = 3
        self.tracked_enemies = [
            {"id": -1, "x": 0.0, "y": 0.0, "speed": 0.0, "heading": 0.0, "energy": 0.0, "distance": 9999.0}
            for _ in range(self.max_tracked_enemies)
        ]
        self.combat_stats = {"bullets_fired": 0, "hits_dealt": 0, "damage_dealt": 0.0, "damage_taken": 0.0}
        
        logger.info(f"InferenceBot (Vector-Only: {vector_only}) initialised on {self.device}")

    # --- Event Handlers for State Tracking ---
    async def on_scanned_bot(self, event: ScannedBotEvent):
        my_x, my_y = self.get_x(), self.get_y()
        dist = ((event.x - my_x)**2 + (event.y - my_y)**2)**0.5
        new_enemy = {
            "id": event.scanned_bot_id, "x": event.x, "y": event.y,
            "speed": event.speed, "heading": event.direction, "energy": event.energy,
            "distance": dist
        }
        # Update/Replace logic
        updated = False
        for i, e in enumerate(self.tracked_enemies):
            if e['id'] == event.scanned_bot_id:
                self.tracked_enemies[i] = new_enemy
                updated = True
                break
        if not updated:
            self.tracked_enemies.append(new_enemy)
        self.tracked_enemies.sort(key=lambda e: e["distance"])
        self.tracked_enemies = self.tracked_enemies[:self.max_tracked_enemies]

    async def on_bullet_fired(self, event: BulletFiredEvent):
        self.combat_stats["bullets_fired"] += 1
    async def on_bullet_hit_bot(self, event: BulletHitBotEvent):
        self.combat_stats["hits_dealt"] += 1
        self.combat_stats["damage_dealt"] += event.damage
    async def on_hit_by_bullet(self, event: HitByBulletEvent):
        self.combat_stats["damage_taken"] += event.damage

    def _get_vector_obs(self):
        # 1. Self State (13 dims)
        o_self = [
            self.get_x() / 800.0, self.get_y() / 600.0, self.get_speed() / 8.0, 
            self.get_energy() / 100.0, self.get_direction() / 360.0,
            self.get_gun_direction() / 360.0, self.get_radar_direction() / 360.0,
            self.get_gun_heat() / 3.0, self.get_enemy_count() / 10.0,
            0.0, 0.0, 0.0, 0.0 # padding for turn rates
        ]
        # 2. Enemy state (18 dims)
        o_enemies = []
        for i in range(self.max_tracked_enemies):
            e = self.tracked_enemies[i] if i < len(self.tracked_enemies) else {
                "x": 0.0, "y": 0.0, "speed": 0.0, "heading": 0.0, "energy": 0.0, "distance": 9999.0
            }
            o_enemies.extend([
                e["x"] / 800.0, e["y"] / 600.0, e["speed"] / 8.0,
                e["heading"] / 360.0, e["energy"] / 100.0, min(e["distance"] / 1000.0, 1.0)
            ])
        # 3. Combat bits (4 dims)
        c = self.combat_stats
        o_combat = [
            min(c["bullets_fired"] / 50.0, 1.0), min(c["hits_dealt"] / 20.0, 1.0),
            min(c["damage_dealt"] / 100.0, 1.0), min(c["damage_taken"] / 100.0, 1.0)
        ]
        return np.array(o_self + o_enemies + o_combat + [0.0, 0.0], dtype=np.float32)

    async def run(self):
        while self.is_running():
            vector_obs = self._get_vector_obs()
            vec_tensor = torch.from_numpy(vector_obs).unsqueeze(0).to(self.device).float()
            
            with torch.no_grad():
                if self.vector_only:
                    outputs = self.model(vec_tensor).cpu().numpy()[0]
                else:
                    frame = self.video_capture.grab_frame()
                    visual_obs = self.video_capture.process_for_model(frame)
                    v_tensor = torch.from_numpy(visual_obs).unsqueeze(0).to(self.device).float()
                    outputs = self.model(v_tensor, vec_tensor).cpu().numpy()[0]
                
                actions = outputs[:5]
            
            self.target_speed = float(actions[0])
            self.turn_rate = float(actions[1])
            self.gun_turn_rate = float(actions[2])
            self.radar_turn_rate = float(actions[3])
            
            if actions[4] > 0.1 and self.get_gun_heat() == 0:
                await self.fire(max(0.1, min(3.0, float(actions[4]))))
            
            await self.go()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=str, default="artifacts/serving/bot_weights.pt")
    parser.add_argument("url", type=str, default="ws://127.0.0.1:7654")
    parser.add_argument("display", type=str, default=":99")
    parser.add_argument("--vector-only", action="store_true")
    args = parser.parse_args()
    
    bot = InferenceBot(args.weights, args.url, args.display, vector_only=args.vector_only)
    asyncio.run(bot.start())
