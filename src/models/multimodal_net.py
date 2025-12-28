import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class MultimodalRoboModel(TorchModelV2, nn.Module):
    """
    Dual-branch neural network for Robocode.
    Branch 1: CNN for visual observation (84x84x3).
    Branch 2: MLP for vector observation (15,).
    Fusion: Concatenation -> Dense -> Action/Value Heads.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if hasattr(obs_space, "original_space"):
            self.visual_shape = obs_space.original_space["visual_obs"].shape
            self.vector_shape = obs_space.original_space["vector_obs"].shape[0]
        else:
            self.visual_shape = obs_space["visual_obs"].shape
            self.vector_shape = obs_space["vector_obs"].shape[0]

        # --- Visual Branch (Nature CNN) ---
        # Input: (B, 84, 84, 3) -> Permute to (B, 3, 84, 84)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        # 84 -> 21 -> 9 -> 7. 64 * 7 * 7 = 3136
        self.cnn_out_size = 3136
        
        self.visual_fc = nn.Sequential(
            nn.Linear(self.cnn_out_size, 512),
            nn.ReLU()
        )

        # --- Vector Branch ---
        self.vector_fc = nn.Sequential(
            nn.Linear(self.vector_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # --- Fusion ---
        # 512 + 128 = 640
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU()
        )
        
        # Action Head
        self.action_head = nn.Linear(512, num_outputs)
        
        # Value Head
        self.value_head = nn.Linear(512, 1)
        
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract observations
        visual_obs = input_dict["obs"]["visual_obs"] # Shape: (B, 84, 84, 3)
        vector_obs = input_dict["obs"]["vector_obs"] # Shape: (B, 15)

        # Preprocess Visual: normalize and permute (B, H, W, C) -> (B, C, H, W)
        x_vis = visual_obs.float() / 255.0
        x_vis = x_vis.permute(0, 3, 1, 2)
        
        # Pass through branches
        vis_feat = self.cnn(x_vis)
        vis_feat = self.visual_fc(vis_feat)
        
        vec_feat = self.vector_fc(vector_obs)
        
        # Concatenate
        fused = torch.cat([vis_feat, vec_feat], dim=1)
        
        # Fusion Layer
        context = self.fusion_fc(fused)
        
        # Heads
        action_logits = self.action_head(context)
        self._value_out = self.value_head(context)
        
        return action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out.reshape(-1)
