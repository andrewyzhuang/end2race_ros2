import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class End2RacePolicyNet(nn.Module):
    """
    End2Race GRU-based Recurrent Racing Policy Network (850,556 parameters).

    Architecture (Section III-D of End2Race paper):
      - Input:
          * 180-dim sigmoid-normalized 2D LiDAR scan: sigma(x) in [0, 1]
          * Scalar ego speed v_t projected via Linear(1, 30) -> 30-dim vector
          * Learnable 30-dim mask embedding e_mask (used during training with p=0.2;
            bypassed during deterministic evaluation)
      - Concatenation: 180 + 30 = 210-dim input vector
      - Backbone: Single-layer GRU(input_size=210, hidden_size=420)
      - Action Head: Two-layer MLP with ReLU activation mapping 420 -> [steer, speed]
    """

    def __init__(
        self,
        lidar_dim: int = 180,
        speed_embed_dim: int = 30,
        gru_hidden_dim: int = 420,
        mlp_hidden_dim: int = 128,
        action_dim: int = 2,
    ):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.speed_embed_dim = speed_embed_dim
        self.gru_hidden_dim = gru_hidden_dim

        # 1. Ego-speed projection: scalar v_t -> 30D vector
        self.speed_encoder = nn.Linear(1, speed_embed_dim)
        # Learnable mask token (kept for checkpoint state_dict compatibility)
        self.mask_embedding = nn.Parameter(torch.zeros(1, speed_embed_dim))

        # 2. Recurrent GRU backbone: 210 -> 420
        self.gru = nn.GRU(
            input_size=lidar_dim + speed_embed_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 3. Two-layer MLP Action Head with ReLU
        self.action_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, action_dim),
        )

    def forward(
        self,
        normalized_lidar: torch.Tensor,
        ego_speed: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step deterministic forward pass for onboard real-time inference.

        Args:
            normalized_lidar: (B, 180) float tensor in [0, 1]
            ego_speed: (B, 1) float tensor in m/s
            hidden_state: (1, B, 420) float tensor

        Returns:
            actions: (B, 2) tensor -> [steering_angle_rad, target_speed_m_s]
            next_hidden_state: (1, B, 420) tensor
        """
        speed_feat = self.speed_encoder(ego_speed)  # (B, 30)
        x = torch.cat([normalized_lidar, speed_feat], dim=-1)  # (B, 210)
        x = x.unsqueeze(1)  # (B, 1, 210) for batch_first GRU

        gru_out, next_hidden_state = self.gru(x, hidden_state)  # gru_out: (B, 1, 420)
        actions = self.action_head(gru_out.squeeze(1))  # (B, 2)
        return actions, next_hidden_state


def load_end2race_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    use_fp16: bool = True,
) -> Tuple[nn.Module, Dict[str, int]]:
    """
    Loads an End2Race checkpoint (`ppo.pt` or `bc.pt`).
    Automatically infers MLP hidden dimension from the checkpoint tensors and maps
    actor/policy weights even if saved inside a PPO Actor-Critic dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # If the checkpoint was saved as a TorchScript / full module, return it directly
    if isinstance(ckpt, nn.Module):
        ckpt.eval()
        if use_fp16 and device.type == "cuda":
            ckpt.half()
        return ckpt, {"mlp_hidden_dim": -1}

    # Extract state_dict from common wrapper keys
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "policy_state_dict", "actor_state_dict", "state_dict", "actor", "policy"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    # Strip DistributedDataParallel 'module.' prefix if present
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Auto-detect MLP hidden dimension from any (H, 420) Linear weight that isn't the GRU or value head
    mlp_hidden_dim = 128
    for k, v in clean_sd.items():
        if v.ndim == 2 and v.shape[1] == 420 and "gru" not in k.lower() and "critic" not in k.lower() and "value" not in k.lower():
            mlp_hidden_dim = int(v.shape[0])
            break

    model = End2RacePolicyNet(
        lidar_dim=180,
        speed_embed_dim=30,
        gru_hidden_dim=420,
        mlp_hidden_dim=mlp_hidden_dim,
        action_dim=2,
    ).to(device)

    # First try strict/direct key match
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)

    # If keys had different attribute names in upstream repo, map by exact tensor shape and role
    if len(missing) > 1:
        mapped_sd = model.state_dict()
        for k, v in clean_sd.items():
            kl = k.lower()
            if "critic" in kl or "value" in kl or "log_std" in kl:
                continue
            if v.shape == (30, 1):
                mapped_sd["speed_encoder.weight"] = v
            elif v.shape == (30,) and ("bias" in kl or "speed" in kl):
                mapped_sd["speed_encoder.bias"] = v
            elif v.shape in [(1, 30), (30,)] and "mask" in kl:
                mapped_sd["mask_embedding"] = v.view(1, 30)
            elif "weight_ih_l0" in kl and v.shape == (1260, 210):
                mapped_sd["gru.weight_ih_l0"] = v
            elif "weight_hh_l0" in kl and v.shape == (1260, 420):
                mapped_sd["gru.weight_hh_l0"] = v
            elif "bias_ih_l0" in kl and v.shape == (1260,):
                mapped_sd["gru.bias_ih_l0"] = v
            elif "bias_hh_l0" in kl and v.shape == (1260,):
                mapped_sd["gru.bias_hh_l0"] = v
            elif v.shape == (mlp_hidden_dim, 420):
                mapped_sd["action_head.0.weight"] = v
            elif v.shape == (mlp_hidden_dim,):
                mapped_sd["action_head.0.bias"] = v
            elif v.shape == (2, mlp_hidden_dim):
                mapped_sd["action_head.2.weight"] = v
            elif v.shape == (2,):
                mapped_sd["action_head.2.bias"] = v
        model.load_state_dict(mapped_sd, strict=True)

    model.eval()
    if use_fp16 and device.type == "cuda":
        model.half()

    return model, {"mlp_hidden_dim": mlp_hidden_dim}