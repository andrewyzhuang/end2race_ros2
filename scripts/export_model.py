#!/usr/bin/env python3
"""
Export the recurrent End2Race GRU policy checkpoint (ppo.pt) to a
standalone TorchScript (.pt) module in FP32 or FP16 for Jetson deployment.
"""
import argparse
import os
import torch
import torch.nn as nn


class End2RaceGRUActor(nn.Module):
    """
    Exact match for imitation/model.py:
      - lidar_normalization_k: scalar buffer ()
      - speed_mlp: Sequential(Linear(1, 30), ReLU())
      - dummy_embedding: Parameter (1, 30)
      - gru: GRU(input_size=210, hidden_size=420, batch_first=True)
      - output_layer: Sequential(Linear(420, 128), ReLU(), Linear(128, 2))
    """
    NUM_LIDAR_FEATURES = 180
    SPEED_EMBEDDING_DIM = 30
    GRU_HIDDEN_SIZE = 420
    MLP_HIDDEN_SIZE = 128

    def __init__(self, use_dummy_speed: bool = False):
        super().__init__()
        self.use_dummy_speed = use_dummy_speed

        self.register_buffer("lidar_normalization_k", torch.tensor(1.0, dtype=torch.float32))

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, self.SPEED_EMBEDDING_DIM),
            nn.ReLU(),
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, self.SPEED_EMBEDDING_DIM))

        processed_features = self.NUM_LIDAR_FEATURES + self.SPEED_EMBEDDING_DIM

        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=self.GRU_HIDDEN_SIZE,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.GRU_HIDDEN_SIZE, self.MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.MLP_HIDDEN_SIZE, 2),
        )

    def forward(
        self,
        lidar: torch.Tensor,
        speed: torch.Tensor,
        h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step recurrent forward pass for real-time ROS 2 control loop.
        Args:
            lidar: (B, 180) raw clipped LiDAR ranges in meters
            speed: (B, 1) current ego longitudinal speed in m/s
            h_in:  (1, B, 420) previous GRU hidden state
        Returns:
            action: (B, 2) raw predicted [steering, speed] command
            h_out:  (1, B, 420) updated GRU hidden state
        """
        processed_lidar = 2.0 * torch.sigmoid(-self.lidar_normalization_k * lidar)

        if self.use_dummy_speed:
            speed_embedding = self.dummy_embedding.expand(lidar.size(0), -1)
        else:
            speed_embedding = self.speed_mlp(speed)

        gru_in = torch.cat([processed_lidar, speed_embedding], dim=-1).unsqueeze(1)
        gru_out, h_out = self.gru(gru_in, h_in)
        action = self.output_layer(gru_out.squeeze(1))
        return action, h_out


def export_torchscript(
    checkpoint_path: str,
    output_path: str,
    use_fp16: bool,
    use_dummy_speed: bool
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[1/4] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    model = End2RaceGRUActor(use_dummy_speed=use_dummy_speed)

    # Strict load verifies 100% key and shape parity with ppo.pt
    model.load_state_dict(state_dict, strict=True)
    print(f"      Loaded state_dict strictly! lidar_normalization_k = {model.lidar_normalization_k.item():.4f}")

    device = torch.device("cuda" if (use_fp16 and torch.cuda.is_available()) else "cpu")
    model = model.to(device).eval()

    dummy_lidar = torch.full((1, 180), 5.0, dtype=torch.float32, device=device)
    dummy_speed = torch.full((1, 1), 2.0, dtype=torch.float32, device=device)
    dummy_h = torch.zeros((1, 1, 420), dtype=torch.float32, device=device)

    print("[2/4] Tracing recurrent model (lidar, speed, h_in) -> (action, h_out)...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (dummy_lidar, dummy_speed, dummy_h))

        if use_fp16:
            print("[3/4] Converting traced TorchScript model to FP16 (.half())...")
            traced_model = traced_model.half()
        else:
            print("[3/4] Keeping traced TorchScript model in FP32...")

        traced_model = torch.jit.freeze(traced_model.eval())

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    traced_model.save(output_path)
    print(f"[4/4] Saved TorchScript model to: {output_path}")

    # Test forward pass in FP32 on CPU (or FP16 on CUDA)
    if not (use_fp16 and device.type == "cpu"):
        loaded_jit = torch.jit.load(output_path, map_location=device)
        dtype = torch.float16 if use_fp16 else torch.float32
        with torch.no_grad():
            act, h_next = loaded_jit(
                dummy_lidar.to(dtype),
                dummy_speed.to(dtype),
                dummy_h.to(dtype)
            )
            print(f"      Verification passed! Sample action: {act.cpu().numpy()}, h_out: {tuple(h_next.shape)}")
    else:
        print("      Exported FP16 weights on CPU. (Will execute in FP16 on Jetson CUDA).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export End2Race GRU ppo.pt to TorchScript.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ppo.pt")
    parser.add_argument(
        "--output",
        type=str,
        default="models/end2race_policy_jit_fp16.pt",
        help="Output TorchScript .pt path"
    )
    parser.add_argument("--fp16", action="store_true", help="Export weights in FP16 (.half())")
    parser.add_argument(
        "--use-dummy-speed",
        action="store_true",
        help="Mask ego speed with learned dummy_embedding"
    )
    args = parser.parse_args()

    export_torchscript(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        use_fp16=args.fp16,
        use_dummy_speed=args.use_dummy_speed
    )