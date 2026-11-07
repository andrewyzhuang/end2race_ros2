#!/usr/bin/env python3
"""
Export a trained PyTorch racing policy checkpoint to ONNX format
and verify numerical equivalence with ONNX Runtime.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn


class End2RaceActor(nn.Module):
    """
    Reference deterministic Actor architecture taking stacked (LiDAR + Ego)
    observations and outputting bounded [-1, 1] steering and speed commands.
    """
    def __init__(self, obs_dim: int = 452, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def export_and_verify(
    checkpoint_path: str,
    output_onnx_path: str,
    num_beams: int,
    history_len: int
):
    single_obs_dim = num_beams + 5
    total_obs_dim = single_obs_dim * history_len

    model = End2RaceActor(obs_dim=total_obs_dim, action_dim=2)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = ckpt.get('actor_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No checkpoint provided or found; exporting initialized reference weights.")

    model.eval()
    dummy_input = torch.randn(1, total_obs_dim, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['obs'],
        output_names=['action'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    print(f"Exported ONNX policy to: {output_onnx_path}")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(output_onnx_path, providers=['CPUExecutionProvider'])
        with torch.no_grad():
            torch_out = model(dummy_input).numpy()
        ort_out = ort_sess.run(None, {'obs': dummy_input.numpy()})[0]
        max_diff = np.max(np.abs(torch_out - ort_out))
        print(f"ONNX Runtime verification passed! Max absolute error: {max_diff:.2e}")
        assert max_diff < 1e-5, f"Verification failed: max_diff={max_diff}"
    except ImportError:
        print("onnxruntime not installed; skipping numerical verification.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export End2Race PyTorch policy to ONNX.')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to .pt/.pth checkpoint')
    parser.add_argument(
        '--output',
        type=str,
        default='models/end2race_policy.onnx',
        help='Output ONNX file path'
    )
    parser.add_argument('--num-beams', type=int, default=108, help='Downsampled LiDAR beam count')
    parser.add_argument('--history-len', type=int, default=4, help='Observation history length')
    args = parser.parse_args()

    export_and_verify(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        num_beams=args.num_beams,
        history_len=args.history_len
    )