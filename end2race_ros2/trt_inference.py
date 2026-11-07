import os
import time
from typing import Tuple
import numpy as np

import torch
from end2race_ros2.model_def import load_end2race_checkpoint


class End2RaceInferenceEngine:
    """
    Stateful inference wrapper for End2Race GRU policy.
    Maintains the (1, 1, 420) GRU hidden state in memory across steps and supports:
      1. 'torch': Native PyTorch CUDA (FP16 or FP32) - <0.45 ms on Jetson Orin Nano
      2. 'onnx': ONNXRuntime with TensorrtExecutionProvider / CUDAExecutionProvider
    """

    def __init__(
        self,
        backend: str = "torch",
        model_path: str = "models/ppo.pt",
        onnx_path: str = "models/end2race_ppo.onnx",
        use_fp16: bool = True,
        gru_hidden_dim: int = 420,
    ):
        self.backend = backend.lower()
        self.gru_hidden_dim = gru_hidden_dim
        self.use_fp16 = use_fp16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.torch_model = None
        self.ort_session = None

        if self.backend == "onnx" and os.path.exists(onnx_path):
            self._init_onnx(onnx_path)
        else:
            if self.backend == "onnx":
                print(f"[End2RaceInference] ONNX file '{onnx_path}' not found; falling back to PyTorch '{model_path}'.")
            self.backend = "torch"
            self._init_torch(model_path)

        self.reset_hidden_state()
        self._warmup()

    def _init_torch(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at '{model_path}'. "
                f"Copy checkpoint/ppo.pt from End2Race repo into end2race_ros2/models/ppo.pt."
            )
        self.torch_model, meta = load_end2race_checkpoint(
            model_path, self.device, use_fp16=self.use_fp16
        )

    def _init_onnx(self, onnx_path: str) -> None:
        import onnxruntime as ort

        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": self.use_fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": os.path.dirname(onnx_path) or ".",
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )

    def reset_hidden_state(self) -> None:
        """Resets GRU hidden state h_0 = 0 (call at episode start, manual override, or E-brake)."""
        if self.backend == "torch":
            dtype = torch.float16 if (self.use_fp16 and self.device.type == "cuda") else torch.float32
            self.hidden_state = torch.zeros(
                (1, 1, self.gru_hidden_dim), dtype=dtype, device=self.device
            )
        else:
            self.hidden_state_np = np.zeros((1, 1, self.gru_hidden_dim), dtype=np.float32)

    def _warmup(self, num_iters: int = 20) -> float:
        """Warms up GPU kernels and returns average inference latency in milliseconds."""
        dummy_lidar = np.full(180, 0.5, dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(num_iters):
            self.infer(dummy_lidar, 2.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        avg_ms = ((time.perf_counter() - t0) * 1000.0) / num_iters
        self.reset_hidden_state()
        return avg_ms

    @torch.inference_mode()
    def infer(self, norm_lidar_180: np.ndarray, scaled_ego_speed: float) -> Tuple[float, float, float]:
        """
        Runs a single forward step and updates internal GRU hidden state h_t.

        Returns:
            raw_steer_rad: predicted steering angle (radians)
            raw_speed_m_s: predicted speed (m/s)
            latency_ms: inference execution time (ms)
        """
        t_start = time.perf_counter()

        if self.backend == "torch":
            dtype = self.hidden_state.dtype
            lidar_t = torch.from_numpy(norm_lidar_180).view(1, 180).to(
                device=self.device, dtype=dtype, non_blocking=True
            )
            speed_t = torch.tensor(
                [[scaled_ego_speed]], dtype=dtype, device=self.device
            )
            actions_t, self.hidden_state = self.torch_model(
                lidar_t, speed_t, self.hidden_state
            )
            actions = actions_t[0].float().cpu().numpy()
        else:
            lidar_in = norm_lidar_180.reshape(1, 180).astype(np.float32)
            speed_in = np.array([[scaled_ego_speed]], dtype=np.float32)
            outputs = self.ort_session.run(
                ["actions", "h_out"],
                {
                    "normalized_lidar": lidar_in,
                    "ego_speed": speed_in,
                    "h_in": self.hidden_state_np,
                },
            )
            actions = outputs[0][0]
            self.hidden_state_np = outputs[1]

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # End2Race action convention: index 0 = steering (rad), index 1 = speed (m/s)
        # Note: if your specific checkpoint outputs [speed, steer], swap indices here.
        # Heuristic check: steering is bounded in [-0.42, 0.42] rad while speed is [0, 9.5] m/s.
        a0, a1 = float(actions[0]), float(actions[1])
        if abs(a0) > 1.2 and abs(a1) <= 0.6:
            raw_speed_m_s, raw_steer_rad = a0, a1
        else:
            raw_steer_rad, raw_speed_m_s = a0, a1

        return raw_steer_rad, raw_speed_m_s, latency_ms