from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Headless MuJoCo / DMControl inside Docker.
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch

BASE = Path(os.environ.get("WORLD_MODELS_TEST", str(Path.home() / "Desktop" / "world_models_test")))
REPO = BASE / "tdmpc2"
TD_CODE = REPO / "tdmpc2"
OUTDIR = BASE / "latent_analysis" / "cartpole_seed2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Make repo importable.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TD_CODE))

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


@hydra.main(config_path="tdmpc2/tdmpc2", config_name="config")
def main(cfg):
    cfg.task = "cartpole-swingup"
    cfg.model_size = 5
    cfg.seed = 2
    cfg.eval_episodes = 50
    cfg.save_video = False
    cfg.enable_wandb = False
    cfg.compile = False
    cfg.checkpoint = str(
        TD_CODE / "logs" / "cartpole-swingup" / "2" / "cartpole_200k_seed2" / "models" / "final.pt"
    )

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    print("Task:", cfg.task)
    print("Model size:", cfg.model_size)
    print("Checkpoint:", cfg.checkpoint)

    env = make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)
    agent.model.eval()

    device = next(agent.model.parameters()).device
    print("Model device:", device)

    latents = []
    rewards = []
    cart_positions = []
    pole_angles = []
    steps = []
    episodes = []
    observations = []

    num_eps = 50
    for ep in range(num_eps):
        obs = env.reset(task_idx=None)
        done = False
        t = 0

        while not done:
            obs_dev = obs.to(device)
            obs_t = obs_dev.unsqueeze(0)

            with torch.no_grad():
                z = agent.model.encode(obs_t, None)

            obs_np = obs.detach().cpu().numpy()

            latents.append(z.squeeze(0).detach().cpu().numpy())
            observations.append(obs_np)
            steps.append(t)
            episodes.append(ep)

            if obs_np.shape[0] >= 3:
                cart_positions.append(float(obs_np[0]))
                pole_angles.append(float(np.arctan2(obs_np[2], obs_np[1])))
            else:
                cart_positions.append(np.nan)
                pole_angles.append(np.nan)

            action = agent.act(obs_dev, t0=(t == 0), eval_mode=True, task=None)
            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))

            t += 1

    data = {
        "latent": np.stack(latents).astype(np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "cart_position": np.array(cart_positions, dtype=np.float32),
        "angle": np.array(pole_angles, dtype=np.float32),
        "step": np.array(steps, dtype=np.int32),
        "episode": np.array(episodes, dtype=np.int32),
        "obs": np.stack(observations).astype(np.float32),
    }

    outpath = OUTDIR / "tdmpc2_seed2_latents_50eps.npz"
    np.savez(outpath, **data)

    print("Saved:", outpath)
    print("latent shape:", data["latent"].shape)
    print("reward shape:", data["reward"].shape)
    print("obs shape:", data["obs"].shape)
    print("cart_position range:", float(np.nanmin(data["cart_position"])), float(np.nanmax(data["cart_position"])))
    print("angle range:", float(np.nanmin(data["angle"])), float(np.nanmax(data["angle"])))


if __name__ == "__main__":
    main()
