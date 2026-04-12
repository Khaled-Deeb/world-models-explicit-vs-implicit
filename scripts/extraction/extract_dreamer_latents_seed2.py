from pathlib import Path
import numpy as np

REPLAY = Path.home() / "Desktop" / "world_models_test" / "runs" / "dreamer_cartpole_gpu_seed2" / "replay"
OUTDIR = Path.home() / "Desktop" / "world_models_test" / "latent_analysis" / "cartpole_seed2"
OUTDIR.mkdir(parents=True, exist_ok=True)

files = sorted(REPLAY.glob("*.npz"))
if not files:
    raise SystemExit(f"No replay files found in {REPLAY}")

latents = []
rewards = []
cart_positions = []
pole_angles = []
steps = []
episodes = []
positions_all = []
velocities_all = []

for ep_id, f in enumerate(files):
    data = np.load(f, allow_pickle=True)

    required = ["dyn/deter", "dyn/stoch", "reward", "position", "velocity"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"Skipping {f.name}: missing keys {missing}")
        continue

    deter = data["dyn/deter"].astype(np.float32)          # (T, 512)
    stoch = data["dyn/stoch"].astype(np.float32)          # (T, 32, 4)
    reward = data["reward"].astype(np.float32)            # (T,)
    position = data["position"].astype(np.float32)        # (T, 3)
    velocity = data["velocity"].astype(np.float32)        # (T, 2)

    # Dreamer latent = deterministic + flattened stochastic
    stoch_flat = stoch.reshape(stoch.shape[0], -1)        # (T, 128)
    latent = np.concatenate([deter, stoch_flat], axis=-1) # (T, 640)

    # DMC cartpole proprio:
    # position[:, 0] = cart position
    # position[:, 1], position[:, 2] = cos(theta), sin(theta)
    cart_pos = position[:, 0]
    pole_angle = np.arctan2(position[:, 2], position[:, 1]).astype(np.float32)

    T = latent.shape[0]

    latents.append(latent)
    rewards.append(reward)
    cart_positions.append(cart_pos)
    pole_angles.append(pole_angle)
    steps.append(np.arange(T, dtype=np.int32))
    episodes.append(np.full(T, ep_id, dtype=np.int32))
    positions_all.append(position)
    velocities_all.append(velocity)

if not latents:
    raise SystemExit("No valid replay chunks found.")

out = {
    "latent": np.concatenate(latents, axis=0),
    "reward": np.concatenate(rewards, axis=0),
    "cart_position": np.concatenate(cart_positions, axis=0),
    "pole_angle": np.concatenate(pole_angles, axis=0),
    "step": np.concatenate(steps, axis=0),
    "episode": np.concatenate(episodes, axis=0),
    "position": np.concatenate(positions_all, axis=0),
    "velocity": np.concatenate(velocities_all, axis=0),
}

outpath = OUTDIR / "dreamer_seed2_latents_fixed.npz"
np.savez(outpath, **out)

print("Saved:", outpath)
print("latent shape:", out["latent"].shape)
print("reward shape:", out["reward"].shape)
print("cart_position shape:", out["cart_position"].shape)
print("pole_angle shape:", out["pole_angle"].shape)
print("position shape:", out["position"].shape)
print("velocity shape:", out["velocity"].shape)
print("cart_position finite:", np.isfinite(out["cart_position"]).sum())
print("pole_angle finite:", np.isfinite(out["pole_angle"]).sum())
print("reward finite:", np.isfinite(out["reward"]).sum())
