from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path.home() / "Desktop" / "world_models_test"

# =========================
# CHANGE THESE PER SEED RUN
# =========================
SEED = 1

# Existing seed-1 runs:
DREAMER_RUN_NAME = "dreamer_cartpole_gpu_real2"
TD_RUN_NAME = "cartpole_400k_seed1"

# =========================
# Paths
# =========================
dreamer_run = BASE / "runs" / DREAMER_RUN_NAME
dreamer_scores_path = dreamer_run / "scores.jsonl"
dreamer_metrics_path = dreamer_run / "metrics.jsonl"

td_run = BASE / "tdmpc2" / "tdmpc2" / "logs" / "cartpole-swingup" / "1" / TD_RUN_NAME
td_eval_path = td_run / "eval.csv"

# Save figures into a seed-specific subfolder
fig_dir = BASE / "figures" / f"seed{SEED}"
fig_dir.mkdir(parents=True, exist_ok=True)

for p in [dreamer_scores_path, dreamer_metrics_path, td_eval_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def load_dreamer_scores(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            x = json.loads(line)
            if "step" in x and "episode/score" in x:
                rows.append({
                    "step": float(x["step"]),
                    "score": float(x["episode/score"]),
                })
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def load_dreamer_metrics(path: Path) -> pd.DataFrame:
    rows = []
    keep = [
        "train/loss/rew",
        "train/loss/value",
        "train/loss/policy",
        "train/loss/dyn",
        "train/loss/con",
    ]
    with path.open() as f:
        for line in f:
            x = json.loads(line)
            if "step" not in x:
                continue
            row = {"step": float(x["step"])}
            for k in keep:
                if k in x:
                    row[k] = float(x[k])
            rows.append(row)
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


dreamer_scores = load_dreamer_scores(dreamer_scores_path)
dreamer_metrics = load_dreamer_metrics(dreamer_metrics_path)
td_eval = pd.read_csv(td_eval_path).sort_values("step").reset_index(drop=True)

dreamer_max_step = int(dreamer_scores["step"].max())
td_max_step = int(td_eval["step"].max())
shared_max_step = min(dreamer_max_step, td_max_step)

# -------------------------
# 1. Dreamer score curve
# -------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(dreamer_scores["step"], dreamer_scores["score"], alpha=0.35, label="raw")
plt.plot(
    dreamer_scores["step"],
    moving_average(dreamer_scores["score"], 20),
    label="moving avg (20)"
)
plt.xlabel("Environment steps")
plt.ylabel("Episode score")
plt.title(f"DreamerV3 on cartpole-swingup (seed {SEED})")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / f"dreamer_score_curve_seed{SEED}.png", dpi=160)
plt.close()

# -------------------------
# 2. Dreamer loss curves
# -------------------------
plt.figure(figsize=(8, 4.5))
for col in [
    "train/loss/rew",
    "train/loss/value",
    "train/loss/policy",
    "train/loss/dyn",
    "train/loss/con",
]:
    if col in dreamer_metrics.columns:
        plt.plot(dreamer_metrics["step"], dreamer_metrics[col], label=col)
plt.xlabel("Environment steps")
plt.ylabel("Loss")
plt.title(f"DreamerV3 training losses (seed {SEED})")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / f"dreamer_loss_curves_seed{SEED}.png", dpi=160)
plt.close()

# -------------------------
# 3. TD-MPC2 eval curve
# -------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(td_eval["step"], td_eval["episode_reward"], marker="o")
plt.xlabel("Environment steps")
plt.ylabel("Evaluation reward")
plt.title(f"TD-MPC2 on cartpole-swingup (seed {SEED})")
plt.tight_layout()
plt.savefig(fig_dir / f"tdmpc2_eval_curve_seed{SEED}.png", dpi=160)
plt.close()

# -------------------------
# 4. Shared-range comparison
# -------------------------
dreamer_shared = dreamer_scores[dreamer_scores["step"] <= shared_max_step].copy()
td_shared = td_eval[td_eval["step"] <= shared_max_step].copy()

plt.figure(figsize=(8, 4.5))
plt.plot(
    dreamer_shared["step"],
    moving_average(dreamer_shared["score"], 20),
    label="Dreamer (moving avg 20)"
)
plt.plot(
    td_shared["step"],
    td_shared["episode_reward"],
    marker="o",
    label="TD-MPC2 eval"
)
plt.xlabel("Environment steps")
plt.ylabel("Return / reward")
plt.title(f"Dreamer vs TD-MPC2 (seed {SEED}, shared <= {shared_max_step} steps)")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / f"comparison_shared_range_seed{SEED}.png", dpi=160)
plt.close()

# -------------------------
# 5. Full-range comparison
# -------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(
    dreamer_scores["step"],
    moving_average(dreamer_scores["score"], 20),
    label="Dreamer (moving avg 20)"
)
plt.plot(
    td_eval["step"],
    td_eval["episode_reward"],
    marker="o",
    label="TD-MPC2 eval"
)
plt.xlabel("Environment steps")
plt.ylabel("Return / reward")
plt.title(f"Dreamer vs TD-MPC2 (seed {SEED}, full ranges)")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / f"comparison_full_ranges_seed{SEED}.png", dpi=160)
plt.close()

# -------------------------
# Summary text
# -------------------------
summary_lines = [
    f"Seed: {SEED}",
    f"Dreamer run folder: {DREAMER_RUN_NAME}",
    f"TD-MPC2 run folder: {TD_RUN_NAME}",
    f"Dreamer max step: {dreamer_max_step}",
    f"TD-MPC2 max step: {td_max_step}",
    f"Shared comparison max step: {shared_max_step}",
    f"Dreamer last score: {float(dreamer_scores['score'].iloc[-1]):.4f}",
    f"Dreamer best score: {float(dreamer_scores['score'].max()):.4f}",
    f"TD-MPC2 last eval reward: {float(td_eval['episode_reward'].iloc[-1]):.4f}",
    f"TD-MPC2 best eval reward: {float(td_eval['episode_reward'].max()):.4f}",
    f"Dreamer score points: {len(dreamer_scores)}",
    f"TD-MPC2 eval points: {len(td_eval)}",
]
summary_path = fig_dir / f"summary_seed{SEED}.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n".join(summary_lines))
print(f"\nSaved files in: {fig_dir}")
