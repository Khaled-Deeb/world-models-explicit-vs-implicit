from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path.home() / "Desktop" / "world_models_test"
FIG_DIR = BASE / "figures" / "cartpole_ci"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Change this only if your seed-1 Dreamer folder name is different.
DREAMER_RUNS = {
    "seed1": BASE / "runs" / "dreamer_cartpole_gpu_real2",
    "seed2": BASE / "runs" / "dreamer_cartpole_gpu_seed2",
    "seed3": BASE / "runs" / "dreamer_cartpole_gpu_seed3",
}

TD_RUNS = {
    "seed1": BASE / "tdmpc2" / "tdmpc2" / "logs" / "cartpole-swingup" / "1" / "cartpole_400k_seed1",
    "seed2": BASE / "tdmpc2" / "tdmpc2" / "logs" / "cartpole-swingup" / "2" / "cartpole_200k_seed2",
    "seed3": BASE / "tdmpc2" / "tdmpc2" / "logs" / "cartpole-swingup" / "3" / "cartpole_150k_seed3",
}

# Shared horizon for fair 3-seed comparison.
GRID = np.arange(5000, 150000 + 1, 5000, dtype=float)


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def load_dreamer_scores(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            x = json.loads(line)
            if "step" in x and "episode/score" in x:
                rows.append(
                    {
                        "step": float(x["step"]),
                        "score": float(x["episode/score"]),
                    }
                )
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No Dreamer scores found in {path}")
    df["score_smooth"] = moving_average(df["score"], 20)
    df = df.drop_duplicates(subset="step", keep="last")
    return df


def load_td_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("step").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No TD-MPC2 eval data found in {path}")
    df = df.drop_duplicates(subset="step", keep="last")
    return df


def interp_on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    mask = grid <= x.max()
    out = np.full_like(grid, np.nan, dtype=float)
    out[mask] = np.interp(grid[mask], x, y)
    return out


def summarize_curves(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0, ddof=1)
    n = np.sum(~np.isnan(curves), axis=0)
    ci = 1.96 * std / np.sqrt(n)
    return mean, std, ci


# Load Dreamer seeds
dreamer_curves = []
for name, run_dir in DREAMER_RUNS.items():
    path = run_dir / "scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing Dreamer scores file for {name}: {path}")
    df = load_dreamer_scores(path)
    curve = interp_on_grid(df["step"].to_numpy(), df["score_smooth"].to_numpy(), GRID)
    dreamer_curves.append(curve)

dreamer_curves = np.vstack(dreamer_curves)
dreamer_mean, dreamer_std, dreamer_ci = summarize_curves(dreamer_curves)

# Load TD-MPC2 seeds
td_curves = []
for name, run_dir in TD_RUNS.items():
    path = run_dir / "eval.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing TD-MPC2 eval file for {name}: {path}")
    df = load_td_eval(path)
    curve = interp_on_grid(df["step"].to_numpy(), df["episode_reward"].to_numpy(), GRID)
    td_curves.append(curve)

td_curves = np.vstack(td_curves)
td_mean, td_std, td_ci = summarize_curves(td_curves)

# 1. Dreamer CI plot
plt.figure(figsize=(8, 4.5))
for i, curve in enumerate(dreamer_curves, start=1):
    plt.plot(GRID, curve, alpha=0.25, label=f"Dreamer seed {i}")
plt.plot(GRID, dreamer_mean, linewidth=2, label="Dreamer mean")
plt.fill_between(
    GRID,
    dreamer_mean - dreamer_ci,
    dreamer_mean + dreamer_ci,
    alpha=0.25,
    label="Dreamer 95% CI",
)
plt.xlabel("Environment steps")
plt.ylabel("Episode score")
plt.title("DreamerV3 cartpole-swingup: mean and 95% CI (3 seeds)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "dreamer_cartpole_mean_ci.png", dpi=160)
plt.close()

# 2. TD-MPC2 CI plot
plt.figure(figsize=(8, 4.5))
for i, curve in enumerate(td_curves, start=1):
    plt.plot(GRID, curve, alpha=0.25, label=f"TD-MPC2 seed {i}")
plt.plot(GRID, td_mean, linewidth=2, label="TD-MPC2 mean")
plt.fill_between(
    GRID,
    td_mean - td_ci,
    td_mean + td_ci,
    alpha=0.25,
    label="TD-MPC2 95% CI",
)
plt.xlabel("Environment steps")
plt.ylabel("Evaluation reward")
plt.title("TD-MPC2 cartpole-swingup: mean and 95% CI (3 seeds, up to 150k)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG_DIR / "tdmpc2_cartpole_mean_ci.png", dpi=160)
plt.close()

# 3. Cross-method fair comparison on shared 150k horizon
plt.figure(figsize=(8, 4.5))
plt.plot(GRID, dreamer_mean, linewidth=2, label="Dreamer mean")
plt.fill_between(
    GRID,
    dreamer_mean - dreamer_ci,
    dreamer_mean + dreamer_ci,
    alpha=0.20,
    label="Dreamer 95% CI",
)
plt.plot(GRID, td_mean, linewidth=2, label="TD-MPC2 mean")
plt.fill_between(
    GRID,
    td_mean - td_ci,
    td_mean + td_ci,
    alpha=0.20,
    label="TD-MPC2 95% CI",
)
plt.xlabel("Environment steps")
plt.ylabel("Return / reward")
plt.title("Cartpole-swingup: Dreamer vs TD-MPC2 (mean ± 95% CI, 3 seeds)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "dreamer_vs_tdmpc2_cartpole_mean_ci.png", dpi=160)
plt.close()

# 4. Save summary
summary = [
    f"Grid max step: {int(GRID.max())}",
    f"Dreamer final mean @150k: {dreamer_mean[-1]:.4f}",
    f"Dreamer final 95% CI half-width @150k: {dreamer_ci[-1]:.4f}",
    f"TD-MPC2 final mean @150k: {td_mean[-1]:.4f}",
    f"TD-MPC2 final 95% CI half-width @150k: {td_ci[-1]:.4f}",
    f"Dreamer best mean on grid: {np.nanmax(dreamer_mean):.4f}",
    f"TD-MPC2 best mean on grid: {np.nanmax(td_mean):.4f}",
]
(FIG_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print(f"\nSaved files in: {FIG_DIR}")
