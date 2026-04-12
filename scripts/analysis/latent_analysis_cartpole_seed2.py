from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path.home() / "Desktop" / "world_models_test" / "latent_analysis" / "cartpole_seed2"
OUTDIR = BASE / "figures_fixed"
OUTDIR.mkdir(parents=True, exist_ok=True)

dreamer = np.load(BASE / "dreamer_seed2_latents_fixed.npz", allow_pickle=True)
tdmpc2 = np.load(BASE / "tdmpc2_seed2_latents_50eps.npz", allow_pickle=True)


def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (X - mu) / sigma, mu, sigma


def pca_2d(X):
    if len(X) < 2:
        return None
    Xs, _, _ = standardize(X)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xs, full_matrices=False)
    if vt.shape[0] < 2:
        return None
    return Xs @ vt[:2].T


def train_test_split_np(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_test = max(1, int(len(X) * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_linear_regression(X, y):
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    Xb = np.concatenate([ones, X], axis=1)
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return w


def predict_linear_regression(X, w):
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    Xb = np.concatenate([ones, X], axis=1)
    return Xb @ w


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def run_probe(X, y, model_name, target_name):
    mask = np.isfinite(y)
    X = X[mask].astype(np.float32)
    y = y[mask].astype(np.float32)

    if len(X) < 50:
        return {
            "model": model_name,
            "target": target_name,
            "r2": np.nan,
            "mae": np.nan,
            "n_train": 0,
            "n_test": 0,
        }

    Xtr, Xte, ytr, yte = train_test_split_np(X, y, test_frac=0.2, seed=0)
    Xtr_s, mu, sigma = standardize(Xtr)
    Xte_s = (Xte - mu) / sigma

    w = fit_linear_regression(Xtr_s, ytr)
    pred = predict_linear_regression(Xte_s, w)

    return {
        "model": model_name,
        "target": target_name,
        "r2": r2_score_np(yte, pred),
        "mae": mae_np(yte, pred),
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }


def make_pca_plot(X, color, title, outname, color_label):
    mask = np.isfinite(color)
    X = X[mask].astype(np.float32)
    color = color[mask].astype(np.float32)

    if len(X) < 2:
        print(f"Skipping {outname}: not enough points")
        return

    max_points = 20000
    if len(X) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X = X[idx]
        color = color[idx]

    Z = pca_2d(X)
    if Z is None:
        print(f"Skipping {outname}: PCA failed")
        return

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=color, s=6)
    plt.colorbar(sc, label=color_label)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTDIR / outname, dpi=160)
    plt.close()


dreamer_latent = dreamer["latent"].astype(np.float32)
dreamer_reward = dreamer["reward"].astype(np.float32)
dreamer_cart = dreamer["cart_position"].astype(np.float32)
dreamer_angle = dreamer["pole_angle"].astype(np.float32)

td_latent = tdmpc2["latent"].astype(np.float32)
td_reward = tdmpc2["reward"].astype(np.float32)
td_cart = tdmpc2["cart_position"].astype(np.float32)
td_angle = tdmpc2["angle"].astype(np.float32)

print("Dreamer finite:", np.isfinite(dreamer_reward).sum(), np.isfinite(dreamer_cart).sum(), np.isfinite(dreamer_angle).sum())
print("TD-MPC2 finite:", np.isfinite(td_reward).sum(), np.isfinite(td_cart).sum(), np.isfinite(td_angle).sum())

make_pca_plot(dreamer_latent, dreamer_reward, "Dreamer latent PCA (reward)", "dreamer_pca_reward.png", "reward")
make_pca_plot(dreamer_latent, dreamer_angle, "Dreamer latent PCA (pole angle)", "dreamer_pca_angle.png", "pole angle")
make_pca_plot(td_latent, td_reward, "TD-MPC2 latent PCA (reward)", "tdmpc2_pca_reward.png", "reward")
make_pca_plot(td_latent, td_angle, "TD-MPC2 latent PCA (pole angle)", "tdmpc2_pca_angle.png", "pole angle")

results = []
results.append(run_probe(dreamer_latent, dreamer_reward, "Dreamer", "reward"))
results.append(run_probe(dreamer_latent, dreamer_angle, "Dreamer", "pole_angle"))
results.append(run_probe(dreamer_latent, dreamer_cart, "Dreamer", "cart_position"))
results.append(run_probe(td_latent, td_reward, "TD-MPC2", "reward"))
results.append(run_probe(td_latent, td_angle, "TD-MPC2", "pole_angle"))
results.append(run_probe(td_latent, td_cart, "TD-MPC2", "cart_position"))

summary_lines = []
for r in results:
    summary_lines.append(
        f"{r['model']} | target={r['target']} | R2={r['r2']:.4f} | MAE={r['mae']:.4f} | "
        f"n_train={r['n_train']} | n_test={r['n_test']}"
    )

summary_path = OUTDIR / "summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n".join(summary_lines))
print("Saved figures to:", OUTDIR)
