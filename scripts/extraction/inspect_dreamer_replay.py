from pathlib import Path
import numpy as np

replay_dir = Path.home() / "Desktop" / "world_models_test" / "runs" / "dreamer_cartpole_gpu_seed2" / "replay"

files = sorted(list(replay_dir.glob("**/*.npz")) + list(replay_dir.glob("**/*.npy")))
print("Replay dir:", replay_dir)
print("Found files:", len(files))
for f in files[:10]:
    print(" -", f)

if not files:
    raise SystemExit("No replay files found.")

f = files[0]
print("\nOpening:", f)

if f.suffix == ".npz":
    data = np.load(f, allow_pickle=True)
    print("Keys:", list(data.keys()))
    for k in data.keys():
        arr = data[k]
        try:
            print(k, getattr(arr, "shape", None), getattr(arr, "dtype", None))
        except Exception:
            print(k, type(arr))
else:
    arr = np.load(f, allow_pickle=True)
    print("Loaded object type:", type(arr))
    if hasattr(arr, "shape"):
        print("Shape:", arr.shape, "dtype:", arr.dtype)
