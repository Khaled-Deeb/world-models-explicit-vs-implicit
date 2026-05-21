# A Lightweight Empirical Comparison of Dreamer-Style and TD-MPC2-Style World Models for Continuous Control

This repository contains the code, analysis scripts, selected checkpoints, curated results, and conference paper draft for a lightweight comparison between two world-model paradigms for decision making:

- **Explicit world model**: a Dreamer-style agent based on an RSSM with imagined latent rollouts
- **Implicit world model**: a TD-MPC2-style agent based on latent consistency, value estimation, and latent-space planning

The study focuses on **state-based continuous control** in DMControl, with:

- **Main matched benchmark**: `cartpole-swingup`
- **Additional extension**: `walker-walk`

---

## Repository structure

```text
.
├── README.md
├── requirements-analysis.txt
├── requirements-dreamer.txt
├── requirements-tdmpc2.txt
├── paper/
├── scripts/
│   ├── train/
│   ├── plotting/
│   ├── extraction/
│   └── analysis/
├── results/
│   ├── final_figures/
│   ├── summaries/
│   └── checkpoints/
└── third_party/
    ├── dreamerv3/
    └── tdmpc2/
```

- `paper/` — conference paper draft
- `scripts/train/` — training wrapper scripts
- `scripts/plotting/` — scripts used to generate learning-curve figures
- `scripts/extraction/` — scripts used to extract latent states from trained models
- `scripts/analysis/` — scripts used for latent-space analysis and probe evaluation
- `results/final_figures/` — final figures used in the report
- `results/summaries/` — numerical summaries used in the report
- `results/checkpoints/` — selected representative checkpoints
- `third_party/` — external upstream codebases used by this project

---

## External code

This project adapts existing open-source implementations rather than re-implementing both agents from scratch.

Clone the external repositories into `third_party/`:

```bash
mkdir -p third_party
git clone https://github.com/danijar/dreamerv3.git third_party/dreamerv3
git clone https://github.com/nicklashansen/tdmpc2.git third_party/tdmpc2
```

---

## Python versions and environments

### Dreamer environment
- Python version used: **3.12.3**
- Local virtual environment expected at:
  - `.venv_dreamer_gpu`

Install Dreamer-related Python dependencies with:

```bash
pip install -r requirements-dreamer.txt
```

### TD-MPC2 environment
- Python version used inside Docker: **3.9.23**
- Reference runtime: **Docker image**
  - `tdmpc2:1.0.1`

A compact reference dependency file is provided:

```text
requirements-tdmpc2.txt
```

However, the **Docker image should be treated as the authoritative runtime** for TD-MPC2 reproduction.

### Analysis environment

```bash
pip install -r requirements-analysis.txt
```

---

## Tasks, seeds, and budgets

### Main matched benchmark: cartpole-swingup
- Dreamer: 3 seeds
- TD-MPC2: 3 seeds
- Intended budget: 200k environment steps per seed
- Final TD-MPC2 seed lengths:
  - 400k
  - 200k
  - 150k
- Reported aggregate comparison:
  - **shared 150k horizon**

### Additional extension: walker-walk
- Dreamer: 3 seeds
- TD-MPC2: 3 seeds
- Nominal budget: 100k environment steps per seed
- Final Dreamer logs reached:
  - 96,096 environment steps
- Reported aggregate comparison:
  - **shared 95k horizon**

### TD-MPC2 model sizes
- Cartpole:
  - `model_size=5`
- Walker:
  - `model_size=1`

This matches the report design, where `cartpole-swingup` is the strongest matched comparison and `walker-walk` is a lower-budget extension.

---

## Training

The training wrappers are placed in:

```text
scripts/train/
```

They assume the external repos were cloned into `third_party/`.

### Dreamer: cartpole

```bash
./scripts/train/run_dreamer_cartpole.sh 1 200000
```

Arguments:
- `$1` = seed
- `$2` = steps
- `$3` = optional run name

Example:

```bash
./scripts/train/run_dreamer_cartpole.sh 2 200000 dreamer_cartpole_seed2_200k
```

### Dreamer: walker

```bash
./scripts/train/run_dreamer_walker.sh 1 100000
```

Example:

```bash
./scripts/train/run_dreamer_walker.sh 2 100000 dreamer_walker_seed2_100k
```

### TD-MPC2: cartpole

```bash
./scripts/train/run_tdmpc2_cartpole.sh 1 200000 5
```

Arguments:
- `$1` = seed
- `$2` = steps
- `$3` = model size
- `$4` = optional experiment name

Example:

```bash
./scripts/train/run_tdmpc2_cartpole.sh 2 200000 5 cartpole_200k_seed2
```

### TD-MPC2: walker

```bash
./scripts/train/run_tdmpc2_walker.sh 1 100000 1
```

Example:

```bash
./scripts/train/run_tdmpc2_walker.sh 2 100000 1 walker_100k_seed2
```

---

## Checkpointing and resume support

Both agents save and restore checkpoints so training can be interrupted and resumed.

### Dreamer
Dreamer checkpoints are stored under each run directory in:

```text
ckpt/
```

### TD-MPC2
TD-MPC2 checkpoints are stored under each run directory in:

```text
models/
```

This repository includes **selected representative checkpoints** rather than every intermediate checkpoint.

---

## Included representative checkpoints

Selected checkpoints are stored under:

```text
results/checkpoints/
```

Included representatives:

- `results/checkpoints/cartpole/dreamer_seed2/ckpt/`
- `results/checkpoints/cartpole/tdmpc2_seed2/models/`
- `results/checkpoints/walker/dreamer_seed2/ckpt/`
- `results/checkpoints/walker/tdmpc2_seed2/models/`

These were included to:
- demonstrate save/restore support
- provide representative trained states for both methods
- keep the repository smaller than a full raw-run dump

---

## Reproducing figures

### Cartpole multiseed CI figures
Script:

```text
scripts/plotting/plot_cartpole_multiseed_ci.py
```

Run:

```bash
python scripts/plotting/plot_cartpole_multiseed_ci.py
```

Expected outputs include:
- Dreamer cartpole mean ± 95% CI
- TD-MPC2 cartpole mean ± 95% CI
- combined cartpole comparison
- summary file with final means and CI widths

### Walker multiseed CI figures
Script:

```text
scripts/plotting/plot_walker_ci.py
```

Run:

```bash
python scripts/plotting/plot_walker_ci.py
```

Expected outputs include:
- Dreamer walker mean ± 95% CI
- TD-MPC2 walker mean ± 95% CI
- combined walker comparison
- walker summary file

### Cartpole latent-space analysis
Script:

```text
scripts/analysis/latent_analysis_cartpole_seed2.py
```

Run:

```bash
python scripts/analysis/latent_analysis_cartpole_seed2.py
```

Expected outputs include:
- latent PCA figures
- latent probe summary table values

---

## Main reported results

### Cartpole-swingup
Main findings:
- TD-MPC2 achieved much stronger mean performance than Dreamer on the shared 150k horizon
- Dreamer showed smaller cross-seed uncertainty
- TD-MPC2 latents were much more linearly predictive of reward and key task-state variables

### Walker-walk
Main findings:
- TD-MPC2 strongly outperformed Dreamer on the shared 95k horizon
- TD-MPC2 remained highly stable across seeds
- walker serves as a lower-budget extension supporting the main trend observed on cartpole

---

## Selected figures and summaries

Final figures used in the paper are placed in:

```text
results/final_figures/
```

Numeric summaries used in the paper are placed in:

```text
results/summaries/
```

These include:
- cartpole CI summary
- walker CI summary
- cartpole latent probe summary

---

## Notes on paths

The training wrappers are written for a repository-based layout, not for a single local machine path. They use environment variables such as:
- `PROJECT_ROOT`
- `DREAMER_ROOT`
- `DREAMER_VENV`
- `TD_ROOT`
- `TD_IMAGE`

This makes them more portable than hardcoded desktop-only scripts.

If necessary, override them at runtime. Example:

```bash
PROJECT_ROOT=/path/to/repo \
DREAMER_ROOT=/path/to/repo/third_party/dreamerv3 \
DREAMER_VENV=/path/to/venv \
./scripts/train/run_dreamer_walker.sh 1 100000
```

---

## What is intentionally excluded

To keep the repository manageable, the following are not fully included:
- full replay buffers
- full raw run directories for every seed
- every intermediate checkpoint
- temporary debug figures
- local-only artifacts unrelated to final reported results

The repo instead contains:
- scripts
- final report
- selected figures
- numeric summaries
- representative checkpoints
- reproduction instructions

---

## Paper

The paper draft and paper assets are in:

```text
paper/
```

It summarizes:
- literature framing
- methods
- experimental setup
- cartpole results
- walker extension
- latent-space analysis
- conclusion

---
