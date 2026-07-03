# UV-based environment setup on Debian cluster

This guide replaces the old conda-based workflow with [`uv`](https://docs.astral.sh/uv/) on a Debian compute cluster. It is based on the dependencies declared in `environment.yaml` and `requirements.txt`.

## 1. Install `uv`

`uv` is distributed as a single static binary. Pick one of the following methods.

### Option A: Install to user `~/.local/bin` (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add to your `~/.bashrc` if the installer does not do it automatically:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Reload and verify:

```bash
source ~/.bashrc
uv --version
```

### Option B: Install via pip

```bash
python3 -m pip install --user uv
```

## 2. System dependencies (optional, no sudo required)

Most packages in this repo install from pre-built wheels on PyPI, including `python-igraph` and `leidenalg` — their wheels bundle the igraph C library, so you usually do **not** need system headers or sudo.

If you *do* have sudo/admin rights, installing the build toolchain makes fallback builds reliable:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential python3-dev pkg-config \
    libigraph-dev libxml2-dev zlib1g-dev
```

If you do **not** have sudo, try the pure `uv`/`pip` install first (steps 3–5). Only come back here if a package fails to build.

## 3. Create a Python 3.11 virtual environment

`environment.yaml` pins Python 3.11. Create a managed environment inside the project:

```bash
cd /path/to/Rewiring-Collective-Action
uv venv .venv --python 3.11
```

Activate it:

```bash
source .venv/bin/activate
```

## 4. Install Python dependencies

`uv` can install directly from `requirements.txt`. Both files list the same runtime packages; `requirements.txt` is the simplest source for `uv`.

```bash
uv pip install -r requirements.txt
```

If you prefer to mirror the `environment.yaml` exactly (Python 3.11 + the same package list):

```bash
uv pip install \
    "numpy==1.24.2" \
    pandas scipy matplotlib seaborn networkx rustworkx joblib \
    setuptools wheel \
    netin node2vec leidenalg python-igraph
```

## 5. Verify the installation

Run a quick import check:

```bash
python - <<'PY'
import numpy, pandas, scipy, matplotlib, seaborn
import networkx, rustworkx, joblib
import igraph, leidenalg, node2vec, netin
print("All imports OK")
print("Python:", __import__("sys").version)
print("numpy:", numpy.__version__)
print("netin:", netin.__version__)
PY
```

## 6. Optional: Rust toolchain for `Auxillary/fast_wtf`

The repository contains a Rust extension for performance-critical rewiring code. If you intend to use it, install Rust and build the extension:

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the extension
cd Auxillary/fast_wtf
cargo build --release
```

Make sure `cargo` can find the Python headers for your activated `.venv`.

## 7. Cluster workflow summary

```bash
# 1. Login and move to project
cd /path/to/Rewiring-Collective-Action

# 2. Activate environment (do this in every job/terminal)
source .venv/bin/activate

# 3. Run model or sweeps
cd Analysis
python run.py
python general_param_sweep.py
```

## 8. Reproducibility tips

- Pin versions in `requirements.txt` after a successful install:
  ```bash
  uv pip freeze > requirements-locked.txt
  ```
- If you submit SLURM jobs, activate the venv inside the job script before calling Python.
- Avoid installing across network filesystems with different architectures; `.venv` paths are not relocatable. Re-create the environment on each node type if CPU instruction sets differ significantly.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Ensure `~/.local/bin` is in `PATH` |
| `python-igraph` build fails | Try `uv pip install --only-binary :all: python-igraph` to force the wheel. If no wheel matches your platform, ask the cluster admin for `libigraph-dev`, or build igraph in user space |
| `leidenalg` build fails | Usually resolved once `python-igraph` is installed correctly |
| Permission denied on shared cluster | Use `uv venv .venv` inside your project/home directory, not system paths |
| Matplotlib backend errors on headless nodes | Set `export MPLBACKEND=Agg` in job scripts |
