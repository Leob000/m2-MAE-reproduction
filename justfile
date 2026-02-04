set shell := ["bash", "-c"]
set dotenv-load

# --- Cluster Configuration ---
# Use a .env file (git-ignored) to set these variables locally
cluster_host := env_var_or_default("CLUSTER_HOST", "user@host")
cluster_path := env_var_or_default("CLUSTER_PATH", "~/projects/m2-ovo-mae")
cluster_partition := env_var_or_default("CLUSTER_PARTITION", "gpu")

# Run all checks (format, lint, type-check) on the codebase
check path=".":
    @echo "Running checks on {{path}}..."
    uv run ruff format {{path}}
    uv run ruff check --fix {{path}}
    uv run ty check {{path}}

# Run the full suite including tests
full-check: (check) test

# Run code formatting
format path=".":
    uv run ruff format {{path}}

# Run linting with auto-fixes
lint path=".":
    uv run ruff check --fix {{path}}

# Run static type checking
type-check path=".":
    uv run ty check {{path}}

# Run tests
test:
    uv run pytest

# Run a single, fast training epoch to verify pipeline functionality
fast-train *args="":
    uv run python src/m2_ovo_mae/train_pretrain.py experiment=fast_run {{args}}

# Run a single, fast evaluation epoch to verify pipeline functionality
fast-eval *args="":
    uv run python src/m2_ovo_mae/train_classifier.py experiment=fast_eval {{args}}

# --- SLURM (Runs on the machine where it is called) ---

# Submit a training job to SLURM
slurm *args:
    mkdir -p slurm_logs
    uv run python scripts/slurm/submit.py {{args}} wandb.mode=online system=cluster_slurm

# --- Remote Cluster Management ---

# Sync local code to cluster (respects .gitignore)
sync:
    @echo "Syncing code to {{cluster_host}}..."
    rsync -avz --filter=':- .gitignore' --exclude '.git' --exclude 'wandb' --exclude 'outputs' ./ {{cluster_host}}:{{cluster_path}}

# Check SLURM queue on the cluster
sq *args="":
    ssh {{cluster_host}} "squeue -p {{cluster_partition}} {{args}}"

# Cancel SLURM jobs on the cluster
sc *args="":
    ssh {{cluster_host}} "scancel {{args}}"

# Remote execution: Always syncs code first, then runs a 'just' command on the cluster
# Usage: just remote <command>
remote *args: sync
    ssh {{cluster_host}} "cd {{cluster_path}} && CLUSTER_PARTITION={{cluster_partition}} just {{args}}"

# Remote execution via srun: Syncs code, then runs a 'just' command on a compute node
# Usage: just remote-srun <command>
remote-srun *args: sync
    ssh {{cluster_host}} "cd {{cluster_path}} && srun -p {{cluster_partition}} --gpus=1 just {{args}}"

# Tail the latest SLURM log on the cluster
logs:
    ssh {{cluster_host}} "ls -t {{cluster_path}}/slurm_logs/*.out | head -n 1 | xargs tail -f"
