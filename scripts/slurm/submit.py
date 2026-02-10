import argparse
import itertools
import os
import subprocess


def parse_args():
    """Parses command line arguments for SLURM submission."""
    parser = argparse.ArgumentParser(
        description="Submit jobs to SLURM with optional sweeping."
    )
    parser.add_argument(
        "--partition",
        default=os.getenv("CLUSTER_PARTITION", "gpu"),
        help="SLURM partition",
    )
    parser.add_argument("--time", default="24:00:00", help="Time limit")
    parser.add_argument("--gpus", default="1", help="Number of GPUs")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (e.g., training.lr=0.01)",
    )
    return parser.parse_args()


def parse_overrides(overrides):
    """Parses a list of override strings into a list of (key, [values]) tuples.

    Args:
        overrides: List of strings like ['lr=0.1,0.01', 'batch=32'].
    """
    parsed = []
    for item in overrides:
        # Remove '--' if passed by argparse separator
        if item == "--":
            continue

        if "=" in item:
            key, val = item.split("=", 1)
            # Check for sweep syntax: comma separated, but not inside brackets []
            if "," in val and not (val.startswith("[") and val.endswith("]")):
                values = val.split(",")
            else:
                values = [val]
            parsed.append((key, values))
        else:
            # Flags or other arguments without = (keep as is, treated as single value)
            parsed.append((item, [None]))
    return parsed


def build_combinations(parsed_overrides):
    """Yields all combinations of override arguments.

    Args:
        parsed_overrides: List of (key, [values]) tuples.
    """
    keys = [item[0] for item in parsed_overrides]
    value_lists = [item[1] for item in parsed_overrides]

    for combination in itertools.product(*value_lists):
        cmd_args = []
        for key, val in zip(keys, combination, strict=False):
            if val is None:
                cmd_args.append(key)
            else:
                cmd_args.append(f"{key}={val}")
        yield cmd_args


def submit_job(partition, time, gpus, cmd_args):
    """Submits a single job to SLURM via sbatch.

    Args:
        partition: SLURM partition name.
        time: Time limit for the job.
        gpus: Number of GPUs requested.
        cmd_args: List of command line arguments for the python script.
    """
    # Determine the entry point (pretrain or classifier)
    script_path = "src/m2_ovo_mae/train_pretrain.py"
    if any(
        "classifier" in arg or "linprobe" in arg or "finetune" in arg
        for arg in cmd_args
    ):
        script_path = "src/m2_ovo_mae/train_classifier.py"

    python_cmd = f"uv run python {script_path} {' '.join(cmd_args)}"

    sbatch_cmd = [
        "sbatch",
        f"--partition={partition}",
        f"--time={time}",
        f"--gres=gpu:{gpus}",
        "scripts/slurm/job.slurm",
        python_cmd,
    ]

    print(f"Submitting: {python_cmd}")
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  -> {result.stdout.strip()}")
    else:
        print(f"  -> ERROR: {result.stderr.strip()}")


def main():
    """Main entry point for the SLURM submission script."""
    args = parse_args()
    parsed_overrides = parse_overrides(args.overrides)

    # Check if we are doing a sweep
    total_jobs = 1
    for _, vals in parsed_overrides:
        total_jobs *= len(vals)

    if total_jobs > 1:
        print(f"Detected sweep configuration. Launching {total_jobs} jobs.")

    for cmd_args in build_combinations(parsed_overrides):
        submit_job(args.partition, args.time, args.gpus, cmd_args)


if __name__ == "__main__":
    main()
