"""
Run both d24 pretraining stages:
1. gigaverbo-v2
2. gigaverbo-v2-synth, resumed from the stage-1 checkpoint

Run from the repository root inside the project environment:
    WANDB_RUN=full python runs/full_base_train.py
"""

import math
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.checkpoint_manager import find_last_step


DEPTH = 24
SCALING_PARAMS = 780_142_272


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def run_command(*args):
    command = [str(arg) for arg in args]
    print()
    print("------------------------------------------------------------")
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def start_command(*args):
    command = [str(arg) for arg in args]
    print()
    print("------------------------------------------------------------")
    print("[BACKGROUND]", " ".join(command), flush=True)
    return subprocess.Popen(command, cwd=REPO_ROOT)


def wait_for_download(process):
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        process.wait()


def torchrun_args(nproc_per_node):
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
    )


def base_train_args(
    nproc_per_node,
    device_batch_size,
    total_batch_size,
    checkpoint_every,
    eval_every,
    sample_every,
    core_metric_every,
    wandb_run,
):
    return (
        *torchrun_args(nproc_per_node),
        "-m",
        "scripts.base_train",
        "--",
        f"--depth={DEPTH}",
        "--target-param-data-ratio=120",
        f"--device-batch-size={device_batch_size}",
        f"--total-batch-size={total_batch_size}",
        f"--eval-every={eval_every}",
        f"--sample-every={sample_every}",
        f"--core-metric-every={core_metric_every}",
        f"--save-every={checkpoint_every}",
        "--fp8",
        "--dataset=gigaverbo-v2",
        f"--run={wandb_run}",
    )


def synth_train_args(
    nproc_per_node,
    device_batch_size,
    total_batch_size,
    checkpoint_every,
    eval_every,
    sample_every,
    core_metric_every,
    synth_ratio,
    synth_warmdown_ratio,
    last_base_step,
    synth_end_step,
    wandb_run,
):
    return (
        *torchrun_args(nproc_per_node),
        "-m",
        "scripts.base_train",
        "--",
        f"--depth={DEPTH}",
        f"--num-iterations={synth_end_step}",
        f"--target-param-data-ratio={synth_ratio}",
        f"--warmdown-ratio={synth_warmdown_ratio}",
        f"--device-batch-size={device_batch_size}",
        f"--total-batch-size={total_batch_size}",
        f"--eval-every={eval_every}",
        f"--sample-every={sample_every}",
        f"--core-metric-every={core_metric_every}",
        f"--save-every={checkpoint_every}",
        "--fp8",
        "--dataset=gigaverbo-v2-synth",
        "--model-tag=d24",
        "--resume-model-tag=d24",
        f"--resume-from-step={last_base_step}",
        f"--run={wandb_run}",
    )


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault(
        "NANOCHAT_BASE_DIR",
        str(Path.home() / ".cache" / "nanochat"),
    )

    nproc_per_node = env_int("NPROC_PER_NODE", 2)
    device_batch_size = env_int("DEVICE_BATCH_SIZE", 19)
    total_batch_size = env_int("TOTAL_BATCH_SIZE", 77_824)
    checkpoint_every = env_int("CHECKPOINT_EVERY", 60_000)
    eval_every = env_int("EVAL_EVERY", 20_000)
    sample_every = env_int("SAMPLE_EVERY", 10_000)
    core_metric_every = env_int("CORE_METRIC_EVERY", 10_000)
    synth_ratio = env_float("SYNTH_TARGET_PARAM_DATA_RATIO", 8.70)
    synth_warmdown_ratio = env_float("SYNTH_WARMDOWN_RATIO", 1.0)
    wandb_run = os.environ.get("WANDB_RUN", "full")

    python = sys.executable
    run_command(python, "-m", "nanochat.report", "reset")

    # Stage 1 starts after four shards; the remaining shards download concurrently.
    run_command(
        python,
        "-m",
        "nanochat.dataset",
        "--dataset",
        "gigaverbo-v2",
        "-n",
        "4",
    )
    base_download = start_command(
        python,
        "-m",
        "nanochat.dataset",
        "--dataset",
        "gigaverbo-v2",
        "-n",
        "120",
    )
    try:
        run_command(
            *base_train_args(
                nproc_per_node,
                device_batch_size,
                total_batch_size,
                checkpoint_every,
                eval_every,
                sample_every,
                core_metric_every,
                wandb_run,
            )
        )
        wait_for_download(base_download)
    finally:
        stop_process(base_download)

    run_command(
        *torchrun_args(nproc_per_node),
        "-m",
        "scripts.base_eval",
        "--",
        f"--device-batch-size={device_batch_size}",
    )

    # Stage 2 resumes the model and optimizer from the latest stage-1 checkpoint.
    run_command(
        python,
        "-m",
        "nanochat.dataset",
        "--dataset",
        "gigaverbo-v2-synth",
        "-n",
        "4",
    )
    synth_download = start_command(
        python,
        "-m",
        "nanochat.dataset",
        "--dataset",
        "gigaverbo-v2-synth",
        "-n",
        "-1",
    )
    checkpoint_dir = (
        Path(os.environ["NANOCHAT_BASE_DIR"]) / "base_checkpoints" / "d24"
    )
    last_base_step = find_last_step(checkpoint_dir)
    synth_extra_steps = math.ceil(
        synth_ratio * SCALING_PARAMS / total_batch_size
    )
    synth_end_step = last_base_step + synth_extra_steps
    print(
        f"[INFO] Resuming d24 from step {last_base_step} to {synth_end_step} "
        f"on gigaverbo-v2-synth",
        flush=True,
    )

    try:
        run_command(
            *synth_train_args(
                nproc_per_node,
                device_batch_size,
                total_batch_size,
                checkpoint_every,
                eval_every,
                sample_every,
                core_metric_every,
                synth_ratio,
                synth_warmdown_ratio,
                last_base_step,
                synth_end_step,
                wandb_run,
            )
        )
        wait_for_download(synth_download)
    finally:
        stop_process(synth_download)

    run_command(
        *torchrun_args(nproc_per_node),
        "-m",
        "scripts.base_eval",
        "--",
        f"--device-batch-size={device_batch_size}",
    )
    run_command(python, "-m", "nanochat.report", "generate")


if __name__ == "__main__":
    main()
