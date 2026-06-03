#!/usr/bin/env python
"""
Run scripts.base_eval on a list of Hugging Face base models and write a Markdown report.

Example:
    python runs/base_eval_hf_benchmark.py --device-type cuda
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterable


DEFAULT_MODELS = [
    "Polygl0t/Tucano2-0.6B-Base",
    "Polygl0t/Tucano2-qwen-1.5B-Base",
    "TucanoBR/Tucano-630m",
    "TucanoBR/Tucano-1b1",
]


def _get_base_dir(env_base_dir: str | None) -> Path:
    if env_base_dir:
        return Path(env_base_dir).expanduser()
    return Path.home() / ".cache" / "nanochat"


def _model_slug(model: str) -> str:
    return model.replace("/", "-")


def _parse_core_metric(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip() == "CORE":
                try:
                    return float(row[2].strip())
                except (IndexError, ValueError):
                    return None
    return None


def _parse_core_metric_from_output(output: str) -> float | None:
    match = re.search(r"CORE metric:\s*([+-]?\d+(?:\.\d+)?)", output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _write_report(
    output_path: Path,
    base_dir: Path,
    eval_mode: str,
    max_per_task: int,
    rows: Iterable[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# Base Eval Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: `{timestamp}`")
    lines.append(f"- Eval mode: `{eval_mode}`")
    lines.append(f"- Max per task: `{max_per_task}`")
    lines.append(f"- NANOCHAT_BASE_DIR: `{base_dir}`")
    lines.append("")
    lines.append("| Model | Status | Exit | CORE metric | Duration (s) | CSV | Log |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")

    for r in rows:
        core_text = "n/a" if r["core_metric"] is None else f'{r["core_metric"]:.6f}'
        csv_text = f'`{r["csv_path"]}`' if r["csv_path"] else "-"
        log_text = f'`{r["log_path"]}`' if r["log_path"] else "-"
        lines.append(
            f'| `{r["model"]}` | {r["status"]} | {r["exit_code"]} | '
            f"{core_text} | {r['duration_s']:.2f} | {csv_text} | {log_text} |"
        )

    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for r in rows:
        lines.append(f"- `{r['command']}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark multiple HF models with scripts.base_eval.")
    parser.add_argument("--output-md", type=str, default="runs/base_eval_hf_report.md")
    parser.add_argument("--eval", type=str, default="core", help="Eval modes for scripts.base_eval.")
    parser.add_argument("--max-per-task", type=int, default=-1)
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=autodetect)")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after first failing model.")
    parser.add_argument("--nanochat-base-dir", type=str, default="")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_md = (repo_root / args.output_md).resolve()
    logs_dir = (repo_root / "runs" / "logs" / "base_eval_hf").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.nanochat_base_dir:
        env["NANOCHAT_BASE_DIR"] = args.nanochat_base_dir
    base_dir = _get_base_dir(env.get("NANOCHAT_BASE_DIR"))

    results: list[dict] = []
    overall_rc = 0

    for model in args.models:
        t0 = time.time()
        model_slug = _model_slug(model)
        log_path = logs_dir / f"{model_slug}.log"
        csv_path = base_dir / "base_eval" / f"{model_slug}.csv"

        cmd = [
            args.python,
            "-m",
            "scripts.base_eval",
            "--eval",
            args.eval,
            "--hf-path",
            model,
            "--max-per-task",
            str(args.max_per_task),
        ]
        if args.device_type:
            cmd.extend(["--device-type", args.device_type])

        print(f"{_timestamp()} - model running - {model}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        combined_lines: list[str] = []
        task_re = re.compile(r"Evaluating:\s*([^\(]+)\(")
        assert proc.stdout is not None
        for line in proc.stdout:
            combined_lines.append(line)
            task_match = task_re.search(line)
            if task_match:
                task = task_match.group(1).strip()
                print(f"{_timestamp()} - model running - {model} - task running - {task}", flush=True)
        proc.wait()
        duration_s = time.time() - t0
        combined_output = "".join(combined_lines)
        log_path.write_text(combined_output, encoding="utf-8", errors="replace")

        core_metric = _parse_core_metric(csv_path)
        if core_metric is None:
            core_metric = _parse_core_metric_from_output(combined_output)

        status = "ok" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            overall_rc = proc.returncode

        results.append(
            {
                "model": model,
                "status": status,
                "exit_code": proc.returncode,
                "core_metric": core_metric,
                "duration_s": duration_s,
                "csv_path": str(csv_path) if csv_path.exists() else "",
                "log_path": str(log_path),
                "command": " ".join(cmd),
            }
        )

        _write_report(output_md, base_dir, args.eval, args.max_per_task, results)

        if proc.returncode != 0 and args.stop_on_error:
            break

    _write_report(output_md, base_dir, args.eval, args.max_per_task, results)
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
