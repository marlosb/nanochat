import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"


@pytest.fixture(autouse=True)
def fake_uv(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$BENCHMARK_UV_CALLS"\n'
        'exit "${BENCHMARK_UV_EXIT:-0}"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    calls = tmp_path / "uv calls.txt"
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("BENCHMARK_UV_CALLS", str(calls))
    return calls


@pytest.fixture
def bash():
    candidates = []
    if os.name == "nt":
        candidates.append(str(Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                              / "Git" / "bin" / "bash.exe"))
    candidates.append(shutil.which("bash"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            result = subprocess.run([candidate, "--version"], capture_output=True)
            if result.returncode == 0:
                return candidate
    pytest.skip("Bash is not available")


def run_bash(bash, *args, cwd=None, env=None, script="base_eval_hf_benchmark.sh"):
    return subprocess.run(
        [bash, str(RUNS / script), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("launcher", ["bash", "batch", "parallel"])
def test_default_models_and_scoped_remote_code(bash, launcher, tmp_path):
    if launcher == "bash":
        result = run_bash(bash, "--dry-run", cwd=tmp_path)
    elif launcher == "parallel":
        output = tmp_path / "dry run output"
        result = run_bash(
            bash, "--dry-run", "--output-dir", str(output), cwd=tmp_path,
            script="base_eval_hf_benchmark_parallel.sh",
        )
        assert not output.exists()
    else:
        if os.name != "nt":
            pytest.skip("Windows batch requires Windows")
        result = subprocess.run(
            [os.environ["ComSpec"], "/c",
             str(RUNS / "base_eval_hf_benchmark.bat"), "--dry-run"],
            cwd=tmp_path, capture_output=True, text=True, timeout=30,
        )
    assert result.returncode == 0, result.stderr + result.stdout
    commands = [line for line in result.stdout.splitlines()
                if line.startswith("[dry-run]") and " -m scripts.base_eval " in line]
    assert len(commands) == 7
    assert "--hf-path openai-community/gpt2" in commands[0]
    assert "--hf-path marlosb/ArthurFriedenreich" in commands[-2]
    assert "--hf-path marlosb/CharlesMiller" in commands[-1]
    assert all("--trust-remote-code" not in line for line in commands[:-2])
    assert all("--trust-remote-code" in line for line in commands[-2:])
    assert all("--eval core" in line and "--max-per-task -1" in line for line in commands)
    assert all("--model-tag" not in line for line in commands)
    if launcher == "parallel":
        assert all("--output-dir" in line and "--device-type cuda" in line for line in commands)
        assert all("--append-report" not in line for line in commands)
    else:
        assert all("--append-report" in line for line in commands)


@pytest.mark.parametrize("stop_on_error", [False, True])
def test_failure_handling_and_option_forwarding(bash, tmp_path, stop_on_error):
    stub = tmp_path / "fake python.sh"
    calls = tmp_path / "calls.txt"
    stub.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$BENCHMARK_CALLS"\n'
        'printf "BASE=%s\\n" "$NANOCHAT_BASE_DIR" >> "$BENCHMARK_CALLS"\n'
        'case " $* " in *" --hf-path fail/model "*) exit 7 ;; esac\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    env = dict(os.environ, BENCHMARK_CALLS=str(calls))
    args = [
        "--python", str(stub), "--models", "fail/model,good/model",
        "--eval", "core,bpb", "--max-per-task", "20", "--device-type", "cuda",
        "--nanochat-base-dir", str(tmp_path / "base dir"),
    ]
    if stop_on_error:
        args.append("--stop-on-error")
    result = run_bash(bash, *args, cwd=tmp_path, env=env)
    assert result.returncode == 7, result.stderr + result.stdout
    recorded = calls.read_text(encoding="utf-8")
    assert "--hf-path fail/model" in recorded
    assert ("--hf-path good/model" in recorded) == (not stop_on_error)
    assert "--eval core,bpb" in recorded
    assert "--max-per-task 20" in recorded
    assert "--device-type cuda" in recorded
    assert f"BASE={tmp_path / 'base dir'}" in recorded


@pytest.mark.parametrize("script", [
    "base_eval_hf_benchmark.sh",
    "base_eval_hf_benchmark_parallel.sh",
])
@pytest.mark.parametrize("args", [
    ["--python"],
    ["--models", ", ,"],
    ["--unknown"],
])
def test_invalid_arguments(bash, args, script):
    result = run_bash(bash, *args, script=script)
    assert result.returncode != 0
    assert result.stderr


@pytest.mark.parametrize("script", [
    "base_eval_hf_benchmark.sh",
    "base_eval_hf_benchmark_parallel.sh",
])
def test_missing_python_fails_before_scheduling(bash, tmp_path, script):
    missing = str(tmp_path / "missing python")
    result = run_bash(bash, "--python", missing, script=script)
    assert result.returncode == 127
    assert "Python executable not found or not executable" in result.stderr
    assert missing in result.stderr
    assert "uv sync --extra gpu --group dev" in result.stderr
    assert "model running" not in result.stdout


@pytest.mark.parametrize("script", [
    "base_eval_hf_benchmark.sh",
    "base_eval_hf_benchmark_parallel.sh",
])
def test_sync_failure_stops_before_scheduling(bash, fake_uv, script):
    env = dict(os.environ, BENCHMARK_UV_EXIT="19")
    result = run_bash(bash, env=env, script=script)
    assert result.returncode == 19, result.stderr + result.stdout
    assert fake_uv.read_text().splitlines() == ["sync --extra gpu --group dev"]
    assert "model running" not in result.stdout


@pytest.mark.parametrize("script", [
    "base_eval_hf_benchmark.sh",
    "base_eval_hf_benchmark_parallel.sh",
])
def test_skip_sync_and_dry_run_do_not_install(bash, tmp_path, fake_uv, script):
    result = run_bash(bash, "--dry-run", script=script)
    assert result.returncode == 0
    assert "[dry-run] uv sync --extra gpu --group dev" in result.stdout
    assert not fake_uv.exists()
    result = run_bash(
        bash, "--skip-sync", "--python", str(tmp_path / "missing"), script=script,
    )
    assert result.returncode == 127
    assert not fake_uv.exists()


@pytest.fixture
def parallel_evaluator(tmp_path):
    stub = tmp_path / "evaluator.py"
    stub.write_text(
        'import argparse, json, os, time\n'
        'from pathlib import Path\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("--hf-path")\n'
        'parser.add_argument("--output-dir")\n'
        'args, extra = parser.parse_known_args()\n'
        'start = time.time()\n'
        'delays = json.loads(os.environ["BENCHMARK_DELAYS"])\n'
        'time.sleep(delays.get(args.hf_path, 0))\n'
        'output = Path(args.output_dir)\n'
        'output.joinpath("timing.json").write_text(json.dumps({\n'
        '    "model": args.hf_path, "start": start, "end": time.time(),\n'
        '    "extra": extra, "base": os.environ.get("NANOCHAT_BASE_DIR"),\n'
        '}))\n'
        'print("evaluated", args.hf_path)\n'
        'raise SystemExit(7 if args.hf_path == "fail/model" else 0)\n',
        encoding="utf-8",
    )
    shim = tmp_path / "fake python.sh"
    shim.write_text(
        '#!/usr/bin/env bash\n'
        'exec "$BENCHMARK_TEST_PYTHON" "$BENCHMARK_TEST_STUB" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    env = dict(
        os.environ, BENCHMARK_TEST_PYTHON=sys.executable,
        BENCHMARK_TEST_STUB=str(stub),
        BENCHMARK_DELAYS=json.dumps({"slow/model": 1, "fast/model": 0.05,
                                    "fail/model": 0.05}),
    )
    return shim, env


def test_parallel_limit_rolling_schedule_and_isolated_logs(bash, tmp_path, parallel_evaluator, fake_uv):
    shim, env = parallel_evaluator
    output = tmp_path / "parallel output"
    base = tmp_path / "shared data"
    result = run_bash(
        bash, "--python", str(shim),
        "--models", "slow/model,fast/model,third/model,fourth/model",
        "--output-dir", str(output), "--nanochat-base-dir", str(base),
        "--max-per-task", "20", "--eval", "core,bpb", cwd=tmp_path, env=env,
        script="base_eval_hf_benchmark_parallel.sh",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert fake_uv.read_text().splitlines() == ["sync --extra gpu --group dev"]
    timings = [json.loads(path.read_text()) for path in output.glob("*/timing.json")]
    assert len(timings) == 4
    by_model = {timing["model"]: timing for timing in timings}
    assert by_model["third/model"]["start"] < by_model["slow/model"]["end"]
    events = sorted((timing[key], change) for timing in timings
                    for key, change in [("start", 1), ("end", -1)])
    active, peak = 0, 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert peak == 2
    assert active == 0
    for timing in timings:
        assert timing["base"] == str(base)
        assert "--eval" in timing["extra"] and "core,bpb" in timing["extra"]
        assert "--max-per-task" in timing["extra"] and "20" in timing["extra"]
    logs = list(output.glob("*/evaluation.log"))
    assert len(logs) == 4
    assert all("evaluated" in log.read_text() for log in logs)


@pytest.mark.parametrize("stop_on_error", [False, True])
def test_parallel_failure_waits_for_active_models(bash, tmp_path, parallel_evaluator, stop_on_error):
    shim, env = parallel_evaluator
    output = tmp_path / "parallel output"
    args = [
        "--python", str(shim), "--models", "fail/model,slow/model,pending/model",
        "--output-dir", str(output),
    ]
    if stop_on_error:
        args.append("--stop-on-error")
    result = run_bash(
        bash, *args, env=env, script="base_eval_hf_benchmark_parallel.sh",
    )
    assert result.returncode == 7, result.stderr + result.stdout
    assert "evaluated fail/model" in result.stderr
    timings = [json.loads(path.read_text()) for path in output.glob("*/timing.json")]
    models = {timing["model"] for timing in timings}
    assert {"fail/model", "slow/model"} <= models
    assert ("pending/model" in models) == (not stop_on_error)
