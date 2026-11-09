"""fixture 端到端：无 stdin、无 torch/LangChain 环境下运行真实 Runner 编排。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import BACKEND, ROOT

REPLAY_CASE = BACKEND / "replay_case.py"


def trial_python() -> str:
    venv_python = ROOT / ".venv-trial" / "Scripts" / "python.exe"
    if venv_python.is_file():
        return str(venv_python)
    venv_python = ROOT / ".venv-trial" / "bin" / "python"
    assert venv_python.is_file(), "试用解释器不存在"
    return str(venv_python)


def run_replay(output_dir: Path, extra: list | None = None) -> subprocess.CompletedProcess:
    import os

    argv = [
        trial_python(),
        str(REPLAY_CASE),
        "--mode", "fixture",
        "--case", "trial/cases/case-001.json",
        "--image", "trial/cases/case-001.png",
        "--config", "trial/configs/current.yaml",
        "--cassette", "trial/fixtures/external.json",
        "--output-dir", str(output_dir.relative_to(ROOT)).replace("\\", "/"),
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DEEP_SEEK", "STEP_FUN", "SERPER", "PYTHON"))
    }
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        argv + (extra or []),
        cwd=ROOT,
        capture_output=True,
        stdin=subprocess.DEVNULL,  # stdin 关闭仍必须可运行
        env=env,
    )


def test_fixture_end_to_end_without_heavy_deps(trial_output_dir):
    proc = run_replay(trial_output_dir)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    result = json.loads((trial_output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["success"] is True
    assert result["failure"] is None
    for name in ("report.md", "trace.jsonl", "consumed.json", "result.json"):
        assert (trial_output_dir / name).is_file(), name


def test_only_expected_output_files(trial_output_dir):
    run_replay(trial_output_dir)
    names = sorted(p.name for p in trial_output_dir.iterdir())
    assert names == ["consumed.json", "report.md", "result.json", "trace.jsonl"]
