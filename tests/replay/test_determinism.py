"""确定性验收：固定输入两次运行的 report 字节相同、事件序列相同。"""

from __future__ import annotations

import json

from replay_case import run as replay_run
from conftest import ROOT


def _outputs(output_dir):
    report = (output_dir / "report.md").read_bytes()
    trace = (output_dir / "trace.jsonl").read_bytes()
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    return report, trace, result


def test_report_and_trace_byte_identical(trial_output_dir):
    first_dir = trial_output_dir / "run-a"
    second_dir = trial_output_dir / "run-b"
    first_dir.mkdir()
    second_dir.mkdir()
    rel = trial_output_dir.relative_to(ROOT).as_posix()
    assert replay_run([
        "--mode", "fixture",
        "--case", "trial/cases/case-001.json",
        "--image", "trial/cases/case-001.png",
        "--config", "trial/configs/current.yaml",
        "--cassette", "trial/fixtures/external.json",
        "--output-dir", f"{rel}/run-a",
    ]) == 0
    assert replay_run([
        "--mode", "fixture",
        "--case", "trial/cases/case-001.json",
        "--image", "trial/cases/case-001.png",
        "--config", "trial/configs/current.yaml",
        "--cassette", "trial/fixtures/external.json",
        "--output-dir", f"{rel}/run-b",
    ]) == 0

    report_a, trace_a, result_a = _outputs(first_dir)
    report_b, trace_b, result_b = _outputs(second_dir)
    assert report_a == report_b, "report 必须逐字节相同"
    assert trace_a == trace_b, "事件序列必须逐字节相同（fixture 固定时钟）"

    # 运行时间字段单列：result.json 的 duration/finished_at 允许不同，
    # 但除时间字段外结构一致，不得用时间字段冒充整份字节相同。
    for key in ("duration_seconds", "finished_at"):
        result_a.pop(key)
        result_b.pop(key)
    assert result_a == result_b
