"""失败边界：各类输入/配置/预算错误必须非零退出并留失败回执。"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from conftest import ROOT_FOR_TESTS
from replay_case import main as replay_main
from replay_support import (
    FixtureSearchBackend,
    TrialError,
)
from agentsRunner import Runner


@pytest.fixture()
def bad_png_in_repo():
    """仓库内一个坏 PNG（文本字节伪装 .png 扩展），测试后删除。"""
    path = ROOT_FOR_TESTS / "trial" / "cases" / "corrupt-test.png"
    path.write_text("这不是PNG", encoding="utf-8")
    yield "trial/cases/corrupt-test.png"
    path.unlink()


def base_argv(output_rel: str) -> list:
    return [
        "--mode", "fixture",
        "--case", "trial/cases/case-001.json",
        "--image", "trial/cases/case-001.png",
        "--config", "trial/configs/current.yaml",
        "--cassette", "trial/fixtures/external.json",
        "--output-dir", output_rel,
    ]


def test_missing_case_fails(tmp_path):
    code = replay_main(base_argv("trial/outputs/pytest-missing-case") + [
        "--case", "trial/cases/no-such-case.json",
    ])
    assert code == 2


def test_corrupt_image_fails(bad_png_in_repo, trial_output_dir):
    rel = trial_output_dir.relative_to(ROOT_FOR_TESTS).as_posix()
    argv = base_argv(rel)
    argv[argv.index("trial/cases/case-001.png")] = bad_png_in_repo
    code = replay_main(argv)
    assert code == 2
    # 早期校验失败不写产物（输出目录保持为空）
    assert list(trial_output_dir.iterdir()) == []


def test_nonempty_output_dir_rejected(trial_output_dir):
    (trial_output_dir / "leftover.txt").write_text("占用", encoding="utf-8")
    rel = trial_output_dir.relative_to(ROOT_FOR_TESTS).as_posix()
    assert replay_main(base_argv(rel)) == 2


def test_out_of_bounds_path_rejected(tmp_path):
    outside = tmp_path / "outside-output"
    outside.mkdir()
    code = replay_main(base_argv(str(outside)))
    assert code == 2


def test_invalid_config_rejected(trial_output_dir):
    bad_config = ROOT_FOR_TESTS / "trial" / "configs" / "pytest-invalid.yaml"
    raw = json.loads(json.dumps(_fixture_config_dict()))
    raw["workflow"]["unknown_key"] = 1
    import yaml

    bad_config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    rel = trial_output_dir.relative_to(ROOT_FOR_TESTS).as_posix()
    argv = base_argv(rel)
    argv[argv.index("trial/configs/current.yaml")] = "trial/configs/pytest-invalid.yaml"
    try:
        assert replay_main(argv) == 2
    finally:
        bad_config.unlink()


def test_budget_exceeded_stops_before_next_request(make_services, fixture_case, trial_output_dir):
    services = make_services()
    import dataclasses

    from replay_support import ModelCallBudget

    cfg = dataclasses.replace(services.config, model_call_limit=3)
    services.config = cfg
    services.budget = ModelCallBudget(3)
    runner = Runner(config=cfg, services=services, root=ROOT_FOR_TESTS)
    outcome = runner.run_case(
        fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", trial_output_dir
    )
    assert outcome["success"] is False
    assert outcome["failure"]["reason_code"] == "budget_exceeded"
    # 超限后没有再发出下一次请求：模型调用数恰为上限
    model_calls = [
        r for r in services.calls.calls if r.kind in ("chat", "vision")
    ]
    assert len(model_calls) == 3
    # 失败回执已写出，保留已完成阶段
    result = json.loads((trial_output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["success"] is False
    assert result["stages"]["classification"]["status"] == "success"
    assert not (trial_output_dir / "report.md").exists()


def test_search_failure_preserves_completed_stages(make_services, fixture_case, trial_output_dir):
    class FailingSearch(FixtureSearchBackend):
        def search_terms(self, items):
            raise TimeoutError("模拟搜索超时")

    services = make_services()
    services.search = FailingSearch(services.search._scripted)
    recorded = services.recorded_search
    runner = Runner(config=services.config, services=services, root=ROOT_FOR_TESTS)
    runner._search_client = recorded
    outcome = runner.run_case(
        fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", trial_output_dir
    )
    assert outcome["success"] is False
    assert outcome["failure"]["stage"] == "links"
    assert outcome["failure"]["reason_code"] == "search_failed"
    # 已完成阶段保留，不生成成功 result，不用 fixture 冒充失败请求的替代
    result = json.loads((trial_output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["success"] is False
    for stage in ("classification", "vision", "informer", "retrieval", "planning", "consultation"):
        assert result["stages"][stage]["status"] == "success", stage
    assert result["stages"]["report"]["status"] == "skipped"
    assert not (trial_output_dir / "report.md").exists()
    trace_text = (trial_output_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "consultation_done" in trace_text


def _fixture_config_dict() -> dict:
    import yaml

    raw = yaml.safe_load(
        (ROOT_FOR_TESTS / "trial/configs/current.yaml").read_text(encoding="utf-8")
    )
    return raw
