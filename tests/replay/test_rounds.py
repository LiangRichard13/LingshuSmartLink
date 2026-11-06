"""轮数语义与边界：5/3 轮真实消费对应次数；提前完成与规划上限留原因。"""

from __future__ import annotations

import copy
import dataclasses
import json

from agentsRunner import Runner
from conftest import ROOT_FOR_TESTS


def test_turn_limit_5_vs_3_consumes_respectively(make_services, fixture_case, trial_output_dir):
    outcomes = {}
    for limit in (5, 3):
        services = make_services()
        cfg = dataclasses.replace(services.config, chat_turn_limit=limit)
        services.config = cfg
        runner = Runner(config=cfg, services=services, root=ROOT_FOR_TESTS)
        outcome = runner.run_case(
            fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", trial_output_dir / f"run-{limit}"
        )
        assert outcome["success"], outcome
        instructor_calls = [
            r for r in services.calls.calls if r.kind == "chat" and r.role == "instructor"
        ]
        assert len(instructor_calls) == limit * 4, f"limit={limit}"
        outcomes[limit] = outcome
    assert outcomes[5]["report_path"] != outcomes[3]["report_path"]


def test_early_done_records_reason(make_services, fixture_case, trial_output_dir):
    cassette_path = ROOT_FOR_TESTS / "trial/fixtures/external.json"
    cassette = json.loads(cassette_path.read_text(encoding="utf-8"))
    done = copy.deepcopy(cassette)
    # 顺序消费语义：每个任务的第一次 Instructor 调用都给出结束标记
    done["chat"]["instructor"] = ["指令：可以了。<CAMEL_TASK_DONE>"] * 4
    services = make_services(cassette=done)
    runner = Runner(config=services.config, services=services, root=ROOT_FOR_TESTS)
    outcome = runner.run_case(
        fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", trial_output_dir / "done"
    )
    assert outcome["success"]
    done_events = [e for e in services.recorder.events if e["event"] == "consultation_done"]
    reasons = [task["reason"] for task in done_events[0]["tasks"]]
    assert reasons == ["task_done", "task_done", "task_done", "task_done"]
    instructor_calls = [
        r for r in services.calls.calls if r.kind == "chat" and r.role == "instructor"
    ]
    assert len(instructor_calls) == 4  # 每任务 1 次，提前完成


def test_planner_exhaustion_is_explicit_failure(make_services, fixture_case, trial_output_dir):
    cassette_path = ROOT_FOR_TESTS / "trial/fixtures/external.json"
    cassette = json.loads(cassette_path.read_text(encoding="utf-8"))
    bad = copy.deepcopy(cassette)
    bad["chat"]["planner"] = [
        "任务1<TASK_SPLIT>任务2",
        "任务1<TASK_SPLIT>任务2<TASK_SPLIT>任务3",
    ]  # 两次尝试数量都不符，随后触发上限
    services = make_services(cassette=bad)
    cfg = dataclasses.replace(services.config, planner_max_attempts=2)
    services.config = cfg
    runner = Runner(config=cfg, services=services, root=ROOT_FOR_TESTS)
    outcome = runner.run_case(
        fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", trial_output_dir
    )
    assert outcome["success"] is False
    assert outcome["failure"]["stage"] == "planning"
    assert outcome["failure"]["reason_code"] == "planner_exhausted"
    # 规划失败后不得进入协作阶段
    assert outcome["stages"]["consultation"]["status"] == "skipped"
    assert not (trial_output_dir / "report.md").exists()
