"""变体效应：模板/图片/检索顺序/搜索返回的真实消费变化必须可观察。"""

from __future__ import annotations

import copy
import json

import dataclasses

from agentsRunner import Runner
from conftest import ROOT_FOR_TESTS
from replay_support import digest_file


def _run(services, fixture_case, output_dir):
    runner = Runner(config=services.config, services=services, root=ROOT_FOR_TESTS)
    outcome = runner.run_case(fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", output_dir)
    assert outcome["success"], outcome
    return outcome


def _consumed(output_dir):
    return json.loads((output_dir / "consumed.json").read_text(encoding="utf-8"))


def test_image_change_changes_consumed_digest(
    make_services, fixture_case, trial_output_dir
):
    services_a = make_services()
    _run(services_a, fixture_case, trial_output_dir / "a")
    consumed_a = _consumed(trial_output_dir / "a")

    # 换一张有效 PNG（不同字节，同一路径），通过新病例与新图片驱动
    alt_png = ROOT_FOR_TESTS / "trial" / "cases" / "alt-test.png"
    from tools.trial_tools import make_png  # 复用确定性 PNG 生成器

    import sys

    sys.path.insert(0, str(ROOT_FOR_TESTS / "tools"))
    alt_png.write_bytes(make_png(48, 48, seed=9))
    try:
        case_v2 = copy.deepcopy(json.loads(
            (ROOT_FOR_TESTS / "trial/cases/case-001.json").read_text(encoding="utf-8")
        ))
        case_v2["image"] = "trial/cases/alt-test.png"
        case_v2_path = trial_output_dir / "case-alt.json"
        case_v2_path.write_text(json.dumps(case_v2, ensure_ascii=False), encoding="utf-8")
        # case 路径在 trial/outputs 下也属于仓库内，可被 resolve
        from replay_support import load_case

        alt_case = load_case(case_v2_path)
        services_b = make_services()
        runner = Runner(config=services_b.config, services=services_b, root=ROOT_FOR_TESTS)
        outcome = runner.run_case(alt_case, alt_png, trial_output_dir / "b")
        assert outcome["success"], outcome
        consumed_b = _consumed(trial_output_dir / "b")
        assert (
            consumed_a["inputs"]["image"]["sha256"] != consumed_b["inputs"]["image"]["sha256"]
        ), "图片字节变化必须改变实际消费摘要"
    finally:
        alt_png.unlink()


def test_prompt_change_changes_actor_request(
    make_services, fixture_case, trial_output_dir
):
    services_a = make_services()
    _run(services_a, fixture_case, trial_output_dir / "a")
    actor_a = [r for r in services_a.calls.calls if r.role == "actor"][0]

    # 用带标记的 actor 模板重建服务
    import json as jsonlib

    cassette = jsonlib.loads(
        (ROOT_FOR_TESTS / "trial/fixtures/external.json").read_text(encoding="utf-8")
    )
    services_b = make_services(cassette=cassette)
    # 替换 actor 模板来源：直接改 config.prompts 不行（frozen），用换路径的配置
    marker_prompt = trial_output_dir / "actor-marker.txt"
    base_prompt = (ROOT_FOR_TESTS / "trial/prompts/actor.txt").read_text(encoding="utf-8")
    marker_prompt.write_text(base_prompt + "\n[PROMPT_V2 测试标记]\n", encoding="utf-8")
    new_prompts = dict(services_b.config.prompts)
    new_prompts["actor"] = str(marker_prompt.relative_to(ROOT_FOR_TESTS)).replace("\\", "/")
    cfg = dataclasses.replace(services_b.config, prompts=new_prompts)
    services_b.config = cfg
    _run(services_b, fixture_case, trial_output_dir / "b")

    actor_b = [r for r in services_b.calls.calls if r.role == "actor"][0]
    assert actor_a.request_digest != actor_b.request_digest, "模板变化必须改变实际消息摘要"


def test_retrieval_reorder_changes_downstream_order(
    make_services, fixture_case, trial_output_dir
):
    import json as jsonlib

    docs = jsonlib.loads(
        (ROOT_FOR_TESTS / "trial/fixtures/retrieval.json").read_text(encoding="utf-8")
    )["docs"]
    reordered = [docs[1], docs[0], docs[2]]

    services_a = make_services(retrieval_docs=docs)
    _run(services_a, fixture_case, trial_output_dir / "a")
    actor_a = jsonlib.dumps(
        [r.payload for r in services_a.calls.calls if r.role == "actor"][0], ensure_ascii=False
    )

    services_b = make_services(retrieval_docs=reordered)
    _run(services_b, fixture_case, trial_output_dir / "b")
    actor_b = jsonlib.dumps(
        [r.payload for r in services_b.calls.calls if r.role == "actor"][0], ensure_ascii=False
    )

    assert actor_a != actor_b, "检索顺序变化必须改变下游实际上下文"
    retrieval_b = [e for e in services_b.recorder.events if e["event"] == "retrieval_docs"][0]
    assert [d["id"] for d in retrieval_b["docs"]] == ["doc-b", "doc-a", "doc-c"]


def test_search_v2_changes_report_input(make_services, fixture_case, trial_output_dir):
    import json as jsonlib

    base_cassette = jsonlib.loads(
        (ROOT_FOR_TESTS / "trial/fixtures/external.json").read_text(encoding="utf-8")
    )
    services_a = make_services(cassette=base_cassette)
    _run(services_a, fixture_case, trial_output_dir / "a")
    report_a = jsonlib.dumps(
        [r.payload for r in services_a.calls.calls if r.role == "report_main"],
        ensure_ascii=False,
    )

    v2 = copy.deepcopy(base_cassette)
    v2["search"]["keyword"][0]["title"] = "脾胃湿热的饮食调理（search-v2 修订版）"
    services_b = make_services(cassette=v2)
    _run(services_b, fixture_case, trial_output_dir / "b")
    report_b = jsonlib.dumps(
        [r.payload for r in services_b.calls.calls if r.role == "report_main"],
        ensure_ascii=False,
    )

    assert report_a != report_b, "搜索返回变化必须进入报告输入"
    assert "search-v2 修订版" in report_b
