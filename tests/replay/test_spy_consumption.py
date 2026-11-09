"""消费见证：spy 侧检查每个角色真正传入 invoke/create 的 model/参数/消息摘要。"""

from __future__ import annotations

from agentsRunner import Runner
from replay_support import CaseSpec


def test_spy_sees_real_models_and_params(make_services, fixture_case: CaseSpec, trial_output_dir):
    services = make_services()
    runner = Runner(config=services.config, services=services, root=ROOT_FOR_TESTS)
    outcome = runner.run_case(fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", trial_output_dir)
    assert outcome["success"], outcome

    records = services.calls.calls
    chat_records = [r for r in records if r.kind == "chat"]
    vision_records = [r for r in records if r.kind == "vision"]

    # 每个角色真正使用的 model
    roles = {r.role for r in chat_records}
    assert roles == {"informer", "planner", "actor", "instructor", "summary", "links",
                     "report_meta", "report_main"}
    assert all(r.model == "fixture-scripted-v1" for r in chat_records)
    # 视觉请求不再使用硬编码 model；与配置一致
    assert {r.model for r in vision_records} == {"fixture-scripted-vision-v1"}

    # 采样参数真正进入请求构造（generation 0.2/1.0/512）
    assert all(
        r.params == {"temperature": 0.2, "top_p": 1.0, "max_tokens": 512}
        for r in chat_records
    )

    # 消息摘要进入 trace（请求/响应摘要逐条记录）
    request_events = [e for e in services.recorder.events if e["event"] == "llm_request"]
    assert len(request_events) == len(chat_records)
    assert all(e["request_digest"].startswith("sha256:") for e in request_events)

    # 视觉请求摘要真实包含图片字节（data URL 进入请求构造）
    vision_requests = [r for r in records if r.kind == "vision"]
    assert vision_requests and vision_requests[0].request_digest.startswith("sha256:")


def test_actor_initial_context_keeps_retrieval_order(make_services, fixture_case, trial_output_dir):
    import json

    services = make_services()
    runner = Runner(config=services.config, services=services, root=ROOT_FOR_TESTS)
    outcome = runner.run_case(fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", trial_output_dir)
    assert outcome["success"]

    actor_requests = [r for r in services.calls.calls if r.kind == "chat" and r.role == "actor"]
    first_payload = json.dumps(actor_requests[0].payload, ensure_ascii=False)
    doc_a = "脾胃湿热的成因与调理原则（合成语料A）"
    doc_b = "脾主运化的证候要点（合成语料B）"
    assert doc_a in first_payload and doc_b in first_payload
    assert first_payload.index(doc_a) < first_payload.index(doc_b), "检索顺序必须进入下游消息"

    retrieval_events = [e for e in services.recorder.events if e["event"] == "retrieval_docs"]
    ranks = [doc["rank"] for doc in retrieval_events[0]["docs"]]
    assert ranks == [1, 2, 3]
    assert [doc["id"] for doc in retrieval_events[0]["docs"]] == ["doc-a", "doc-b", "doc-c"]


def test_search_receipts_enter_report_messages(make_services, fixture_case, trial_output_dir):
    import json

    services = make_services()
    runner = Runner(config=services.config, services=services, root=ROOT_FOR_TESTS)
    outcome = runner.run_case(fixture_case, ROOT_FOR_TESTS / "trial/cases/case-001.png", trial_output_dir)
    assert outcome["success"]

    report_main = [r for r in services.calls.calls if r.role == "report_main"]
    payload = json.dumps([r.payload for r in report_main], ensure_ascii=False)
    assert "fixture.example/terms/湿热" in payload, "术语搜索返回必须进入报告输入"
    assert "fixture.example/images/黄连.jpg" in payload, "药材图片返回必须进入报告输入"
    assert "fixture.example/articles" in payload, "关键词搜索返回必须进入报告输入"


from pathlib import Path

ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
