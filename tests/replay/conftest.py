"""tests/replay 公共夹具：backend 导入路径、仓库内临时输出目录助手。"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROOT_FOR_TESTS = ROOT  # 测试模块统一使用该名称
BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from replay_support import (  # noqa: E402
    FixtureChatClient,
    FixtureClassifier,
    FixtureRetriever,
    FixtureSearchBackend,
    FixtureVisionClient,
    FixedClock,
    CallRecorder,
    RunContext,
    TraceRecorder,
    TrialConfig,
    build_fixture_services,
    load_case,
    load_config,
)


@pytest.fixture()
def repo_root() -> Path:
    return ROOT


@pytest.fixture()
def trial_output_dir() -> Path:
    """仓库内临时输出目录（replay_case 只接受根内路径），测试后清理。"""
    output = ROOT / "trial" / "outputs" / f"pytest-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    yield output
    for item in output.iterdir():
        if item.is_file():
            item.unlink()
        else:
            __import__("shutil").rmtree(item)
    output.rmdir()


@pytest.fixture()
def fixture_config() -> TrialConfig:
    return load_config(ROOT / "trial" / "configs" / "current.yaml")


@pytest.fixture()
def fixture_case():
    return load_case(ROOT / "trial" / "cases" / "case-001.json")


@pytest.fixture()
def make_services(fixture_config, repo_root):
    """以仓库 trial fixtures 构造 fixture 服务束；可覆盖 cassette/检索/预算。"""

    def _make(
        config: TrialConfig | None = None,
        cassette: dict | None = None,
        retrieval_docs: list | None = None,
        search_override=None,
        chat_override=None,
    ):
        cfg = config or fixture_config
        if cassette is None:
            cassette_path = repo_root / cfg.search["cassette"]
            import json

            cassette = json.loads(cassette_path.read_text(encoding="utf-8"))
        if retrieval_docs is None:
            import json

            retrieval_file = repo_root / cfg.retrieval["fixture"]
            retrieval_docs = json.loads(retrieval_file.read_text(encoding="utf-8"))["docs"]
        services = build_fixture_services(cfg, repo_root, repo_root / cfg.search["cassette"])
        if cassette is not None or retrieval_docs is not None:
            # 重建：直接以注入的脚本构造，避免读盘
            clock = FixedClock(cfg.report_time)
            recorder = TraceRecorder(mode="fixture", clock=clock)
            from replay_support import ModelCallBudget

            ctx = RunContext()
            chat = chat_override or FixtureChatClient(cassette["chat"], ctx)
            services = type(services)(
                config=cfg,
                chat=chat,
                vision=FixtureVisionClient(cassette["vision"]["response"]),
                classifier=FixtureClassifier(
                    cassette["classification"]["label"],
                    cassette["classification"]["confidence"],
                ),
                retriever=FixtureRetriever(retrieval_docs),
                search=search_override or FixtureSearchBackend(cassette["search"]),
                clock=clock,
                budget=ModelCallBudget(cfg.model_call_limit),
                recorder=recorder,
                calls=CallRecorder(),
                context=ctx,
            )
        return services

    return _make
