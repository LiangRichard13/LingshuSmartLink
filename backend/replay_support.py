"""单病例试用内核：消息类型、严格配置、预算、录制器与 fixture/real 服务工厂。

本模块是轻量编排的唯一共享依赖：fixture 模式下不得导入 langchain、torch、
openai、modelscope 等重依赖，也不发起任何网络请求。真实服务全部延迟导入，
集中在 RealServices 工厂内。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import yaml

# ---------------------------------------------------------------------------
# 失败原因码与异常
# ---------------------------------------------------------------------------

REASON_CONFIG_INVALID = "config_invalid"
REASON_MODE_CONFLICT = "mode_conflict"
REASON_PATH_ESCAPE = "path_escape"
REASON_CASE_MISSING = "case_missing"
REASON_IMAGE_MISSING = "image_missing"
REASON_IMAGE_CORRUPT = "image_corrupt"
REASON_OUTPUT_DIR_NOT_EMPTY = "output_dir_not_empty"
REASON_FIXTURE_KEY_MISSING = "fixture_key_missing"
REASON_BUDGET_EXCEEDED = "budget_exceeded"
REASON_PLANNER_EXHAUSTED = "planner_exhausted"
REASON_INFORMER_EXHAUSTED = "informer_exhausted"
REASON_LINKS_PARSE_FAILED = "links_parse_failed"
REASON_SEARCH_FAILED = "search_failed"
REASON_CHAT_FAILURE = "chat_failure"
REASON_VISION_FAILURE = "vision_failure"
REASON_CLASSIFIER_FAILURE = "classifier_failure"
REASON_RETRIEVAL_FAILURE = "retrieval_failure"
REASON_REPORT_WRITE_FAILED = "report_write_failed"
REASON_ASSET_MISSING = "asset_missing"
REASON_UNEXPECTED = "unexpected_error"


class TrialError(Exception):
    """试用失败基类；reason_code 进入 result.json 的 failure 字段。"""

    reason_code = "trial_error"

    def __init__(self, message: str, reason_code: str | None = None) -> None:
        super().__init__(message)
        if reason_code is not None:
            self.reason_code = reason_code


class TrialConfigError(TrialError):
    reason_code = REASON_CONFIG_INVALID


class TrialModeError(TrialError):
    reason_code = REASON_MODE_CONFLICT


class TrialPathError(TrialError):
    reason_code = REASON_PATH_ESCAPE


class TrialInputError(TrialError):
    reason_code = REASON_CASE_MISSING


class TrialBudgetError(TrialError):
    reason_code = REASON_BUDGET_EXCEEDED


class TrialPlannerError(TrialError):
    reason_code = REASON_PLANNER_EXHAUSTED


class TrialInformerError(TrialError):
    reason_code = REASON_INFORMER_EXHAUSTED


class TrialLinksError(TrialError):
    reason_code = REASON_LINKS_PARSE_FAILED


class TrialFixtureError(TrialError):
    reason_code = REASON_FIXTURE_KEY_MISSING


# ---------------------------------------------------------------------------
# 消息与响应类型（不依赖 langchain；真实服务在工厂内转换）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def with_content(self, content: str) -> "Message":
        return replace(self, content=content)


def system_message(content: str) -> Message:
    return Message("system", content)


def user_message(content: str) -> Message:
    return Message("user", content)


def ai_message(content: str) -> Message:
    return Message("assistant", content)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: dict | None = None  # 真实 provider 提供时记录；fixture 为 None


@dataclass(frozen=True)
class ClassifierResult:
    label: str
    confidence: str  # 原链路中的 "置信度: 97.30%" 字符串


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    source: str
    content: str
    rank: int
    score: float


# ---------------------------------------------------------------------------
# 角色绑定 invoker：把已解析的 model/params 绑到客户端上，供各 Agent 注入
# ---------------------------------------------------------------------------


class RoleInvoker:
    """绑定一个 ChatClient 与单一角色的 model/params。"""

    def __init__(self, client: ChatClient, model: str, params: dict, role: str) -> None:
        self._client = client
        self.model = model
        self.params = params
        self.role = role

    def invoke(self, messages: list) -> LLMResponse:
        return self._client.invoke(self.model, messages, self.params)


class VisionInvoker:
    """绑定视觉模型的 create 调用；模型名只来自配置。"""

    def __init__(self, client: VisionClient, model: str, params: dict) -> None:
        self._client = client
        self.model = model
        self.params = params

    def create(self, messages: list) -> LLMResponse:
        return self._client.create(self.model, messages, self.params)


# ---------------------------------------------------------------------------
# 摘要工具
# ---------------------------------------------------------------------------


def digest_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return digest_text(payload)


def digest_messages(messages: Iterable[Message]) -> str:
    return digest_json([{"role": m.role, "content": m.content} for m in messages])


def digest_file(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


# ---------------------------------------------------------------------------
# 时钟
# ---------------------------------------------------------------------------


class Clock(Protocol):
    def now(self) -> datetime: ...


class FixedClock:
    """fixture 模式使用：所有时间观察值都是同一个固定时刻。"""

    def __init__(self, instant: datetime) -> None:
        self._instant = instant

    def now(self) -> datetime:
        return self._instant


class SystemClock:
    """local/live 模式使用：真实墙钟。"""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def format_report_time(moment: datetime) -> str:
    """原 ReportGenAgent 的报告时间格式。"""
    return moment.strftime("%Y年%m月%d日 %H时%M分%S秒")


# ---------------------------------------------------------------------------
# 请求预算：限额前检查，超限不发下一次请求
# ---------------------------------------------------------------------------


class ModelCallBudget:
    def __init__(self, model_call_limit: int) -> None:
        if model_call_limit <= 0:
            raise TrialConfigError("workflow.model_call_limit 必须为正整数")
        self.model_call_limit = model_call_limit
        self.used = 0

    def check(self, stage: str) -> None:
        if self.used >= self.model_call_limit:
            raise TrialBudgetError(
                f"模型调用次数已达上限 {self.model_call_limit}（阶段 {stage} 被拒绝，未发出请求）"
            )

    def consume(self, stage: str) -> None:
        self.check(stage)
        self.used += 1


# ---------------------------------------------------------------------------
# 严格配置读取
# ---------------------------------------------------------------------------

PROMPT_ROLES = (
    "vision",
    "actor",
    "instructor",
    "planner",
    "informer",
    "informer_wrapper",
    "summary",
    "links",
    "report_meta",
    "report_main",
)

_MODES = ("fixture", "local", "live")


@dataclass(frozen=True)
class RoleModelConfig:
    model: str
    params: dict
    base_url: str | None = None      # OpenAI 兼容 endpoint；缺省用该角色的历史 endpoint
    api_key_env: str | None = None   # 凭据的环境变量名（不是密钥值，可入库）


@dataclass(frozen=True)
class TrialConfig:
    schema_version: int
    mode: str
    case_id: str
    report_time: datetime
    chat_turn_limit: int
    planner_max_attempts: int
    model_call_limit: int
    prompts: dict
    primary: RoleModelConfig
    vision: RoleModelConfig
    generation: dict
    rag: dict
    classifier: dict
    search: dict
    retrieval: dict
    timeouts: dict
    source_path: str

    @property
    def report_time_text(self) -> str:
        return format_report_time(self.report_time)


def _require_keys(section: str, data: dict, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise TrialConfigError(f"配置段 {section} 缺少必需字段: {', '.join(missing)}")


def _reject_unknown(section: str, data: dict, allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise TrialConfigError(f"配置段 {section} 含未知字段: {', '.join(unknown)}")


def _require_positive_int(section: str, data: dict, key: str) -> int:
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrialConfigError(f"{section}.{key} 必须为正整数，实际为 {value!r}")
    return value


def parse_config(raw: dict, source_path: str = "<memory>") -> TrialConfig:
    if not isinstance(raw, dict):
        raise TrialConfigError("配置根节点必须是映射")
    _reject_unknown(
        "<root>",
        raw,
        (
            "schema_version",
            "trial",
            "workflow",
            "prompts",
            "models",
            "generation",
            "rag",
            "classifier",
            "search",
            "retrieval",
            "timeouts",
        ),
    )
    if raw.get("schema_version") != 1:
        raise TrialConfigError("schema_version 必须为 1")

    trial = raw.get("trial")
    if not isinstance(trial, dict):
        raise TrialConfigError("缺少 trial 配置段")
    _require_keys("trial", trial, ("mode", "case_id", "report_time"))
    _reject_unknown("trial", trial, ("mode", "case_id", "report_time"))
    mode = trial["mode"]
    if mode not in _MODES:
        raise TrialConfigError(f"trial.mode 必须是 {'/'.join(_MODES)}，实际为 {mode!r}")
    report_time_raw = trial["report_time"]
    if not isinstance(report_time_raw, str):
        raise TrialConfigError("trial.report_time 必须是 ISO 8601 字符串")
    try:
        report_time = datetime.fromisoformat(report_time_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrialConfigError(f"trial.report_time 无法解析: {exc}") from exc
    if report_time.tzinfo is None:
        report_time = report_time.replace(tzinfo=timezone.utc)

    workflow = raw.get("workflow")
    if not isinstance(workflow, dict):
        raise TrialConfigError("缺少 workflow 配置段")
    _require_keys("workflow", workflow, ("chat_turn_limit", "planner_max_attempts", "model_call_limit"))
    _reject_unknown("workflow", workflow, ("chat_turn_limit", "planner_max_attempts", "model_call_limit"))
    chat_turn_limit = _require_positive_int("workflow", workflow, "chat_turn_limit")
    planner_max_attempts = _require_positive_int("workflow", workflow, "planner_max_attempts")
    model_call_limit = _require_positive_int("workflow", workflow, "model_call_limit")

    prompts = raw.get("prompts")
    if not isinstance(prompts, dict):
        raise TrialConfigError("缺少 prompts 配置段")
    missing_roles = [role for role in PROMPT_ROLES if role not in prompts]
    if missing_roles:
        raise TrialConfigError(f"prompts 缺少角色: {', '.join(missing_roles)}")
    _reject_unknown("prompts", prompts, PROMPT_ROLES)
    for role, path in prompts.items():
        if not isinstance(path, str) or not path:
            raise TrialConfigError(f"prompts.{role} 必须是非空相对路径字符串")

    models = raw.get("models")
    if not isinstance(models, dict):
        raise TrialConfigError("缺少 models 配置段")
    _reject_unknown("models", models, ("primary", "vision"))

    def _role(section: str) -> RoleModelConfig:
        data = models.get(section)
        if not isinstance(data, dict):
            raise TrialConfigError(f"models.{section} 必须是映射")
        _require_keys(f"models.{section}", data, ("model",))
        _reject_unknown(f"models.{section}", data, ("model", "params", "base_url", "api_key_env"))
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise TrialConfigError(f"models.{section}.params 必须是映射")
        model_name = data["model"]
        if not isinstance(model_name, str) or not model_name:
            raise TrialConfigError(f"models.{section}.model 必须是非空字符串")
        base_url = data.get("base_url")
        api_key_env = data.get("api_key_env")
        for key, value in (("base_url", base_url), ("api_key_env", api_key_env)):
            if value is not None and (not isinstance(value, str) or not value):
                raise TrialConfigError(f"models.{section}.{key} 必须是非空字符串或省略")
        return RoleModelConfig(
            model=model_name, params=dict(params), base_url=base_url, api_key_env=api_key_env
        )

    primary = _role("primary")
    vision = _role("vision")

    generation = raw.get("generation")
    if not isinstance(generation, dict):
        raise TrialConfigError("缺少 generation 配置段")
    _require_keys("generation", generation, ("temperature", "top_p", "max_tokens"))
    _reject_unknown(
        "generation", generation, ("temperature", "top_p", "max_tokens", "seed")
    )
    for key in ("temperature", "top_p", "max_tokens"):
        value = generation[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TrialConfigError(f"generation.{key} 必须是数值")

    rag = raw.get("rag")
    if not isinstance(rag, dict):
        raise TrialConfigError("缺少 rag 配置段")
    _require_keys(
        "rag",
        rag,
        ("chunk_size", "chunk_overlap", "separators", "k", "corpus_manifest", "index_manifest"),
    )
    _reject_unknown(
        "rag",
        rag,
        ("chunk_size", "chunk_overlap", "separators", "k", "corpus_manifest", "index_manifest", "persist_directory", "embedding_model"),
    )
    _require_positive_int("rag", rag, "chunk_size")
    _require_positive_int("rag", rag, "k")
    if not isinstance(rag["chunk_overlap"], int) or isinstance(rag["chunk_overlap"], bool) or rag["chunk_overlap"] < 0:
        raise TrialConfigError("rag.chunk_overlap 必须是非负整数")
    if not isinstance(rag["separators"], list) or not all(isinstance(s, str) for s in rag["separators"]):
        raise TrialConfigError("rag.separators 必须是字符串列表")

    classifier = raw.get("classifier")
    if not isinstance(classifier, dict):
        raise TrialConfigError("缺少 classifier 配置段")
    _require_keys("classifier", classifier, ("mode",))
    _reject_unknown("classifier", classifier, ("mode", "checkpoint", "device"))
    if classifier["mode"] not in ("fixture", "local"):
        raise TrialConfigError("classifier.mode 必须是 fixture 或 local")

    search = raw.get("search")
    if not isinstance(search, dict):
        raise TrialConfigError("缺少 search 配置段")
    _require_keys("search", search, ("mode",))
    _reject_unknown("search", search, ("mode", "cassette", "timeout_seconds"))
    if search["mode"] not in ("fixture", "baidu_serper", "serper"):
        raise TrialConfigError(
            "search.mode 必须是 fixture、baidu_serper 或 serper"
            "（serper：原代码中备选的 Serper 网页搜索路径，baike.deno.dev 服务失效后经贡献者授权启用）"
        )

    retrieval = raw.get("retrieval")
    if not isinstance(retrieval, dict):
        raise TrialConfigError("缺少 retrieval 配置段")
    _reject_unknown("retrieval", retrieval, ("fixture",))

    timeouts = raw.get("timeouts")
    if not isinstance(timeouts, dict):
        raise TrialConfigError("缺少 timeouts 配置段")
    _require_keys("timeouts", timeouts, ("chat_seconds", "vision_seconds", "search_seconds"))
    _reject_unknown("timeouts", timeouts, ("chat_seconds", "vision_seconds", "search_seconds"))
    for key in ("chat_seconds", "vision_seconds", "search_seconds"):
        value = timeouts[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise TrialConfigError(f"timeouts.{key} 必须为正数")

    config = TrialConfig(
        schema_version=1,
        mode=mode,
        case_id=trial["case_id"],
        report_time=report_time,
        chat_turn_limit=chat_turn_limit,
        planner_max_attempts=planner_max_attempts,
        model_call_limit=model_call_limit,
        prompts=dict(prompts),
        primary=primary,
        vision=vision,
        generation=dict(generation),
        rag=dict(rag),
        classifier=dict(classifier),
        search=dict(search),
        retrieval=dict(retrieval),
        timeouts=dict(timeouts),
        source_path=source_path,
    )
    validate_mode_consistency(config)
    return config


def validate_mode_consistency(config: TrialConfig) -> None:
    """模式冲突检查：fixture 不得要求真实资产，local/live 不得留在 fixture 服务上。"""
    if config.mode == "fixture":
        if config.classifier["mode"] != "fixture":
            raise TrialModeError("fixture 模式要求 classifier.mode=fixture")
        if config.search["mode"] != "fixture":
            raise TrialModeError("fixture 模式要求 search.mode=fixture")
        if config.classifier.get("checkpoint") is not None:
            raise TrialModeError("fixture 模式不消费真实 checkpoint")
        if not config.retrieval.get("fixture"):
            raise TrialModeError("fixture 模式要求 retrieval.fixture 指向检索夹具文件")
    else:
        if config.classifier["mode"] != "local":
            raise TrialModeError(f"{config.mode} 模式要求 classifier.mode=local")
        if not config.classifier.get("checkpoint"):
            raise TrialModeError(f"{config.mode} 模式要求 classifier.checkpoint 指向真实权重")
        if not config.rag.get("persist_directory"):
            raise TrialModeError(f"{config.mode} 模式要求 rag.persist_directory 指向已构建索引")
        if not config.rag.get("embedding_model"):
            raise TrialModeError(f"{config.mode} 模式要求 rag.embedding_model 指向本地 GTE 模型目录")
        if config.retrieval.get("fixture"):
            raise TrialModeError(f"{config.mode} 模式不得声明 retrieval.fixture")


def load_config(path: Path) -> TrialConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrialConfigError(f"无法读取配置文件 {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TrialConfigError(f"配置文件 {path} 不是合法 YAML: {exc}") from exc
    return parse_config(raw, source_path=str(path))


def load_prompt(root: Path, rel_path: str) -> str:
    path = resolve_under_root(root, rel_path, label=f"prompt {rel_path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrialConfigError(f"无法读取 prompt {rel_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# 路径安全：一切输入/输出必须落在仓库根内
# ---------------------------------------------------------------------------


def resolve_under_root(root: Path, rel_path: str, *, label: str = "path") -> Path:
    candidate = Path(rel_path)
    if candidate.is_absolute():
        raise TrialPathError(f"{label} 必须是仓库内相对路径，收到绝对路径 {rel_path!r}")
    parts = candidate.parts
    if any(part == ".." for part in parts):
        raise TrialPathError(f"{label} 不得包含 '..' 段: {rel_path!r}")
    if not parts:
        raise TrialPathError(f"{label} 不能为空")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise TrialPathError(f"{label} 越出仓库根: {rel_path!r}")
    return resolved


# ---------------------------------------------------------------------------
# 病例读取
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    schema_version: int
    case_id: str
    synthetic: bool
    image: str
    messages: list
    report_time: str
    source_path: str


def load_case(path: Path) -> CaseSpec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrialInputError(f"无法读取病例文件 {path}: {exc}", REASON_CASE_MISSING) from exc
    except json.JSONDecodeError as exc:
        raise TrialInputError(f"病例文件 {path} 不是合法 JSON: {exc}", REASON_CASE_MISSING) from exc
    required = ("schema_version", "case_id", "synthetic", "image", "messages", "report_time")
    missing = [key for key in required if key not in raw]
    if missing:
        raise TrialInputError(f"病例缺少字段: {', '.join(missing)}", REASON_CASE_MISSING)
    if raw["schema_version"] != 1:
        raise TrialInputError("病例 schema_version 必须为 1", REASON_CASE_MISSING)
    if not isinstance(raw["synthetic"], bool) or not raw["synthetic"]:
        raise TrialInputError("病例必须标记 synthetic:true", REASON_CASE_MISSING)
    if not isinstance(raw["messages"], list) or not raw["messages"]:
        raise TrialInputError("病例 messages 必须是非空列表", REASON_CASE_MISSING)
    for item in raw["messages"]:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            raise TrialInputError("病例 messages 每项必须是含 content 字符串的对象", REASON_CASE_MISSING)
    return CaseSpec(
        schema_version=1,
        case_id=raw["case_id"],
        synthetic=raw["synthetic"],
        image=raw["image"],
        messages=list(raw["messages"]),
        report_time=raw["report_time"],
        source_path=str(path),
    )


def validate_image(image_path: Path) -> None:
    """图片必须存在且是合法 PNG/JPEG 字节，不能是 LFS pointer 或文本占位。"""
    if not image_path.is_file():
        raise TrialInputError(f"图片文件不存在: {image_path}", REASON_IMAGE_MISSING)
    data = image_path.read_bytes()
    if data.startswith(b"version https://git-lfs.github.com/spec/"):
        raise TrialInputError(f"图片是 LFS pointer，不是真实字节: {image_path}", REASON_IMAGE_CORRUPT)
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    if not (is_png or is_jpeg):
        raise TrialInputError(f"图片不是合法 PNG/JPEG 字节: {image_path}", REASON_IMAGE_CORRUPT)


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise TrialInputError(f"输出路径已存在且不是目录: {output_dir}", REASON_OUTPUT_DIR_NOT_EMPTY)
        existing = [p for p in output_dir.iterdir() if p.name != ".gitkeep"]
        if existing:
            names = ", ".join(sorted(p.name for p in existing)[:5])
            raise TrialInputError(
                f"输出目录非空，拒绝覆盖: {output_dir}（{names}）",
                REASON_OUTPUT_DIR_NOT_EMPTY,
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 服务协议
# ---------------------------------------------------------------------------


class ChatClient(Protocol):
    def invoke(self, model: str, messages: list, params: dict) -> LLMResponse: ...


class VisionClient(Protocol):
    """舌苔视觉诊断客户端；messages 为 openai chat 原始结构（含 image_url）。"""

    def create(self, model: str, messages: list, params: dict) -> LLMResponse: ...


class ClassifierBackend(Protocol):
    def predict(self, image_path: Path) -> ClassifierResult: ...


class RetrieverBackend(Protocol):
    def retrieve(self, query: str) -> list: ...


class SearchBackend(Protocol):
    def search_terms(self, items: list) -> list: ...
    def search_herbs(self, items: list) -> list: ...
    def search_acupoints(self, items: list) -> list: ...
    def search_herb_images(self, items: list) -> list: ...
    def search_keyword(self, keyword: str) -> list: ...


# ---------------------------------------------------------------------------
# 运行上下文与录制器（trace.jsonl / consumed.json 的事实来源）
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    stage: str = "init"
    task_index: int | None = None
    turn: int | None = None
    parent: str | None = None
    #: consultation 阶段内真正发起 chat 请求的角色（actor/instructor/summary 等）
    chat_role: str | None = None

    @property
    def role(self) -> str:
        return self.chat_role or self.stage


class TraceRecorder:
    """按序追加事件；fixture 事件时间来自注入时钟，保证字节可复现。"""

    def __init__(self, mode: str, clock: Clock) -> None:
        self.mode = mode
        self.clock = clock
        self.events: list = []
        self._sequence = 0

    def record(self, context: RunContext, **fields: Any) -> str:
        self._sequence += 1
        event_id = f"e{self._sequence:04d}"
        event = {
            "schema_version": 1,
            "sequence": self._sequence,
            "event_id": event_id,
            "time": self.clock.now().isoformat(),
            "stage": context.stage,
            "task": context.task_index,
            "turn": context.turn,
            "parent": context.parent,
            "source": self.mode,
        }
        event.update({key: value for key, value in fields.items() if value is not None})
        self.events.append(event)
        return event_id

    def write(self, path: Path) -> None:
        lines = [json.dumps(event, sort_keys=True, ensure_ascii=False) for event in self.events]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@dataclass
class CallRecord:
    kind: str  # chat | vision | search | retrieval
    stage: str
    role: str | None
    model: str | None
    params: dict | None
    request_digest: str | None
    response_digest: str | None
    payload: dict = field(default_factory=dict)


class CallRecorder:
    """记录真正发生的调用（模型、参数、消息摘要、次数），供 consumed.json 与测试 spy 使用。"""

    def __init__(self) -> None:
        self.calls: list = []

    def add(self, record: CallRecord) -> CallRecord:
        self.calls.append(record)
        return record

    def call_counts(self) -> dict:
        counts: dict = {}
        for call in self.calls:
            counts[call.stage] = counts.get(call.stage, 0) + 1
        return counts


class RecordingChatClient:
    """包装任意 ChatClient：预算检查 → 请求事件 → 调用 → 响应/错误事件。"""

    def __init__(
        self,
        inner: ChatClient,
        budget: ModelCallBudget,
        recorder: TraceRecorder,
        calls: CallRecorder,
        context: RunContext,
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._recorder = recorder
        self._calls = calls
        self._context = context

    def invoke(self, model: str, messages: list, params: dict) -> LLMResponse:
        stage = self._context.stage
        role = self._context.role
        self._budget.consume(stage)
        request_digest = digest_messages(messages)
        parent = self._recorder.record(
            self._context,
            event="llm_request",
            role=role,
            model=model,
            params=params,
            request_digest=request_digest,
            n_messages=len(messages),
        )
        record = self._calls.add(
            CallRecord(
                kind="chat",
                stage=stage,
                role=role,
                model=model,
                params=dict(params),
                request_digest=request_digest,
                response_digest=None,
                payload={"messages": [dict(role=m.role, content=m.content) for m in messages]},
            )
        )
        try:
            response = self._inner.invoke(model, messages, params)
        except Exception as exc:
            self._recorder.record(
                self._context, event="llm_error", parent=parent, reason=str(exc)
            )
            raise
        response_digest = digest_text(response.content)
        self._recorder.record(
            self._context,
            event="llm_response",
            parent=parent,
            model=response.model,
            response_digest=response_digest,
        )
        record.response_digest = response_digest
        return response


class RecordingVisionClient:
    def __init__(
        self,
        inner: VisionClient,
        budget: ModelCallBudget,
        recorder: TraceRecorder,
        calls: CallRecorder,
        context: RunContext,
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._recorder = recorder
        self._calls = calls
        self._context = context

    def create(self, model: str, messages: list, params: dict) -> LLMResponse:
        stage = self._context.stage
        self._budget.consume(stage)
        request_digest = digest_json(messages)
        parent = self._recorder.record(
            self._context,
            event="vision_request",
            role="vision",
            model=model,
            params=params,
            request_digest=request_digest,
            n_messages=len(messages),
        )
        self._calls.add(
            CallRecord(
                kind="vision",
                stage=stage,
                role="vision",
                model=model,
                params=dict(params),
                request_digest=request_digest,
                response_digest=None,
            )
        )
        try:
            response = self._inner.create(model, messages, params)
        except Exception as exc:
            self._recorder.record(
                self._context, event="vision_error", parent=parent, reason=str(exc)
            )
            raise
        response_digest = digest_text(response.content)
        self._recorder.record(
            self._context,
            event="vision_response",
            parent=parent,
            model=response.model,
            response_digest=response_digest,
        )
        return response


class RecordingSearchBackend:
    def __init__(
        self,
        inner: SearchBackend,
        recorder: TraceRecorder,
        calls: CallRecorder,
        context: RunContext,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._calls = calls
        self._context = context

    def _record(self, action: str, query: Any, result: Any) -> Any:
        result_digest = digest_json(result)
        self._recorder.record(
            self._context,
            event=f"search_{action}",
            request_digest=digest_json(query),
            response_digest=result_digest,
        )
        self._calls.add(
            CallRecord(
                kind="search",
                stage=self._context.stage,
                role=action,
                model=None,
                params=None,
                request_digest=digest_json(query),
                response_digest=result_digest,
            )
        )
        return result

    def search_terms(self, items):
        return self._record("terms", items, self._inner.search_terms(items))

    def search_herbs(self, items):
        return self._record("herbs", items, self._inner.search_herbs(items))

    def search_acupoints(self, items):
        return self._record("acupoints", items, self._inner.search_acupoints(items))

    def search_herb_images(self, items):
        return self._record("herb_images", items, self._inner.search_herb_images(items))

    def search_keyword(self, keyword):
        return self._record("keyword", keyword, self._inner.search_keyword(keyword))


# ---------------------------------------------------------------------------
# fixture 服务：严格按键消费人工构造的确定响应，缺失即失败，不回退
# ---------------------------------------------------------------------------


class FixtureChatClient:
    """chat 响应按 (角色, 序号) 从 cassette 顺序消费；角色由编排上下文给出，越界即失败。"""

    def __init__(self, scripted: dict, context: RunContext) -> None:
        self._scripted = scripted
        self._cursor: dict = {}
        self._context = context

    def invoke(self, model: str, messages: list, params: dict) -> LLMResponse:
        role = self._context.role
        entries = self._scripted.get(role)
        if entries is None:
            raise TrialFixtureError(f"cassette 缺少 chat 角色 {role!r} 的脚本列表")
        if not isinstance(entries, list):
            raise TrialFixtureError(
                f"cassette chat 角色 {role!r} 的脚本必须是响应列表，实际为 {type(entries).__name__}"
            )
        index = self._cursor.get(role, 0)
        if index >= len(entries):
            raise TrialFixtureError(
                f"cassette 角色 {role!r} 的脚本已耗尽（需要第 {index + 1} 条响应）"
            )
        self._cursor[role] = index + 1
        entry = entries[index]
        if not isinstance(entry, str):
            raise TrialFixtureError(f"cassette 角色 {role!r} 第 {index + 1} 条响应必须是字符串")
        return LLMResponse(content=entry, model=model)

    @property
    def remaining(self) -> dict:
        result = {}
        for role, entries in self._scripted.items():
            if isinstance(entries, list):
                result[role] = len(entries) - self._cursor.get(role, 0)
        return result

    def consumption_summary(self) -> dict:
        """每个 fixture 角色已消费的响应数（进入 consumed.json）。"""
        return {
            "responses_consumed_per_role": {
                role: self._cursor.get(role, 0) for role in sorted(self._scripted)
            }
        }


class FixtureVisionClient:
    """视觉请求按真实结构构造（含图片 data URL），返回 cassette 固定诊断文本。

    请求摘要真实反映输入图片字节，fixture 不声称真实视觉模型改变了响应。
    """

    def __init__(self, scripted_response: str) -> None:
        self._scripted_response = scripted_response

    def create(self, model: str, messages: list, params: dict) -> LLMResponse:
        return LLMResponse(content=self._scripted_response, model=model)


class FixtureClassifier:
    """按 cassette 固定标签返回；输出格式与原分类器一致（带前缀字符串）。"""

    def __init__(self, label: str, confidence: str) -> None:
        if not confidence.endswith("%"):
            confidence = f"{confidence}%"
        self._result = ClassifierResult(label=f"预测类别: {label}", confidence=f"置信度: {confidence}")

    def predict(self, image_path: Path) -> ClassifierResult:
        return self._result


class FixtureRetriever:
    """按 cassette 顺序返回固定文档集合；rank 由位置决定，score 来自脚本。"""

    def __init__(self, docs: list) -> None:
        self._docs = [
            DocRecord(
                doc_id=str(doc["id"]),
                source=str(doc.get("source", "")),
                content=str(doc["content"]),
                rank=position + 1,
                score=float(doc.get("score", 0.0)),
            )
            for position, doc in enumerate(docs)
        ]

    def retrieve(self, query: str) -> list:
        return list(self._docs)

    @property
    def docs(self) -> list:
        return list(self._docs)


class FixtureSearchBackend:
    """搜索按 cassette 固定映射返回；缺 key 即失败。"""

    def __init__(self, scripted: dict) -> None:
        self._scripted = scripted

    def _entries(self, key: str) -> list:
        entries = self._scripted.get(key)
        if not isinstance(entries, list):
            raise TrialFixtureError(f"cassette 缺少 search.{key} 条目")
        return entries

    def search_terms(self, items):
        return self._format(self._entries("terms"), items, "item_search_result")

    def search_herbs(self, items):
        return self._format(self._entries("herbs"), items, "item_search_result")

    def search_acupoints(self, items):
        return self._format(self._entries("acupoints"), items, "item_search_result")

    def search_herb_images(self, items):
        return self._format(self._entries("herb_images"), items, "item_image_search_result")

    def search_keyword(self, keyword):
        entries = self._entries("keyword")
        results = [{"title": e["title"], "link": e["link"]} for e in entries]
        return [{"item": keyword, "item_search_result": results}]

    @staticmethod
    def _format(entries, items, result_key):
        by_item = {entry["item"]: entry[result_key] for entry in entries}
        formatted = []
        for item in items:
            if item not in by_item:
                raise TrialFixtureError(f"cassette search 条目缺少 {item!r} 的返回")
            formatted.append({"item": item, result_key: by_item[item]})
        return formatted


# ---------------------------------------------------------------------------
# RealServices：真实组件，全部延迟导入；local/live 才允许构造
# ---------------------------------------------------------------------------


class RealServicesError(TrialError):
    reason_code = REASON_ASSET_MISSING


def _import(name: str, hint: str) -> Any:
    try:
        return __import__(name, fromlist=["*"])
    except ImportError as exc:
        raise RealServicesError(
            f"{config_mode_hint()}: 需要 {hint}（导入 {name} 失败: {exc}）。"
            "fixture 模式不需要这些依赖；local/live 请使用 .venv-live 真实环境。"
        ) from exc


def config_mode_hint() -> str:
    return "local/live 模式"


class RealChatClient:
    """langchain_openai ChatOpenAI 包装；模型与参数来自配置。"""

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float) -> None:
        langchain_openai = _import("langchain_openai", "langchain-openai")
        self._client = langchain_openai.ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def invoke(self, model: str, messages: list, params: dict) -> LLMResponse:
        from langchain.schema import AIMessage as _AIMessage

        client = self._client.bind(model=model, **params)
        langchain_messages = [_to_langchain(m) for m in messages]
        result = client.invoke(langchain_messages)
        assert isinstance(result, _AIMessage)
        # 计划 5.3：真实 provider 上报的 model 与 usage 若可得则保存；不可用写 None
        metadata = getattr(result, "response_metadata", {}) or {}
        reported_model = metadata.get("model_name") or metadata.get("model")
        usage = metadata.get("token_usage")
        return LLMResponse(
            content=result.content,
            model=reported_model if isinstance(reported_model, str) else model,
            usage=dict(usage) if isinstance(usage, dict) else None,
        )


def _to_langchain(message: Message):
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(content=message.content)
    return HumanMessage(content=message.content)


class RealVisionClient:
    """StepFun openai 兼容视觉客户端；模型名只来自配置。"""

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float) -> None:
        openai_module = _import("openai", "openai")
        self._client = openai_module.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0
        )

    def create(self, model: str, messages: list, params: dict) -> LLMResponse:
        completion = self._client.chat.completions.create(
            model=model, messages=messages, **params
        )
        usage = None
        if getattr(completion, "usage", None) is not None:
            usage = completion.usage.model_dump() if hasattr(completion.usage, "model_dump") else dict(completion.usage)
        reported_model = getattr(completion, "model", None)
        return LLMResponse(
            content=completion.choices[0].message.content,
            model=reported_model if isinstance(reported_model, str) else model,
            usage=usage,
        )


class RealClassifier:
    """真实 ResNet18 分类器适配器；实现唯一来源于 tongueCoatingClassifier 模块。"""

    def __init__(self, checkpoint: str, device: str | None = None) -> None:
        classifier_module = _import(
            "agents.tongueCoatingAgent.tongueCoatingClassifier", "舌苔分类器实现"
        )
        try:
            self._impl = classifier_module.TongueCoatingClassifier(
                model_path=checkpoint, device=device
            )
        except Exception as exc:
            raise RealServicesError(
                f"分类权重加载失败: {exc}。请核对该 .pth 的来源与网络结构。",
                REASON_ASSET_MISSING,
            ) from exc

    def predict(self, image_path: Path) -> ClassifierResult:
        try:
            label, confidence = self._impl.predict_image(str(image_path))
        except TrialError:
            raise
        except Exception as exc:
            raise TrialError(f"分类器预测失败: {exc}", REASON_CLASSIFIER_FAILURE) from exc
        return ClassifierResult(label=label, confidence=confidence)


class RealRetriever:
    """Chroma + GTE 检索；k 显式来自配置；返回文档身份/来源/排名。"""

    def __init__(self, persist_directory: str, embedding_model: Any, k: int) -> None:
        chroma_module = _import("langchain_chroma", "langchain-chroma")
        self._db = chroma_module.Chroma(
            persist_directory=persist_directory, embedding_function=embedding_model
        )
        self._k = k

    def retrieve(self, query: str) -> list:
        docs_with_scores = self._db.similarity_search_with_score(query, k=self._k)
        records = []
        for rank, (doc, score) in enumerate(docs_with_scores, start=1):
            source = str(doc.metadata.get("source", ""))
            start_index = doc.metadata.get("start_index")
            doc_id = f"{source}#{start_index}" if start_index is not None else source
            records.append(
                DocRecord(
                    doc_id=doc_id,
                    source=source,
                    content=doc.page_content,
                    rank=rank,
                    score=float(score),
                )
            )
        return records


class RealSearchBackend:
    """词条搜索（百度百科镜像或 Serper 网页搜索）+ Serper 图片/关键词搜索；超时来自配置。

    baike.deno.dev（社区百科镜像）2026-11-06 起直连/代理均不可达（服务失效）；
    search.mode=serper 时词条类搜索改走 Serper 网页搜索——对应原代码
    linksAddAgent 中已存在但被注释的 serper_search 备选路径。
    """

    def __init__(
        self,
        baidu_api: str,
        serper_api_key: str,
        timeout_seconds: float,
        terms_transport: str = "baidu",
    ) -> None:
        self._baidu_api = baidu_api
        self._serper_api_key = serper_api_key
        self._timeout = timeout_seconds
        self._terms_transport = terms_transport

    def _serper_web(self, items: list) -> list:
        payload = [{"q": item, "gl": "cn", "hl": "zh-cn"} for item in items]
        data, _ = self._serper(payload, "/search", "organic")
        formatted = []
        for item, entry in zip(items, data):
            results = [
                {"title": one.get("title", ""), "link": one.get("link", "")}
                for one in entry.get("organic", [])[:3]
            ]
            formatted.append({"item": item, "item_search_result": results})
        return formatted

    def _baidu(self, items: list) -> list:
        import urllib.parse

        requests = _import("requests", "requests")
        formatted = []
        for item in items:
            url = f"{self._baidu_api}/item/{urllib.parse.quote(item)}"
            response = requests.get(url, timeout=self._timeout)
            data = response.json()
            if data.get("status") == 200:
                formatted.append({"item": item, "item_search_result": data["data"]["link"]})
        return formatted

    def _serper(self, payload: list, endpoint: str, result_key: str) -> list:
        import http.client
        import json as json_module

        conn = http.client.HTTPSConnection("google.serper.dev", timeout=self._timeout)
        headers = {"X-API-KEY": self._serper_api_key, "Content-Type": "application/json"}
        conn.request("POST", endpoint, json_module.dumps(payload), headers)
        response = conn.getresponse()
        data = json_module.loads(response.read().decode("utf-8"))
        return data, result_key

    def search_terms(self, items):
        if self._terms_transport == "serper":
            return self._serper_web(items)
        return self._baidu(items)

    def search_herbs(self, items):
        if self._terms_transport == "serper":
            return self._serper_web(items)
        return self._baidu(items)

    def search_acupoints(self, items):
        if self._terms_transport == "serper":
            return self._serper_web(items)
        return self._baidu(items)

    def search_herb_images(self, items):
        payload = [{"q": item, "gl": "cn", "hl": "zh-cn"} for item in items]
        data, _ = self._serper(payload, "/images", "images")
        formatted = []
        for item, entry in zip(items, data):
            formatted.append(
                {"item": item, "item_image_search_result": entry["images"][0]["imageUrl"]}
            )
        return formatted

    def search_keyword(self, keyword):
        data, _ = self._serper([{"q": keyword, "gl": "cn", "hl": "zh-cn"}], "/search", "organic")
        results = [
            {"title": entry.get("title", ""), "link": entry.get("link", "")}
            for entry in data["organic"][:5]
        ]
        return [{"item": keyword, "item_search_result": results}]


# ---------------------------------------------------------------------------
# 服务束
# ---------------------------------------------------------------------------


@dataclass
class ServiceBundle:
    config: TrialConfig
    chat: ChatClient
    vision: VisionClient
    classifier: ClassifierBackend
    retriever: RetrieverBackend
    search: SearchBackend
    clock: Clock
    budget: ModelCallBudget
    recorder: TraceRecorder
    calls: CallRecorder
    context: RunContext

    @property
    def recorded_chat(self) -> RecordingChatClient:
        return RecordingChatClient(
            self.chat, self.budget, self.recorder, self.calls, self.context
        )

    @property
    def recorded_vision(self) -> RecordingVisionClient:
        return RecordingVisionClient(
            self.vision, self.budget, self.recorder, self.calls, self.context
        )

    @property
    def recorded_search(self) -> RecordingSearchBackend:
        return RecordingSearchBackend(self.search, self.recorder, self.calls, self.context)


def build_fixture_services(
    config: TrialConfig,
    root: Path,
    cassette_path: Path,
) -> ServiceBundle:
    """fixture 模式服务工厂：无网络、无重依赖导入、无静默回退。"""
    try:
        cassette = json.loads(cassette_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrialFixtureError(f"无法读取 cassette {cassette_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TrialFixtureError(f"cassette {cassette_path} 不是合法 JSON: {exc}") from exc
    if not isinstance(cassette, dict) or cassette.get("schema_version") != 1:
        raise TrialFixtureError("cassette schema_version 必须为 1")
    if not cassette.get("synthetic"):
        raise TrialFixtureError("cassette 必须标记 synthetic:true（人工构造响应）")

    vision_block = cassette.get("vision")
    if not isinstance(vision_block, dict) or not isinstance(vision_block.get("response"), str):
        raise TrialFixtureError("cassette 缺少 vision.response 固定诊断文本")
    classification = cassette.get("classification")
    if not isinstance(classification, dict) or "label" not in classification or "confidence" not in classification:
        raise TrialFixtureError("cassette 缺少 classification.label/confidence")
    chat_scripted = cassette.get("chat")
    if not isinstance(chat_scripted, dict):
        raise TrialFixtureError("cassette 缺少 chat 脚本映射")
    search_scripted = cassette.get("search")
    if not isinstance(search_scripted, dict):
        raise TrialFixtureError("cassette 缺少 search 映射")

    retrieval_path = config.retrieval.get("fixture")
    assert retrieval_path, "fixture 配置必须带 retrieval.fixture"
    retrieval_file = resolve_under_root(root, retrieval_path, label="retrieval.fixture")
    try:
        retrieval_raw = json.loads(retrieval_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrialFixtureError(f"无法读取检索夹具 {retrieval_file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TrialFixtureError(f"检索夹具 {retrieval_file} 不是合法 JSON: {exc}") from exc
    docs = retrieval_raw.get("docs") if isinstance(retrieval_raw, dict) else None
    if not isinstance(docs, list) or not docs:
        raise TrialFixtureError("检索夹具必须包含非空 docs 列表")

    clock = FixedClock(config.report_time)
    recorder = TraceRecorder(mode="fixture", clock=clock)
    budget = ModelCallBudget(config.model_call_limit)
    context = RunContext()
    chat = FixtureChatClient(chat_scripted, context)

    return ServiceBundle(
        config=config,
        chat=chat,
        vision=FixtureVisionClient(vision_block["response"]),
        classifier=FixtureClassifier(classification["label"], classification["confidence"]),
        retriever=FixtureRetriever(docs),
        search=FixtureSearchBackend(search_scripted),
        clock=clock,
        budget=budget,
        recorder=recorder,
        calls=CallRecorder(),
        context=context,
    )


def build_real_services(config: TrialConfig, root: Path) -> ServiceBundle:
    """local/live 服务工厂；重依赖在此延迟导入，失败给出明确原因。

    计划 5.2 的模式边界：local = 真实分类/embedding/RAG + 注入的固定
    LLM/搜索响应（cassette）；live = 全部真实 provider 与搜索。
    """
    checkpoint = config.classifier.get("checkpoint")
    assert checkpoint, "local/live 配置必须带 classifier.checkpoint"
    checkpoint_path = resolve_under_root(root, checkpoint, label="classifier.checkpoint")
    if not checkpoint_path.is_file():
        raise RealServicesError(
            f"分类权重不存在: {checkpoint}。local 模式不得用合成权重替代。",
            REASON_ASSET_MISSING,
        )
    persist_directory = config.rag.get("persist_directory")
    assert persist_directory, "local/live 配置必须带 rag.persist_directory"
    persist_path = resolve_under_root(root, persist_directory, label="rag.persist_directory")
    if not persist_path.is_dir():
        raise RealServicesError(
            f"向量索引目录不存在: {persist_directory}。请先用构建脚本生成索引。",
            REASON_ASSET_MISSING,
        )

    if config.mode == "local":
        if config.search.get("mode") != "fixture" or not config.search.get("cassette"):
            raise RealServicesError(
                "local 模式的 LLM/搜索为注入响应，要求 search.mode=fixture 且提供 cassette",
                REASON_MODE_CONFLICT,
            )
    else:
        if config.search.get("mode") not in ("baidu_serper", "serper"):
            raise RealServicesError(
                "live 模式要求 search.mode=baidu_serper 或 serper（真实搜索）",
                REASON_MODE_CONFLICT,
            )

    deepseek_key = os.getenv("DEEP_SEEK_API_KEY", "")
    stepfun_key = os.getenv("STEP_FUN_API_KEY", "")
    serper_key = os.getenv("SERPER_API_KEY", "")
    if config.mode == "live":
        missing = [
            name
            for name, value in (
                ("DEEP_SEEK_API_KEY", deepseek_key),
                ("STEP_FUN_API_KEY", stepfun_key),
                ("SERPER_API_KEY", serper_key),
            )
            if not value
        ]
        if missing:
            raise RealServicesError(
                f"live 模式缺少环境变量: {', '.join(missing)}", REASON_ASSET_MISSING
            )

    gte_path = config.rag.get("embedding_model")
    if not gte_path:
        raise RealServicesError(
            f"{config.mode} 模式要求 rag.embedding_model 指向本地 GTE 模型目录",
            REASON_ASSET_MISSING,
        )
    gte_dir = resolve_under_root(root, gte_path, label="rag.embedding_model")
    if not gte_dir.is_dir():
        raise RealServicesError(
            f"GTE 模型目录不存在: {gte_path}", REASON_ASSET_MISSING
        )
    gte_module = _import("model.embedding_model.gteEmbeddings", "GTE embedding 加载器")
    embeddings = gte_module.GTEEmbeddings(model_path=str(gte_dir))

    clock = SystemClock()
    recorder = TraceRecorder(mode=config.mode, clock=clock)
    budget = ModelCallBudget(config.model_call_limit)
    context = RunContext()
    if config.mode == "local":
        cassette_path = resolve_under_root(
            root, config.search["cassette"], label="search.cassette"
        )
        cassette = json.loads(cassette_path.read_text(encoding="utf-8"))
        chat = FixtureChatClient(cassette["chat"], context)
        vision = FixtureVisionClient(cassette["vision"]["response"])
        search = FixtureSearchBackend(cassette["search"])
    else:
        chat = RealChatClient(
            base_url=config.primary.base_url or "https://api.deepseek.com",
            api_key=deepseek_key,
            timeout_seconds=config.timeouts["chat_seconds"],
        )
        vision_base_url = config.vision.base_url or "https://api.stepfun.com/v1"
        vision_key_env = config.vision.api_key_env or "STEP_FUN_API_KEY"
        vision_key = os.getenv(vision_key_env, "")
        if not vision_key:
            raise RealServicesError(
                f"live 模式视觉角色要求环境变量 {vision_key_env}（models.vision.api_key_env）",
                REASON_ASSET_MISSING,
            )
        vision = RealVisionClient(
            base_url=vision_base_url,
            api_key=vision_key,
            timeout_seconds=config.timeouts["vision_seconds"],
        )
        search = RealSearchBackend(
            baidu_api="https://baike.deno.dev",
            serper_api_key=serper_key,
            timeout_seconds=config.timeouts["search_seconds"],
            terms_transport=(
                "serper" if config.search.get("mode") == "serper" else "baidu"
            ),
        )
    classifier = RealClassifier(
        checkpoint=str(checkpoint_path), device=config.classifier.get("device", "cpu")
    )
    retriever = RealRetriever(
        persist_directory=str(persist_path), embedding_model=embeddings, k=config.rag["k"]
    )
    return ServiceBundle(
        config=config,
        chat=chat,
        vision=vision,
        classifier=classifier,
        retriever=retriever,
        search=search,
        clock=clock,
        budget=budget,
        recorder=recorder,
        calls=CallRecorder(),
        context=context,
    )
