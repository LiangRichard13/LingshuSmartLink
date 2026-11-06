"""Lingshu 试用 Runner：非交互 run_case 与交互 run_chain 共用同一业务编排。

改造要点（LS01）：
- Runner 构造器接收 config、services（ServiceBundle）与仓库根 root；
- run_case(case, image_path, output_dir) 以病例消息驱动 Informer，无 input()；
- run_chain() 仅做交互适配（input() 读图片路径与用户轮次），随后调用
  同一编排实现，不再另造一条业务链。
原语义保持：分类/视觉 → Informer → RAG 检索 → 规划 → Actor/Instructor
协作 → Summary → 链接搜索 → 报告；Actor 跨任务保留历史；Instructor 在
每任务结束（含超限）后重置；协作轮数为每任务 Instructor 调用上限，
初次 Actor 调用不计入该数。与原实现的有意差异单独记录在试用文档：
- 分类从视觉 Agent 中拆为独立阶段，便于回执记录（调用顺序不变）；
- 交互 Informer 以 exit 结束时按明确失败处理（原实现返回 None 并在
  后续崩溃）。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.consultationAgents.consultation_config import (
    actor_role_name,
    instructor_role_name,
)
from agents.consultationAgents.consultationAgents import (
    ActorAgent,
    InstructorAgent,
    SummaryAgent,
)
from agents.consultationAgents.systemMessageGen import SystemMessageGenerator
from agents.informerAgent.informerAgent import InformerAgent
from agents.planAgent.planAgent import PlanAgent
from agents.planAgent.plan_config import treatment_plan
from agents.reportGenAgents.linksAddAgent import LinksAddAgent
from agents.reportGenAgents.reporGenAgent import ReportGenAgent
from agents.tongueCoatingAgent.tongueCoatingAgent import TongueCoatingAgent
from replay_support import (
    REASON_CLASSIFIER_FAILURE,
    REASON_PLANNER_EXHAUSTED,
    REASON_REPORT_WRITE_FAILED,
    REASON_RETRIEVAL_FAILURE,
    REASON_SEARCH_FAILED,
    REASON_UNEXPECTED,
    REASON_VISION_FAILURE,
    CallRecord,
    CaseSpec,
    Message,
    RoleInvoker,
    ServiceBundle,
    TrialConfig,
    TrialError,
    TrialInformerError,
    VisionInvoker,
    digest_file,
    digest_json,
    digest_text,
    load_prompt,
    resolve_under_root,
    system_message,
    user_message,
)
from tools.colorPrinter import ColorPrinter
from tools.markdownTool import markdownTool

STAGE_ORDER = (
    "classification",
    "vision",
    "informer",
    "retrieval",
    "planning",
    "consultation",
    "links",
    "report",
)

INFORMER_DONE_MARKER = "<INFORMER_TASK_DONE>"
INSTRUCTOR_DONE_MARKER = "<CAMEL_TASK_DONE>"


class Runner:
    def __init__(self, config: TrialConfig, services: ServiceBundle, root: Path) -> None:
        self.config = config
        self.services = services
        self.root = root
        self._prompts = {
            role: load_prompt(root, rel_path) for role, rel_path in config.prompts.items()
        }
        self._chat_client = services.recorded_chat
        self._vision_client = services.recorded_vision
        self._search_client = services.recorded_search

    # ------------------------------------------------------------------
    # invoker 工厂
    # ------------------------------------------------------------------

    def _chat_invoker(self, role: str) -> RoleInvoker:
        params = {**self.config.generation, **self.config.primary.params}
        return RoleInvoker(self._chat_client, self.config.primary.model, params, role)

    def _vision_invoker(self) -> VisionInvoker:
        return VisionInvoker(
            self._vision_client, self.config.vision.model, self.config.vision.params
        )

    # ------------------------------------------------------------------
    # 非交互入口（LS01 验收）
    # ------------------------------------------------------------------

    def run_case(self, case: CaseSpec, image_path: Path, output_dir: Path) -> dict:
        """非交互单病例执行：按原链路顺序跑完八个阶段并写出四个产物文件。"""
        started = time.monotonic()
        stages = {stage: {"status": "skipped"} for stage in STAGE_ORDER}
        failure_box: dict = {}
        report_path: str | None = None
        chain: dict = {}

        def mark(stage: str) -> None:
            stages[stage] = {"status": "success"}

        def fail(stage: str, exc: TrialError) -> None:
            stages[stage] = {"status": "failed", "reason_code": exc.reason_code}
            failure_box.clear()
            failure_box.update(
                {"stage": stage, "reason_code": exc.reason_code, "message": str(exc)}
            )

        try:
            chain["diagnosis"] = self._stage_classification_and_vision(image_path, mark, fail)
            chain["background"] = self._stage_informer(
                chain["diagnosis"],
                case_messages=[m["content"] for m in case.messages],
                mark=mark,
                fail=fail,
            )
            chain["rag"] = self._stage_retrieval(chain["background"], mark, fail)
            chain["plan"] = self._stage_planning(chain["background"], mark, fail)
            chain["consultation"] = self._stage_consultation(
                chain["background"], chain["rag"], chain["plan"], mark, fail
            )
            chain["search"] = self._stage_links(chain["consultation"], mark, fail)
            report_path = self._stage_report(
                image_path,
                chain["diagnosis"],
                chain["background"],
                chain["consultation"],
                chain["search"],
                output_dir,
                mark,
                fail,
            )
        except TrialError:
            pass  # 阶段处理器已记录失败与原因码
        except Exception as exc:  # 未预期异常也必须有失败回执
            fail(
                self.services.context.stage,
                TrialError(f"未预期异常: {exc!r}", REASON_UNEXPECTED),
            )

        failure = failure_box or None
        self._finalize(case, image_path, output_dir, stages, failure, report_path, started)
        return {
            "success": failure is None,
            "failure": failure,
            "stages": stages,
            "report_path": report_path,
        }

    # ------------------------------------------------------------------
    # 交互入口：仅做适配，调用同一编排
    # ------------------------------------------------------------------

    def run_chain(self) -> str:
        stages: dict = {}
        mark = self._interactive_mark()
        fail = self._interactive_fail()

        ColorPrinter.green("TongueCoatingAgent:")
        ColorPrinter.white("请输入舌苔图片路径:")
        image_path = Path(input(ColorPrinter.color_text("User: ", "blue")).strip())

        diagnosis = self._stage_classification_and_vision(image_path, mark, fail)
        ColorPrinter.green("TongueCoatingAgent:")
        ColorPrinter.white(diagnosis)

        background = self._stage_informer(
            diagnosis, case_messages=None, mark=mark, fail=fail, interactive=True
        )
        ColorPrinter.red("SystemMessage:")
        ColorPrinter.yellow(f"PatientBackgroundInformation:{background}")

        rag_information = self._stage_retrieval(background, mark, fail)
        ColorPrinter.red("SystemMessage:")
        ColorPrinter.yellow(f"已检索到以下相关资料:{rag_information}")

        plan_tasks = self._stage_planning(background, mark, fail)
        ColorPrinter.green("PlanAgent:")
        ColorPrinter.white(str(plan_tasks))

        consultation_result = self._stage_consultation(
            background, rag_information, plan_tasks, mark, fail, verbose=True
        )
        search_reference = self._stage_links(consultation_result, mark, fail)
        ColorPrinter.green("LinksAddAgent:")
        ColorPrinter.white(str(search_reference))

        report_path = self._stage_report(
            image_path,
            diagnosis,
            background,
            consultation_result,
            search_reference,
            output_dir=None,
            mark=mark,
            fail=fail,
            verbose=True,
        )
        return report_path

    def _interactive_mark(self):
        def mark(stage: str) -> None:
            ColorPrinter.red("SystemMessage:")
            ColorPrinter.yellow(f"阶段完成:{stage}")

        return mark

    def _interactive_fail(self):
        def fail(stage: str, exc: TrialError) -> None:
            ColorPrinter.red("SystemMessage:")
            ColorPrinter.yellow(f"阶段失败:{stage}: {exc}")
            raise exc

        return fail

    # ------------------------------------------------------------------
    # 各阶段实现（两条入口共用）
    # ------------------------------------------------------------------

    def _stage_classification_and_vision(self, image_path: Path, mark, fail) -> str:
        context = self.services.context
        context.stage = "classification"
        try:
            classifier_result = self.services.classifier.predict(image_path)
        except TrialError as exc:
            fail("classification", exc)
            raise
        except Exception as exc:
            wrapped = TrialError(f"分类失败: {exc}", REASON_CLASSIFIER_FAILURE)
            fail("classification", wrapped)
            raise wrapped from exc
        self.services.recorder.record(
            context,
            event="classification_result",
            response_digest=digest_json(
                {"label": classifier_result.label, "confidence": classifier_result.confidence}
            ),
        )
        self.services.calls.add(
            CallRecord(
                kind="classifier",
                stage="classification",
                role="classifier",
                model=None,
                params=None,
                request_digest=digest_text(str(image_path)),
                response_digest=digest_json(
                    {"label": classifier_result.label, "confidence": classifier_result.confidence}
                ),
            )
        )
        mark("classification")

        context.stage = "vision"
        try:
            agent = TongueCoatingAgent(
                vision_invoker=self._vision_invoker(),
                system_prompt=self._prompts["vision"],
            )
            diagnosis = agent.llm_call(
                tongue_coating_result=classifier_result.label,
                confidence_score=classifier_result.confidence,
                image_path=image_path,
            )
        except TrialError as exc:
            fail("vision", exc)
            raise
        except Exception as exc:
            wrapped = TrialError(f"视觉诊断失败: {exc}", REASON_VISION_FAILURE)
            fail("vision", wrapped)
            raise wrapped from exc
        mark("vision")
        return diagnosis

    def _stage_informer(
        self,
        diagnosis: str,
        case_messages: list | None,
        mark,
        fail,
        interactive: bool = False,
    ) -> str:
        context = self.services.context
        context.stage = "informer"
        context.chat_role = "informer"
        informer = InformerAgent(
            system_message=Message("system", self._prompts["informer"]),
            invoker=self._chat_invoker("informer"),
        )
        wrapper_content = self._prompts["informer_wrapper"].format(diagnosis=diagnosis)
        informer.history.append(Message("system", wrapper_content))

        turn_index = 0
        background: str | None = None
        if interactive:
            turn_source = self._interactive_turns()
        else:
            turn_source = iter(case_messages)
        try:
            for turn_content in turn_source:
                turn_index += 1
                context.turn = turn_index
                response = informer.get_response(user_message(turn_content))
                if interactive:
                    ColorPrinter.green("InformerAgent: ")
                    ColorPrinter.white(response.content)
                if INFORMER_DONE_MARKER in response.content:
                    background = response.content
                    break
            if background is None:
                raise TrialInformerError(
                    "输入消息已耗尽但 Informer 未给出 <INFORMER_TASK_DONE> 结束标记"
                )
        except TrialError as exc:
            fail("informer", exc)
            raise
        self.services.recorder.record(
            context, event="informer_done", reason="task_done", turn=turn_index
        )
        mark("informer")
        context.chat_role = None
        return background

    def _interactive_turns(self):
        while True:
            user_input = input(ColorPrinter.color_text("User: ", "blue"))
            if user_input == "exit":
                return
            yield user_input

    def _stage_retrieval(self, background: str, mark, fail) -> list:
        context = self.services.context
        context.stage = "retrieval"
        try:
            docs = self.services.retriever.retrieve(background)
        except TrialError as exc:
            fail("retrieval", exc)
            raise
        except Exception as exc:
            wrapped = TrialError(f"检索失败: {exc}", REASON_RETRIEVAL_FAILURE)
            fail("retrieval", wrapped)
            raise wrapped from exc
        doc_view = [
            {"id": d.doc_id, "source": d.source, "rank": d.rank, "score": d.score} for d in docs
        ]
        self.services.recorder.record(
            context,
            event="retrieval_docs",
            query_digest=digest_text(background),
            docs=doc_view,
        )
        self.services.calls.add(
            CallRecord(
                kind="retrieval",
                stage="retrieval",
                role="retriever",
                model=None,
                params={"k": self.config.rag["k"]},
                request_digest=digest_text(background),
                response_digest=digest_json(doc_view),
            )
        )
        mark("retrieval")
        return [d.content for d in docs]

    def _stage_planning(self, background: str, mark, fail) -> list:
        context = self.services.context
        context.stage = "planning"
        context.chat_role = "planner"
        plan_agent = PlanAgent(
            patient_background_information=background,
            invoker=self._chat_invoker("planner"),
            system_prompt_template=self._prompts["planner"],
            treatment_plan=treatment_plan,
            max_attempts=self.config.planner_max_attempts,
        )
        try:
            plan_tasks = plan_agent.get_plan()
        except TrialError as exc:
            fail("planning", exc)
            raise
        self.services.recorder.record(
            context, event="planning_done", tasks=list(plan_tasks)
        )
        context.chat_role = None
        mark("planning")
        return plan_tasks

    def _stage_consultation(
        self,
        background: str,
        rag_information: list,
        plan_tasks: list,
        mark,
        fail,
        verbose: bool = False,
    ) -> list:
        context = self.services.context
        context.stage = "consultation"
        generator = SystemMessageGenerator(
            actor_template=self._prompts["actor"],
            instructor_template=self._prompts["instructor"],
            summary_template=self._prompts["summary"],
            actor_role_name=actor_role_name,
            instructor_role_name=instructor_role_name,
        )
        actor = ActorAgent(background, self._chat_invoker("actor"), generator)
        actor.stored_messages.append(
            system_message(
                f"以下是参考资料:\n{rag_information},如果在执行任务过程中用到了以上资料，请引用参考资料原文"
            )
        )
        instructor = InstructorAgent(background, self._chat_invoker("instructor"), generator)

        summary_content_list: list = []
        task_reasons: list = []
        try:
            for task_index, task in enumerate(plan_tasks):
                context.task_index = task_index
                if verbose:
                    ColorPrinter.red("SystemMessage:")
                    ColorPrinter.yellow(f"开始任务:{task}")
                dialogue: list = []
                context.chat_role = "actor"
                actor_msg = actor.step(user_message(f"现在开始我们的任务:{task}"))
                dialogue.append({"role": "actor", "content": actor_msg.content})
                if verbose:
                    ColorPrinter.green("ActorAgent: ")
                    ColorPrinter.white(actor_msg.content)

                n = 0
                reason = "turn_limit"
                while n < self.config.chat_turn_limit:
                    context.chat_role = "instructor"
                    context.turn = n
                    instructor_msg = instructor.step(
                        user_message(f"这是我的任务:{task}\n\n这是我的内容{actor_msg.content}\n\n")
                    )
                    dialogue.append({"role": "instructor", "content": instructor_msg.content})
                    if verbose:
                        ColorPrinter.green("InstructorAgent: ")
                        ColorPrinter.white(instructor_msg.content)
                    if INSTRUCTOR_DONE_MARKER in instructor_msg.content:
                        reason = "task_done"
                        break
                    context.chat_role = "actor"
                    actor_msg = actor.step(user_message(instructor_msg.content))
                    dialogue.append({"role": "actor", "content": actor_msg.content})
                    if verbose:
                        ColorPrinter.green("ActorAgent: ")
                        ColorPrinter.white(actor_msg.content)
                    n += 1

                context.chat_role = "summary"
                instructor.init_messages()
                summary_agent = SummaryAgent(
                    dialogue, task, self._chat_invoker("summary"), generator
                )
                summary_content = summary_agent.step(user_message("请输出:")).content
                summary_content_list.append(summary_content)
                task_reasons.append({"task_index": task_index, "turn": n, "reason": reason})
                if verbose:
                    ColorPrinter.green("SummaryAgent: ")
                    ColorPrinter.white(summary_content)
        except TrialError as exc:
            fail("consultation", exc)
            raise
        self.services.recorder.record(context, event="consultation_done", tasks=task_reasons)
        context.task_index = None
        context.turn = None
        context.chat_role = None
        mark("consultation")
        return [{"task": t, "answer": a} for t, a in zip(plan_tasks, summary_content_list)]

    def _stage_links(self, consultation_result: list, mark, fail) -> dict:
        context = self.services.context
        context.stage = "links"
        context.chat_role = "links"
        links_agent = LinksAddAgent(
            invoker=self._chat_invoker("links"),
            system_message_template=self._prompts["links"],
            search=self._search_client,
        )
        main_content = ""
        for content in consultation_result:
            main_content += content["answer"] + "\n"
        try:
            search_reference = links_agent.add_links(main_content)
        except TrialError as exc:
            fail("links", exc)
            raise
        except Exception as exc:
            wrapped = TrialError(f"搜索失败: {exc}", REASON_SEARCH_FAILED)
            fail("links", wrapped)
            raise wrapped from exc
        context.chat_role = None
        mark("links")
        return search_reference

    def _stage_report(
        self,
        image_path: Path,
        diagnosis: str,
        background: str,
        consultation_result: list,
        search_reference: dict,
        output_dir: Path | None,
        mark,
        fail,
        verbose: bool = False,
    ) -> str:
        context = self.services.context
        context.stage = "report"
        if output_dir is not None:
            tool = markdownTool(output_dir=str(output_dir), now=self.services.clock.now())

            def saver(content: str) -> str:
                return tool.save_markdown(content, "report.md")

        else:
            tool = markdownTool(now=self.services.clock.now())
            saver = tool.save_markdown

        report_agent = ReportGenAgent(
            meta_invoker=self._chat_invoker("report_meta"),
            main_invoker=self._chat_invoker("report_main"),
            meta_template=self._prompts["report_meta"],
            main_template=self._prompts["report_main"],
            report_time_text=self.config.report_time_text,
            saver=saver,
        )
        try:
            context.chat_role = "report_meta"
            report_agent.init_meta_data_system_message(
                patient_background_information=background,
                tongue_coating_image_path=str(image_path),
                tongue_coating_diagnosis=diagnosis,
            )
            report_content = report_agent.get_content()
            if verbose:
                ColorPrinter.green("ReportGenAgent:")
                ColorPrinter.white(report_content)
            for index, content in enumerate(consultation_result):
                context.chat_role = "report_main"
                report_agent.init_main_content_system_message(main_content=content)
                report_agent.add_reference_to_history(index, search_reference)
                new_report_content = report_agent.get_content()
                if verbose:
                    ColorPrinter.green("ReportGenAgent:")
                    ColorPrinter.white(new_report_content)
                report_content = report_content + "\n\n" + new_report_content
            context.chat_role = None
            save_path = report_agent.save_report(report_content)
        except TrialError as exc:
            fail("report", exc)
            raise
        saved = Path(save_path)
        if not saved.is_file():
            wrapped = TrialError(f"报告保存失败: {save_path}", REASON_REPORT_WRITE_FAILED)
            fail("report", wrapped)
            raise wrapped
        self.services.recorder.record(context, event="report_saved", path=saved.name)
        mark("report")
        return str(saved)

    # ------------------------------------------------------------------
    # 产物写盘
    # ------------------------------------------------------------------

    def _finalize(self, case, image_path, output_dir, stages, failure, report_path, started) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.services.recorder.write(output_dir / "trace.jsonl")
        consumed = self._build_consumed(case, image_path)
        (output_dir / "consumed.json").write_text(
            json.dumps(consumed, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        duration = time.monotonic() - started
        result = {
            "schema_version": 1,
            "case_id": self.config.case_id,
            "mode": self.config.mode,
            "success": failure is None,
            "stages": stages,
            "report_path": Path(report_path).name if report_path else None,
            "failure": failure,
            "duration_seconds": round(duration, 3),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _build_consumed(self, case, image_path) -> dict:
        config = self.config
        calls = self.services.calls
        by_stage: dict = {}
        kind_counts: dict = {}
        requests = []
        for call in calls.calls:
            by_stage[call.stage] = by_stage.get(call.stage, 0) + 1
            kind_counts[call.kind] = kind_counts.get(call.kind, 0) + 1
            requests.append(
                {
                    "kind": call.kind,
                    "stage": call.stage,
                    "role": call.role,
                    "model": call.model,
                    "params": call.params,
                    "request_digest": call.request_digest,
                    "response_digest": call.response_digest,
                }
            )
        prompts_section = {}
        for role, rel_path in config.prompts.items():
            prompts_section[role] = digest_file(
                resolve_under_root(self.root, rel_path, label=f"prompt {role}")
            )
        inputs = {
            "case": digest_file(Path(case.source_path)),
            "image": digest_file(image_path),
            "prompts": prompts_section,
        }
        if config.mode == "fixture":
            cassette = resolve_under_root(
                self.root, config.search["cassette"], label="search.cassette"
            )
            inputs["cassette"] = digest_file(cassette)
            retrieval_file = resolve_under_root(
                self.root, config.retrieval["fixture"], label="retrieval.fixture"
            )
            inputs["retrieval_fixture"] = digest_file(retrieval_file)
        environment_path = self.root / "trial" / "environment.json"
        if environment_path.is_file():
            inputs["environment"] = digest_file(environment_path)

        consumed = {
            "schema_version": 1,
            "case_id": config.case_id,
            "mode": config.mode,
            "report_time_used": config.report_time.isoformat(),
            "config_source": config.source_path,
            "effective": {
                "workflow": {
                    "chat_turn_limit": config.chat_turn_limit,
                    "planner_max_attempts": config.planner_max_attempts,
                    "model_call_limit": config.model_call_limit,
                },
                "generation": config.generation,
                "models": {
                    "primary": {"model": config.primary.model, "params": config.primary.params},
                    "vision": {"model": config.vision.model, "params": config.vision.params},
                },
                "rag": {
                    key: config.rag.get(key)
                    for key in (
                        "chunk_size",
                        "chunk_overlap",
                        "separators",
                        "k",
                        "corpus_manifest",
                        "index_manifest",
                        "persist_directory",
                        "embedding_model",
                    )
                },
                "classifier": config.classifier,
                "search": config.search,
                "timeouts": config.timeouts,
                "prompts": dict(config.prompts),
            },
            "inputs": inputs,
            "requests": requests,
            "call_counts": {
                "by_stage": by_stage,
                "by_kind": kind_counts,
                "model_total": kind_counts.get("chat", 0) + kind_counts.get("vision", 0),
            },
            "budget": {
                "model_call_limit": config.model_call_limit,
                "used": self.services.budget.used,
            },
        }
        chat_client = self.services.chat
        if hasattr(chat_client, "consumption_summary"):
            consumed["fixture_consumption"] = chat_client.consumption_summary()
        return consumed
