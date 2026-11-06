#!/usr/bin/env python
"""Lingshu 单病例试用助手（LS09）：seed / prepare / env / verify / archive / probe-pth / package。

这是试用者自己的下游工具，不新增 ReproLLM 子命令；只写本计划约定的
trial 文件与显式传入的证据目录。prepare 不是 ReproLLM 的自动修复功能，
而是把试用者选定的实验输入（变体资产 + current.yaml + reprollm.yaml）
确定性落盘。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "trial"

SCENARIO_VARIANTS = (
    "baseline",
    "rounds-3",
    "prompt-v2",
    "image-v2",
    "retrieval-reordered",
    "search-v2",
)

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

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def digest_entry(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.as_posix(), "sha256": sha256_bytes(data), "size_bytes": len(data)}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_text_if_absent(path: Path, content: str) -> None:
    if path.exists():
        raise SystemExit(f"seed 拒绝覆盖已存在文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_bytes_if_absent(path: Path, data: bytes) -> None:
    if path.exists():
        raise SystemExit(f"seed 拒绝覆盖已存在文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def make_png(width: int, height: int, seed: int) -> bytes:
    """确定性纯 Python PNG 生成器（真彩色 8 位，无外部依赖）。"""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: none
        for x in range(width):
            raw.extend(
                bytes(
                    (
                        (x * 4 + seed) % 256,
                        (y * 4 + seed * 2) % 256,
                        ((x + y) * 2 + seed * 3) % 256,
                    )
                )
            )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# seed：合成病例、语料、fixtures、模板迁移
# ---------------------------------------------------------------------------


def case_messages() -> list:
    return [
        {"role": "user", "content": "你好，我最近肚子不太舒服，想咨询一下。"},
        {"role": "user", "content": "间歇性下腹胀痛三天了，饭后发作，持续一小时，还有点腹泻。"},
        {"role": "user", "content": "24岁男性学生，无慢性病史，偶尔吃辛辣，作息基本规律。"},
    ]


def chat_scripts() -> dict:
    tasks = [f"任务{i + 1}" for i in range(4)]
    actor, instructor = [], []
    for task_index, task in enumerate(tasks, start=1):
        actor.append(f"这是任务的完整解决方案：{task}初次方案（fixture 合成内容）。下一个请求。")
        for turn in range(1, 6):  # chat_turn_limit=5 时每任务 5 轮 instructor
            instructor.append(
                f"指令：完善{task}第{turn}轮内容（fixture 合成指令）。 输入：无 。"
            )
            actor.append(
                f"这是任务的完整解决方案：{task}第{turn}轮修订（fixture 合成内容）。下一个请求。"
            )
    return {
        "informer": [
            "您好，请问症状出现的具体部位在哪里？",
            "请描述症状的持续时间与发作规律，是否伴随腹泻？",
            (
                "患者为24岁男性学生，主诉间歇性下腹胀痛3天，饭后发作持续1小时，伴糊状腹泻，"
                "无慢性病史，饮食偶辛辣，作息规律，舌象黑苔提示脾胃湿热可能。<INFORMER_TASK_DONE>"
            ),
        ],
        "planner": [
            "<TASK_SPLIT>".join(
                [
                    "1.给出针对下腹胀痛的诊断与辨证结论（fixture 合成任务）",
                    "2.给出基础调整方案（fixture 合成任务）",
                    "3.给出中医治疗方案（fixture 合成任务）",
                    "4.给出需警惕的预警（fixture 合成任务）",
                ]
            )
        ],
        "actor": actor,
        "instructor": instructor,
        "summary": [
            f"这是任务的经过修改后的完整解决方案：任务{i}的合成总结（fixture）。"
            for i in range(1, 5)
        ],
        "links": [
            json.dumps(
                {
                    "terms": ["湿热", "脾胃湿热"],
                    "herbs": ["黄连"],
                    "acupoints": ["足三里"],
                    "keyword": "如何调理脾胃湿热",
                },
                ensure_ascii=False,
            )
        ],
        "report_meta": [
            "# 诊断报告（fixture 合成 meta 部分）\n\n## 患者基本信息\n\n## 舌象描述\n"
        ],
        "report_main": [
            f"## 报告第{i}部分（fixture 合成）\n\n内容要点{i}。" for i in range(1, 5)
        ],
    }


def search_map(marker: str = "") -> dict:
    suffix = marker or "base"
    return {
        "terms": [
            {"item": "湿热", "item_search_result": f"https://fixture.example/terms/湿热?rev={suffix}"},
            {
                "item": "脾胃湿热",
                "item_search_result": f"https://fixture.example/terms/脾胃湿热?rev={suffix}",
            },
        ],
        "herbs": [
            {"item": "黄连", "item_search_result": f"https://fixture.example/herbs/黄连?rev={suffix}"}
        ],
        "acupoints": [
            {
                "item": "足三里",
                "item_search_result": f"https://fixture.example/acupoints/足三里?rev={suffix}",
            }
        ],
        "herb_images": [
            {
                "item": "黄连",
                "item_image_search_result": f"https://fixture.example/images/黄连.jpg?rev={suffix}",
            }
        ],
        "keyword": [
            {
                "title": f"脾胃湿热的饮食调理（fixture 搜索结果{suffix}）",
                "link": "https://fixture.example/articles/1",
            },
            {"title": "合成养生文章二", "link": "https://fixture.example/articles/2"},
            {"title": "合成养生文章三", "link": "https://fixture.example/articles/3"},
            {"title": "合成养生文章四", "link": "https://fixture.example/articles/4"},
            {"title": "合成养生文章五", "link": "https://fixture.example/articles/5"},
        ],
    }


def cassette(marker: str = "") -> dict:
    return {
        "schema_version": 1,
        "synthetic": True,
        "note": "人工构造的确定响应（synthetic scripted），不是历史录制；不声称真实模型输出。",
        "classification": {"label": "黑苔", "confidence": "97.30"},
        "vision": {
            "response": "（fixture 合成诊断）舌象显示黑苔，舌质偏红，苔厚均匀分布，考虑脾胃湿热可能，建议结合面脉诊综合判断，本段不构成医疗建议。"
        },
        "chat": chat_scripts(),
        "search": search_map(marker),
    }


RETRIEVAL_DOCS = [
    {
        "id": "doc-a",
        "source": "trial/corpus/texts/synthetic-01.txt",
        "content": "脾胃湿热的成因与调理原则（合成语料A）：湿邪困脾，运化失司，宜健脾利湿。",
        "score": 0.91,
    },
    {
        "id": "doc-b",
        "source": "trial/corpus/texts/synthetic-02.txt",
        "content": "脾主运化的证候要点（合成语料B）：腹胀便溏，苔厚腻，饮食宜清淡，忌辛辣肥甘。",
        "score": 0.87,
    },
    {
        "id": "doc-c",
        "source": "trial/corpus/texts/synthetic-03.txt",
        "content": "针灸取穴与食疗示例（合成语料C）：足三里、中脘为主穴，薏苡仁粥辅助。",
        "score": 0.83,
    },
]


def cmd_seed(args) -> int:
    base = TRIAL / "cases"
    write_bytes_if_absent(base / "case-001-base.png", make_png(48, 48, seed=1))
    write_bytes_if_absent(base / "case-001-v2.png", make_png(48, 48, seed=2))
    write_text_if_absent(
        base / "case-001.json",
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "case-001",
                "synthetic": True,
                "image": "trial/cases/case-001.png",
                "messages": case_messages(),
                "report_time": "2026-11-05T00:00:00Z",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    corpus = TRIAL / "corpus" / "texts"
    corpus_files = {
        "synthetic-01.txt": "脾胃湿热证治要点（合成语料A）\n湿邪困脾，运化失司。<目录>调理宜健脾利湿，饮食清淡，忌辛辣肥甘。<目录>参考条目：薏苡仁、冬瓜利湿食材。",
        "synthetic-02.txt": "脾主运化的证候要点（合成语料B）\n腹胀便溏，苔厚腻。<目录>治法：健脾化湿，配穴足三里、中脘。<目录>食疗：薏苡仁粥为基础方。",
        "synthetic-03.txt": "针灸与食疗示例（合成语料C）\n主穴足三里、中脘，配穴阴陵泉。<目录>平补平泻，留针二十分钟。<目录>湿热重者加赤小豆。",
    }
    for name, text in corpus_files.items():
        write_text_if_absent(corpus / name, text + "\n")
    write_text_if_absent(
        TRIAL / "corpus" / "manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "synthetic": True,
                "files": [
                    digest_entry(TRIAL / "corpus" / "texts" / name)
                    for name in sorted(corpus_files)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    fixtures = TRIAL / "fixtures"
    write_text_if_absent(
        fixtures / "external-base.json",
        json.dumps(cassette(), ensure_ascii=False, indent=2) + "\n",
    )
    write_text_if_absent(
        fixtures / "external-search-v2.json",
        json.dumps(cassette(marker="search-v2"), ensure_ascii=False, indent=2) + "\n",
    )
    write_text_if_absent(
        fixtures / "retrieval-base.json",
        json.dumps(
            {"schema_version": 1, "synthetic": True, "docs": RETRIEVAL_DOCS},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    write_text_if_absent(
        fixtures / "retrieval-reordered.json",
        json.dumps(
            {
                "schema_version": 1,
                "synthetic": True,
                "docs": [RETRIEVAL_DOCS[1], RETRIEVAL_DOCS[0], RETRIEVAL_DOCS[2]],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    write_text_if_absent(
        fixtures / "index-manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "kind": "fixture-retrieval-descriptor",
                "chunk_size": 200,
                "chunk_overlap": 20,
                "separators": ["<目录>"],
                "k": 2,
                "docs": 3,
                "note": "fixture 模式的固定检索集合描述；真实 Chroma 索引由构建脚本生成 build-manifest.json。",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    # 激活路径初始化为 baseline 资产
    for active, source in (
        (fixtures / "external.json", fixtures / "external-base.json"),
        (fixtures / "retrieval.json", fixtures / "retrieval-base.json"),
    ):
        if not active.exists():
            active.write_bytes(source.read_bytes())

    write_prompt_files()
    write_legacy_live_config()
    print("seed 完成：trial/ 下的合成病例、语料、fixtures、prompts、legacy-live 配置已生成。")
    return 0


def write_prompt_files() -> None:
    prompts = TRIAL / "prompts"
    templates = {
        "vision.txt": VISION_PROMPT,
        "actor.txt": ACTOR_PROMPT,
        "actor-v2.txt": ACTOR_PROMPT + "\n[prompt-v2 测试标记：请在本轮回答末尾附上 <PROMPT_V2_MARKER>。]\n",
        "instructor.txt": INSTRUCTOR_PROMPT,
        "summary.txt": SUMMARY_PROMPT,
        "planner.txt": PLANNER_PROMPT,
        "informer.txt": INFORMER_PROMPT,
        "informer_wrapper.txt": INFORMER_WRAPPER_PROMPT,
        "links.txt": LINKS_PROMPT,
        "report_meta.txt": REPORT_META_PROMPT,
        "report_main.txt": REPORT_MAIN_PROMPT,
    }
    for name, text in templates.items():
        write_text_if_absent(prompts / name, text)
    write_text_if_absent(prompts / "actor-base.txt", ACTOR_PROMPT)


def write_legacy_live_config() -> None:
    config = {
        "schema_version": 1,
        "trial": {
            "mode": "live",
            "case_id": "case-001",
            "report_time": "2026-11-05T00:00:00Z",
        },
        "workflow": {"chat_turn_limit": 5, "planner_max_attempts": 2, "model_call_limit": 80},
        "prompts": {role: f"trial/prompts/{role}.txt" for role in PROMPT_ROLES},
        "models": {
            "primary": {
                "model": "deepseek-chat",
                "params": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
            },
            "vision": {
                "model": "step-1v-8k",
                "params": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 4095},
            },
        },
        "generation": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
        "rag": {
            "chunk_size": 200,
            "chunk_overlap": 20,
            "separators": ["<目录>"],
            "k": 2,
            "corpus_manifest": "trial/corpus/manifest.json",
            "index_manifest": "trial/assets-local/chroma-200/build-manifest.json",
            "persist_directory": "trial/assets-local/chroma-200",
            "embedding_model": "trial/assets-local/gte",
        },
        "classifier": {
            "mode": "local",
            "checkpoint": "trial/assets-local/classifier/tongue_coating_resnet18.pth",
            "device": "cpu",
        },
        "search": {"mode": "baidu_serper", "timeout_seconds": 30},
        "retrieval": {},
        "timeouts": {"chat_seconds": 60, "vision_seconds": 60, "search_seconds": 30},
    }
    write_text_if_absent(
        TRIAL / "configs" / "legacy-live.yaml",
        "# legacy 交互链路（python backend/agentsRunner.py）的真实模式默认配置。\n"
        "# generation 值为本试用新选定的请求设置（DeepSeek 文档默认附近），不宣称原代码原先显式使用这些值。\n"
        + yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
    )


# ---------------------------------------------------------------------------
# 原始模板迁移文件（与原 *_config.py 中文本逐字一致；单一运行时来源）
# ---------------------------------------------------------------------------

VISION_PROMPT = """            你是一个舌苔诊断专家，你的任务是帮助用户诊断舌苔问题。\n            你需要根据用户提供的舌苔病理图片、舌苔分类结果和置信度来进行综合诊断。\n            请描述舌苔的特征（如舌质、舌苔颜色厚薄等）和可能的健康问题。\n            不要给出任何医疗建议或治疗方案。\n            """

ACTOR_PROMPT = """
永远不要忘记你是{actor_role_name},我是{instructor_role_name},永远不要角色互换！\n
以下是患者的病历信息:{patient_background_information}\n
永远不要指示我！
我们在高质量完成任务方面有共同的兴趣。
你必须帮助我完成我所给出的修改指令。
永远不要忘记我们的任务！
我必须根据你的专业知识和我的需求来指示你完成任务。
我每次只能给你一个指令。
请注意！！！！:你每次不能简单只回复我想要你修改的部分，而是给出修改后符合任务要求的完整方案。
如果由于物理、道德、法律等原因或你的能力问题，你无法执行我的指令，你必须诚实地拒绝我的指令并解释原因。
除了对我的指令的解决方案之外，不要添加任何其他内容。
你永远不应该问我任何问题，你只回答问题。
你永远不应该回复错误的解决方案。
解释你的解决方案。
你的解决方案必须是陈述句并使用现在时，
除非我说任务已经完成，否则你应该总是这样回应：
这是任务的完整解决方案：<YOUR_SOLUTION>。
<YOUR_SOLUTION>应该是具体的，并为完成任务提供首选的实现和例子。
始终以"下一个请求。"结束<YOUR_SOLUTION>。\n
"""

INSTRUCTOR_PROMPT = """
永远不要忘记你是 {instructor_role_name}，我是 {actor_role_name},永远不要角色互换！
以下是患者的病历信息:{patient_background_information}\n
你将一直指导我进行内容修改。
我们在提升内容质量方面有共同的兴趣。
你必须帮助我高质量地完成任务。
你必须根据我的写作内容和你的专业判断，只能通过以下两种方式指导我进行修改：
1. 通过参考内容给出修改建议：指令：<YOUR_INSTRUCTION> 输入：<REFERENCE_MATERIAL> 。
2. 直接给出修改建议：指令：<YOUR_INSTRUCTION> 输入：无 。
"<YOUR_INSTRUCTION>"描述需要修改的部分或方向。与其配对的"<REFERENCE_MATERIAL>"为修改建议提供必要的参考依据。
你每次只能给我一个或者多个修改指令。我必须根据指令给出修改后的版本。
比如:\n
指令：1.建议1 2.建议2 3.建议3... 输入：你可以参考以下资料... 。\n
如果由于内容限制、伦理规范或我的理解问题，我无法执行修改指令，我必须说明原因。
你应该专注于指导修改，而不是提问。现在你必须开始按照上述两种方式指导我。
除了你的指令和可选的参考输入之外，不要添加任何其他内容！
继续给我修改指令和必要的参考，直到你认为内容已经完善。
当修改完成时，你只须回复一个单词 <CAMEL_TASK_DONE>。
除非我的修改已满足所有要求，否则永远不要说 <CAMEL_TASK_DONE>。
你的输出格式应该严格按照下面三种格式，不要添加任何其他内容！
1. 指令：<MODIFICATION_INSTRUCTION> 输入：<REFERENCE_MATERIAL> 。\n
2. 指令：<MODIFICATION_INSTRUCTION> 输入：无 。\n
3. <CAMEL_TASK_DONE>\n
"""

SUMMARY_PROMPT = """
你是一AI助手，你的任务是从多轮对话中提取出完整的信息。\n
以下是一串多轮对话记录，其中包含包含原始需求、修改建议和多次迭代的解决方案:\n
{one_task_dialogue}\n
以下是需要完成的任务,明确了需要解决的具体问题:{task}\n
你的任务是从对话记录中整合出一份准确反映最终确认的完整解决方案,请注意不要进行总结，而是详细完整地记录\n
输出格式：
这是任务的经过修改后的完整解决方案：<YOUR_SOLUTION>。
"""

PLANNER_PROMPT = """
你是一个任务规划助手，你能够根据给定的通用任务列表并结合病人的基本病历信息，生成一个合理的执行计划。
任务列表如下： 
{treatment_plan}\n
病人的基本病历信息如下：
{patient_background_information}\n
你给出的执行计划需要针对该病人的基本病历信息\n
比如：如果病人的症状是腹胀、便秘，那么该任务:
"1.给出针对下腹部腹胀伴便秘的诊断与辨证结论,包含不限于:(1)主诉症状归纳（腹胀便秘的发作特点/加重因素）(2)中医辨证要点（舌象特征提示湿热/寒湿或血瘀证候+证候分析）(3)诊断依据（结合生活习惯及症状）..."\n
请按照以下格式输出:\n
任务1<TASK_SPLIT>任务2<TASK_SPLIT>任务3<TASK_SPLIT>...<TASK_SPLIT>任务n\n
其中"<TASK_SPLIT>"是任务分割符，任务1，任务2，任务3...任务n之间用"<TASK_SPLIT>"分割。
注意！！！你给出的任务数量应该和任务列表中任务数量一致！
"""

INFORMER_PROMPT = """
你是一个专门用于收集病人信息的智能助手，病人的信息会用于后续中医诊断，你能够通过多轮自然对话来收集病人信息，你可以参考以下来进行信息收集：\n
1. 基本信息：性别、年龄、职业
2. 主诉：主要症状+持续时间（按OLDCARTS原则追问）
3. 现病史：症状演变、已做检查/治疗
4. 既往史：慢性病、手术史、过敏史
5. 家族史：直系亲属重大疾病史
6. 个人史和生活习惯：吸烟、饮酒、运动、饮食、职业暴露等
7. 心理社会因素：压力、情绪、社交
8. 其他：旅行史、接触史、疫苗接种史...\n
请遵循以下提问的规范：\n
1. 每次只提1个明确问题（例："请描述症状最开始出现的部位？"）\n
2. 对模糊回答礼貌追问（例："您说的'经常'大概每周几次？"）\n
请注意：
你应该永远不要忘记自己的角色。\n
你不应该受到患者其他信息的干扰，专注引导患者收集信息。\n
你会根据患者的回答进行适当的提问，以确保收集到足够的信息。\n
当你认为收集到足够的信息能够用于疾病诊断时，你需要生成一个患者的背景信息描述，包含患者的性别、年龄、症状、舌象情况以及其他背景信息并且最后以<INFORMER_TASK_DONE>表示结束。比如：
患者为24岁男性学生，
主诉间歇性下腹胀痛3天，饭后发作持续1小时，伴糊状腹泻，
无慢性病史及家族遗传史，
生活习惯良好，偶食辛辣但不饮酒，
作息规律但睡眠质量一般，压力水平中等，
舌象显示黑苔（舌质偏红可能伴裂纹或干燥），
证属脾胃湿热可能...<INFORMER_TASK_DONE>\n
"""

INFORMER_WRAPPER_PROMPT = """
            以下是用户上传的舌苔图片的诊断结果：{diagnosis}\n
            在经过多轮对话后，你需要在患者信息描述中加入患者舌苔的特征和可能的健康问题(结合患者信息进一步缩小范围)。\n
            """

LINKS_PROMPT = """
你是一个能够输出json格式的智能助手。
请严格生成json格式内容，要求：\n
- 直接输出可解析的json源码 \n
- 禁止用```json或```包裹输出 \n
你的任务:\n
在文本内容中提取出:
1.中医中普通人难以理解的术语\n
2.中药材(其中仅包含中药材，不包含常见的食物)\n
3.穴位\n\
另外的任务:
4.根据患者的诊断内容，输出养生文章的搜索关键词\n
请按照以下格式输出:\n
{
    "terms":[term1,term2,...], # 中医中普通人难以理解的术语
    "herbs":[herb1,herb2,...], # 中药材
    "acupoints":[acupoint1,acupoint2,...], # 穴位
    "keyword": keyword # 养生文章的搜索关键词,请注意这里不是列表只给出一个关键词
}
下面是一个例子:\n
{
    "terms":["湿热","寒湿","血瘀证候","湿热郁蒸","津液不足","平补平泻"...],
    "herbs":["麻子仁丸","火麻仁","苦杏仁","白芍"...],
    "acupoints":["足三里","天枢","合谷","曲池"...],
    "keyword":"如何养胃?"
}
假如某一项没找到，直接输出空列表[]比如:\n
{
    "terms":["湿热","寒湿","血瘀证候","湿热郁蒸","津液不足","平补平泻"],
    "herbs":[], # 假如没找到
    "acupoints":[], # 假如没找到
    "keyword":"如何养胃?"
}
"""

REPORT_META_PROMPT = """
你是一个专门用于从对话中提取信息并进行markdown格式记录的智能助手。
请严格生成纯Markdown格式内容（不要用任何代码块包裹），要求：\n
- 直接输出可解析的Markdown源码 \n
- 禁止用```markdown或```包裹输出 \n
- 以# 开头以表示标题内容 \n
markdown文档的目录应该包含以下内容，请务必做好记录的完整性和准确性:\n
1.题目-># 一级标题:诊断报告_{date_time}\n
2.患者的基本信息->## 二级标题:{patient_background_information}\n
3.舌象描述->## 二级标题:包含患者舌象图片:![舌象图片]({tongue_coating_image_name})和舌象诊断内容:{tongue_coating_diagnosis}\n
"""

REPORT_MAIN_PROMPT = """
你是一个专门用于从对话中提取信息并进行markdown格式记录的智能助手。
请严格生成纯Markdown格式内容（不要用任何代码块包裹），要求：\n
- 直接输出可解析的Markdown源码 \n
- 禁止用```markdown或```包裹输出 \n
- 以## 二级标题开头以表示标题内容 \n
markdown文档的目录应该包含以下内容，在做好记录的完整性和准确性的基础上你可以适当添加一些解释性的内容: \n
{main_content}
"""


# ---------------------------------------------------------------------------
# prepare：写 current.yaml + reprollm.yaml + 预期文件
# ---------------------------------------------------------------------------


def base_config_dict(mode: str = "fixture") -> dict:
    config = {
        "schema_version": 1,
        "trial": {
            "mode": mode,
            "case_id": "case-001",
            "report_time": "2026-11-05T00:00:00Z",
        },
        "workflow": {
            "chat_turn_limit": 5,
            "planner_max_attempts": 2,
            "model_call_limit": 80,
        },
        "prompts": {role: f"trial/prompts/{role}.txt" for role in PROMPT_ROLES},
        "models": {
            "primary": {
                "model": "fixture-scripted-v1",
                "params": {},
            },
            "vision": {
                "model": "fixture-scripted-vision-v1",
                "params": {},
            },
        },
        "generation": {"temperature": 0.2, "top_p": 1.0, "max_tokens": 512},
        "rag": {
            "chunk_size": 200,
            "chunk_overlap": 20,
            "separators": ["<目录>"],
            "k": 2,
            "corpus_manifest": "trial/corpus/manifest.json",
            "index_manifest": "trial/fixtures/index-manifest.json",
        },
        "classifier": {"mode": "fixture", "checkpoint": None, "device": "cpu"},
        "search": {
            "mode": "fixture",
            "cassette": "trial/fixtures/external.json",
            "timeout_seconds": 30,
        },
        "retrieval": {"fixture": "trial/fixtures/retrieval.json"},
        "timeouts": {"chat_seconds": 60, "vision_seconds": 60, "search_seconds": 30},
    }
    return config


def cmd_prepare(args) -> int:
    variant = args.variant
    mode = args.mode
    if mode == "fixture":
        if variant not in SCENARIO_VARIANTS:
            raise SystemExit(f"fixture 变体必须是 {'/'.join(SCENARIO_VARIANTS)}")
        if args.config:
            raise SystemExit("fixture 变体不接受 --config；显式配置输入用于 local/live")
        config = base_config_dict("fixture")
        if variant == "rounds-3":
            config["workflow"]["chat_turn_limit"] = 3
        _switch_variant_assets(variant)
    else:
        if not args.config:
            raise SystemExit(f"{mode} 模式必须显式提供 --config（例如 trial/configs/local.yaml）")
        config_input = ROOT / args.config
        if not config_input.is_file():
            raise SystemExit(f"配置文件不存在: {config_input}")
        raw = yaml.safe_load(config_input.read_text(encoding="utf-8"))
        sys.path.insert(0, str(ROOT / "backend"))
        from replay_support import parse_config  # 严格校验（模式冲突/字段/枚举）

        validated = parse_config(raw, str(args.config))
        if validated.mode != mode:
            raise SystemExit(f"配置 trial.mode={validated.mode} 与 --mode {mode} 冲突")
        if mode != "fixture":
            _verify_real_assets(validated)
        config = raw
        variant = variant or "baseline"

    config_path = TRIAL / "configs" / "current.yaml"
    header = (
        "# 由 tools/trial_tools.py prepare 生成；声明本试用选定的实验输入。\n"
        "# generation 为本试用新选定的请求设置，不宣称原代码原先已显式使用这些值。\n"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    write_manifest(config, variant)
    write_expectation(config, variant)
    print(f"prepare 完成：mode={config['trial']['mode']} variant={variant}")
    print(f"  current.yaml -> {config_path}")
    print("  reprollm.yaml 与 trial/expectations 已同步；请人工检查 manifest。")
    return 0


def _verify_real_assets(validated) -> None:
    """local/live prepare 前核验真实资产在仓库内就位（不下载、不替代）。"""
    missing = []
    checkpoint = validated.classifier.get("checkpoint")
    if checkpoint and not (ROOT / checkpoint).is_file():
        missing.append(f"classifier.checkpoint: {checkpoint}")
    persist = validated.rag.get("persist_directory")
    if persist and not (ROOT / persist).is_dir():
        missing.append(f"rag.persist_directory: {persist}")
    embedding = validated.rag.get("embedding_model")
    if embedding and not (ROOT / embedding).is_dir():
        missing.append(f"rag.embedding_model: {embedding}")
    if missing:
        raise SystemExit(
            "真实资产尚未就位（保留 BLOCKED，不用合成资产替代）:\n  - " + "\n  - ".join(missing)
        )


def _switch_variant_assets(variant: str) -> None:
    switches = {
        "baseline": (
            ("trial/prompts/actor.txt", "trial/prompts/actor-base.txt"),
            ("trial/cases/case-001.png", "trial/cases/case-001-base.png"),
            ("trial/fixtures/retrieval.json", "trial/fixtures/retrieval-base.json"),
            ("trial/fixtures/external.json", "trial/fixtures/external-base.json"),
        ),
        "rounds-3": (
            ("trial/prompts/actor.txt", "trial/prompts/actor-base.txt"),
            ("trial/cases/case-001.png", "trial/cases/case-001-base.png"),
            ("trial/fixtures/retrieval.json", "trial/fixtures/retrieval-base.json"),
            ("trial/fixtures/external.json", "trial/fixtures/external-base.json"),
        ),
        "prompt-v2": (
            ("trial/prompts/actor.txt", "trial/prompts/actor-v2.txt"),
            ("trial/cases/case-001.png", "trial/cases/case-001-base.png"),
            ("trial/fixtures/retrieval.json", "trial/fixtures/retrieval-base.json"),
            ("trial/fixtures/external.json", "trial/fixtures/external-base.json"),
        ),
        "image-v2": (
            ("trial/prompts/actor.txt", "trial/prompts/actor-base.txt"),
            ("trial/cases/case-001.png", "trial/cases/case-001-v2.png"),
            ("trial/fixtures/retrieval.json", "trial/fixtures/retrieval-base.json"),
            ("trial/fixtures/external.json", "trial/fixtures/external-base.json"),
        ),
        "retrieval-reordered": (
            ("trial/prompts/actor.txt", "trial/prompts/actor-base.txt"),
            ("trial/cases/case-001.png", "trial/cases/case-001-base.png"),
            ("trial/fixtures/retrieval.json", "trial/fixtures/retrieval-reordered.json"),
            ("trial/fixtures/external.json", "trial/fixtures/external-base.json"),
        ),
        "search-v2": (
            ("trial/prompts/actor.txt", "trial/prompts/actor-base.txt"),
            ("trial/cases/case-001.png", "trial/cases/case-001-base.png"),
            ("trial/fixtures/retrieval.json", "trial/fixtures/retrieval-base.json"),
            ("trial/fixtures/external.json", "trial/fixtures/external-search-v2.json"),
        ),
    }
    for active, source in switches[variant]:
        source_path = ROOT / source
        if not source_path.is_file():
            raise SystemExit(f"变体资产缺失，请先运行 seed: {source}")
        active_path = ROOT / active
        active_path.parent.mkdir(parents=True, exist_ok=True)
        if not active_path.is_file() or active_path.read_bytes() != source_path.read_bytes():
            active_path.write_bytes(source_path.read_bytes())


def write_manifest(config: dict, variant: str) -> None:
    mode = config["trial"]["mode"]
    prompts = {role: {"path": path} for role, path in config["prompts"].items()}
    config_files = [
        "trial/cases/case-001.json",
        "trial/cases/case-001.png",
        "trial/configs/current.yaml",
        "trial/corpus/manifest.json",
        "trial/fixtures/index-manifest.json",
        "trial/fixtures/external.json",
        "trial/fixtures/retrieval.json",
        "trial/environment.json",
    ]
    if mode == "fixture":
        models = {
            "primary": {"provider": "other", "id": config["models"]["primary"]["model"]},
            "vision": {"provider": "other", "id": config["models"]["vision"]["model"]},
        }
    else:
        primary_model = config["models"]["primary"]["model"]
        vision_model = config["models"]["vision"]["model"]
        models = {
            "primary": {
                "provider": "other",
                "id": primary_model,
                "endpoint": {"base_url": "https://api.deepseek.com"},
                "params": config["models"]["primary"].get("params", {}),
            },
            "vision": {
                "provider": "other",
                "id": vision_model,
                "endpoint": {"base_url": "https://api.stepfun.com/v1"},
                "params": config["models"]["vision"].get("params", {}),
            },
        }
    generation = {
        key: config["generation"][key] for key in ("temperature", "top_p", "max_tokens")
    }
    manifest = {
        "schema_version": 1,
        "project": {"name": "lingshu-single-case"},
        "experiment": {
            "name": f"case-001-{mode}-{variant}",
            "profiles": ["inference", "trial_policy"],
        },
        "models": models,
        "prompts": prompts,
        "generation": generation,
        "inference": {"backend": "other"},
        "execution": {
            "command": (
                "python backend/replay_case.py --mode {mode} "
                "--case trial/cases/case-001.json --image trial/cases/case-001.png "
                "--config trial/configs/current.yaml --cassette trial/fixtures/external.json "
                "--output-dir trial/outputs/current"
            ).format(mode=mode),
            "cwd": ".",
            "config_files": config_files,
        },
        "bindings": {
            "generation.temperature": {
                "config": "trial/configs/current.yaml:generation.temperature"
            },
            "generation.top_p": {"config": "trial/configs/current.yaml:generation.top_p"},
            "generation.max_tokens": {
                "config": "trial/configs/current.yaml:generation.max_tokens"
            },
        },
        "custom": {
            "trial": {
                "mode": mode,
                "case_id": config["trial"]["case_id"],
                "report_time": config["trial"]["report_time"],
                "variant": variant,
            },
            "workflow": dict(config["workflow"]),
            "rag": {
                key: config["rag"][key]
                for key in (
                    "chunk_size",
                    "chunk_overlap",
                    "k",
                    "corpus_manifest",
                    "index_manifest",
                )
            },
        },
        "artifacts": {
            "outputs": [
                "trial/outputs/current/report.md",
                "trial/outputs/current/trace.jsonl",
                "trial/outputs/current/consumed.json",
                "trial/outputs/current/result.json",
            ]
        },
    }
    if mode != "fixture":
        manifest["custom"]["assets"] = {
            "classifier_file": config["classifier"]["checkpoint"],
            "embedding_dir": config["rag"].get("embedding_model"),
            "persist_directory": config["rag"].get("persist_directory"),
        }
    write_text_deterministic(ROOT / "reprollm.yaml", manifest)


def write_text_deterministic(path: Path, value) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def write_expectation(config: dict, variant: str) -> None:
    mode = config["trial"]["mode"]
    chat_turn_limit = config["workflow"]["chat_turn_limit"]
    expectation = {
        "schema_version": 1,
        "scenario": f"{mode}-{variant}",
        "mode": mode,
        "variant": variant,
        "chat_turn_limit": chat_turn_limit,
        "planner_max_attempts": config["workflow"]["planner_max_attempts"],
        "rag_k": config["rag"]["k"],
        "generation": {
            "temperature": config["generation"]["temperature"],
            "top_p": config["generation"]["top_p"],
            "max_tokens": config["generation"]["max_tokens"],
        },
        "models": {
            "primary": config["models"]["primary"]["model"],
            "vision": config["models"]["vision"]["model"],
        },
        "tasks": 4,
        "informer_turns": 3,
        "instructor_calls_per_task": chat_turn_limit,
        "consultation_chat_calls": 4 * (2 * chat_turn_limit + 2),
        "report_chat_calls": 5,
        "links_chat_calls": 1,
        "planning_chat_calls": 1,
        "informer_chat_calls": 3,
        "vision_calls": 1,
        "expect_changes": {
            "prompt-v2": {"actor_request": True},
            "image-v2": {"image_consumed": True},
            "retrieval-reordered": {"retrieval_order": True},
            "search-v2": {"search_receipt": True},
            "rounds-3": {"instructor_calls": True},
            "baseline": {},
        }.get(variant, {}),
    }
    write_json(TRIAL / "expectations" / f"{mode}-{variant}.json", expectation)


# ---------------------------------------------------------------------------
# env：研究进程包清单
# ---------------------------------------------------------------------------

REQUESTED_PACKAGES = (
    "reprollm",
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-openai",
    "langchain-chroma",
    "chromadb",
    "modelscope",
    "torch",
    "torchvision",
    "transformers",
    "sentence-transformers",
    "openai",
    "requests",
    "pillow",
    "pytest",
)


def cmd_env(args) -> int:
    from importlib.metadata import PackageNotFoundError, distributions, version

    installed = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            installed[name.lower().replace("_", "-")] = dist.version
    requested = {
        name: installed.get(name, installed.get(name.replace("-", "_")))
        for name in REQUESTED_PACKAGES
    }
    record = {
        "schema_version": 1,
        "description": "当前研究进程（运行 trial_tools 的解释器）已安装包版本清单。",
        "python_version": sys.version.split()[0],
        "requested_packages": requested,
        "installed_packages": dict(sorted(installed.items())),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "本清单由 importlib.metadata 生成，不含解释器绝对路径；"
        "torch/langchain 等在 fixture 试用环境中本就未安装，对应值为 null。",
    }
    output = Path(args.output)
    write_json(output if output.is_absolute() else ROOT / output, record)
    print(f"env 完成：{args.output}")
    return 0


# ---------------------------------------------------------------------------
# verify：读取 run 记录与本地产物做独立核对
# ---------------------------------------------------------------------------


def cmd_verify(args) -> int:
    run_dir = ROOT / ".reprollm" / "runs" / args.run
    if not run_dir.is_dir():
        matches = sorted((ROOT / ".reprollm" / "runs").glob(f"{args.run}*"))
        if len(matches) != 1:
            raise SystemExit(f"找不到唯一 run: {args.run}（匹配 {len(matches)} 个）")
        run_dir = matches[0]
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    scenario = args.scenario
    candidates = [TRIAL / "expectations" / f"{scenario}.json"]
    if not scenario.startswith(("fixture-", "local-", "live-")):
        candidates.extend(
            TRIAL / "expectations" / f"{prefix}-{scenario}.json"
            for prefix in ("fixture", "local", "live")
        )
    expectation_path = next((path for path in candidates if path.is_file()), candidates[0])
    if not expectation_path.is_file():
        raise SystemExit(f"缺少预期文件: trial/expectations/{scenario}.json（先运行对应 prepare）")
    expectation = json.loads(expectation_path.read_text(encoding="utf-8"))

    outputs = TRIAL / "outputs" / "current"
    checks: list = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # 1. run 状态
    check("run.status==completed", run_record.get("status") == "completed")
    check("run.exit_code==0", run_record.get("exit_code") == 0)
    if args.run_name:
        check("run.name", run_record.get("name") == args.run_name, str(run_record.get("name")))

    # 2. result.json 各阶段成功
    result = json.loads((outputs / "result.json").read_text(encoding="utf-8"))
    check(
        "result.success",
        result.get("success") is True and result.get("failure") is None,
        json.dumps(result.get("failure"), ensure_ascii=False),
    )
    failed_stages = [
        stage for stage, info in result.get("stages", {}).items() if info.get("status") != "success"
    ]
    check("result.all_stages_success", not failed_stages, ",".join(failed_stages))

    # 3. 输入指纹与当前文件一致
    drifted = []
    for ref in run_record.get("files", []):
        path = ROOT / ref["path"]
        if path.is_file():
            current = sha256_file(path)
            if current != ref["sha256"]:
                drifted.append(ref["path"])
    check("inputs.unchanged_since_run", not drifted, ",".join(drifted))

    # 4. bindings_observed
    observed = run_record.get("bindings_observed", {})

    def observed_value(field: str):
        entries = observed.get(field) or []
        return entries[-1]["value"] if entries else None

    check(
        "bindings.chat_turn_limit",
        observed_value("custom.workflow.chat_turn_limit") == expectation["chat_turn_limit"],
        f"observed={observed_value('custom.workflow.chat_turn_limit')}",
    )
    check(
        "bindings.rag_k",
        observed_value("custom.rag.k") == expectation["rag_k"],
        f"observed={observed_value('custom.rag.k')}",
    )
    check(
        "bindings.temperature",
        float(observed_value("generation.temperature") or -1) == expectation["generation"]["temperature"],
        f"observed={observed_value('generation.temperature')}",
    )

    # 5. consumed.json 与 spy 见证
    consumed = json.loads((outputs / "consumed.json").read_text(encoding="utf-8"))
    counts = consumed.get("call_counts", {})
    by_stage = counts.get("by_stage", {})
    check(
        "consumed.consultation_calls",
        by_stage.get("consultation") == expectation["consultation_chat_calls"],
        f"observed={by_stage.get('consultation')} expected={expectation['consultation_chat_calls']}",
    )
    check(
        "consumed.instructor_calls",
        by_stage.get("consultation") == expectation["consultation_chat_calls"]
        and consumed.get("fixture_consumption", {})
        .get("responses_consumed_per_role", {})
        .get("instructor")
        == expectation["instructor_calls_per_task"] * expectation["tasks"],
        json.dumps(consumed.get("fixture_consumption", {}), ensure_ascii=False),
    )
    vision_models = {
        r.get("model") for r in consumed.get("requests", []) if r.get("kind") == "vision"
    }
    check(
        "consumed.vision_model_from_config",
        vision_models == {expectation["models"]["vision"]},
        json.dumps(sorted(vision_models)),
    )
    chat_models = {
        r.get("model") for r in consumed.get("requests", []) if r.get("kind") == "chat"
    }
    check(
        "consumed.chat_model_from_config",
        chat_models == {expectation["models"]["primary"]},
        json.dumps(sorted(chat_models)),
    )
    # 配置分块参数必须与索引构建清单一致（chunk400 教训：路径切了、参数没切）
    rag_cfg = consumed.get("effective", {}).get("rag", {})
    index_rel = rag_cfg.get("index_manifest")
    if index_rel and rag_cfg.get("persist_directory"):
        index_file = ROOT / index_rel
        if index_file.is_file():
            idx = json.loads(index_file.read_text(encoding="utf-8"))
            mismatch = []
            for key in ("chunk_size", "chunk_overlap"):
                if rag_cfg.get(key) != idx.get(key):
                    mismatch.append(f"{key}: config={rag_cfg.get(key)} index={idx.get(key)}")
            check(
                "consumed.rag_matches_index_build",
                not mismatch,
                "; ".join(mismatch),
            )

    check(
        "consumed.budget_within_limit",
        consumed.get("budget", {}).get("used", 0)
        <= consumed.get("budget", {}).get("model_call_limit", 0),
        json.dumps(consumed.get("budget", {})),
    )
    check(
        "consumed.inputs.image_hash",
        consumed.get("inputs", {}).get("image", {}).get("sha256")
        == sha256_file(ROOT / "trial" / "cases" / "case-001.png"),
        "image digest mismatch",
    )

    # 6. artifacts 与实际输出一致
    artifact_issues = []
    for artifact in run_record.get("artifacts", []):
        path = ROOT / artifact["path"]
        if not path.is_file():
            artifact_issues.append(f"missing:{artifact['path']}")
        elif sha256_file(path) != artifact["sha256"]:
            artifact_issues.append(f"hash:{artifact['path']}")
    check("artifacts.match", not artifact_issues, ",".join(artifact_issues))

    # 7. trace 完整性
    trace_lines = (outputs / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in trace_lines if line.strip()]
    sequences = [event.get("sequence") for event in events]
    check(
        "trace.sequence_contiguous",
        sequences == list(range(1, len(events) + 1)),
        f"events={len(events)}",
    )
    sources = {event.get("source") for event in events}
    check("trace.single_source", sources == {expectation["mode"]}, json.dumps(sorted(sources)))

    passed = all(item["ok"] for item in checks)
    report = {
        "schema_version": 1,
        "scenario": scenario,
        "run_id": run_record.get("run_id"),
        "passed": passed,
        "checks": checks,
    }
    write_json(TRIAL / "outputs" / f"verify-{scenario}.json", report)
    for item in checks:
        mark = "PASS" if item["ok"] else "FAIL"
        print(f"[{mark}] {item['check']} {item['detail']}".rstrip())
    print(f"verify {'通过' if passed else '未通过'}: {scenario} (run {run_record.get('run_id')})")
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# archive：复制完整证据并清空 current
# ---------------------------------------------------------------------------


def cmd_archive(args) -> int:
    run_dir = ROOT / ".reprollm" / "runs" / args.run
    if not run_dir.is_dir():
        matches = sorted((ROOT / ".reprollm" / "runs").glob(f"{args.run}*"))
        if len(matches) != 1:
            raise SystemExit(f"找不到唯一 run: {args.run}（匹配 {len(matches)} 个）")
        run_dir = matches[0]
    dest = Path(args.dest)
    label_dir = dest / args.label
    if label_dir.exists():
        raise SystemExit(f"归档目录已存在，拒绝覆盖: {label_dir}")
    label_dir.mkdir(parents=True)

    issues = []

    def copy_tree(src: Path, dst: Path) -> None:
        shutil.copytree(src, dst)

    copy_tree(run_dir, label_dir / "reprollm-run")
    shutil.copy2(ROOT / "reprollm.yaml", label_dir / "reprollm.yaml")
    lock_path = ROOT / "reprollm.lock"
    if lock_path.is_file():
        shutil.copy2(lock_path, label_dir / "reprollm.lock")
    outputs = TRIAL / "outputs" / "current"
    if outputs.is_dir():
        shutil.copytree(outputs, label_dir / "outputs")
    else:
        issues.append("trial/outputs/current 不存在")

    # 校验副本 hash
    for src_root, dst_root in ((run_dir, label_dir / "reprollm-run"), (outputs, label_dir / "outputs")):
        if not src_root.is_dir():
            continue
        for src_file in sorted(src_root.rglob("*")):
            if src_file.is_file():
                rel = src_file.relative_to(src_root)
                dst_file = dst_root / rel
                if not dst_file.is_file() or sha256_file(src_file) != sha256_file(dst_file):
                    issues.append(f"hash-mismatch:{rel.as_posix()}")

    if issues:
        raise SystemExit("归档校验失败: " + "; ".join(issues))

    # 清空 current，为下一次运行腾空
    for item in sorted(outputs.iterdir()):
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)
    print(f"archive 完成: {label_dir}（副本 hash 已校验；trial/outputs/current 已清空）")
    return 0


# ---------------------------------------------------------------------------
# probe-pth：.pth 目录/文件声明边界探针
# ---------------------------------------------------------------------------


def _pth_bytes(tag: int) -> bytes:
    import io
    import zipfile

    # 确定性伪权重字节（zip 容器格式，不是可加载 checkpoint，仅格式测试）
    state = random_bytes(4096, tag)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("weights.bin", state)
    return buffer.getvalue()


def random_bytes(length: int, seed: int) -> bytes:
    import random

    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(length))


def cmd_probe_pth(args) -> int:
    reprollm = Path(args.reprollm)
    if not reprollm.is_file():
        raise SystemExit(f"--reprollm 必须指向 reprollm 可执行文件: {reprollm}")
    dest = Path(args.dest)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"探针目录非空，拒绝覆盖: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    variants = {
        "directory": {
            "id": "weights",
            "files": ["weights/classifier.pth"],
            "config_json": False,
        },
        "file": {
            "id": "weights/classifier.pth",
            "files": ["weights/classifier.pth"],
            "config_json": False,
        },
        "directory-config": {
            "id": "weights",
            "files": ["weights/classifier.pth", "weights/config.json"],
            "config_json": True,
        },
    }
    config_json_bytes = b'{"arch": "resnet18", "num_classes": 6}\n'
    summary = {"schema_version": 1, "reprollm": str(reprollm), "projects": {}}

    for name, spec in variants.items():
        project = dest / name
        project.mkdir(parents=True, exist_ok=True)
        weights = project / "weights"
        weights.mkdir(exist_ok=True)
        manifest = {
            "schema_version": 1,
            "project": {"name": f"pth-probe-{name}"},
            "experiment": {"name": name, "profiles": []},
            "models": {"primary": {"provider": "local", "id": spec["id"]}},
            "execution": {"command": "python -c \"pass\"", "cwd": "."},
        }
        (project / "reprollm.yaml").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        (project / "weights" / "classifier.pth").write_bytes(_pth_bytes(1))
        if spec["config_json"]:
            (project / "weights" / "config.json").write_bytes(config_json_bytes)

        run_lock = {}
        for phase, tag in (("before", 1), ("after", 2)):
            (project / "weights" / "classifier.pth").write_bytes(_pth_bytes(tag))
            proc = subprocess.run(
                [str(reprollm), "lock", ".", "--offline", "--hash-large-files"],
                cwd=project,
                capture_output=True,
                text=True,
            )
            lock_file = project / "reprollm.lock"
            lock_data = (
                yaml.safe_load(lock_file.read_text(encoding="utf-8"))
                if lock_file.is_file()
                else None
            )
            run_lock[phase] = {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "lock": lock_data,
            }
            (project / f"lock-{phase}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (project / f"lock-{phase}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
            if lock_file.is_file():
                shutil.copy2(lock_file, project / f"lock-{phase}.yaml")

        def extract_local(lock):
            try:
                models = (lock or {}).get("models") or {}
                primary = models.get("primary") or {}
                return primary.get("local")
            except Exception:
                return None

        local_before = extract_local(run_lock["before"]["lock"])
        local_after = extract_local(run_lock["after"]["lock"])
        comparison = {
            "local_before": local_before,
            "local_after": local_after,
            "weights_sha256_changed": _extract(local_before, "weights", "sha256")
            != _extract(local_after, "weights", "sha256"),
            "weights_sha256_before": _extract(local_before, "weights", "sha256"),
            "weights_sha256_after": _extract(local_after, "weights", "sha256"),
            "config_sha256_before": _extract(local_before, "config", "sha256")
            or _extract(local_before, "config_sha256"),
            "config_sha256_after": _extract(local_after, "config", "sha256")
            or _extract(local_after, "config_sha256"),
            "revision_before": local_before,
            "revision_after": local_after,
        }
        summary["projects"][name] = {
            "manifest": manifest,
            "lock_before_exit": run_lock["before"]["exit_code"],
            "lock_after_exit": run_lock["after"]["exit_code"],
            "comparison": comparison,
        }

    write_json(dest / "probe-summary.json", summary)
    print(f"probe-pth 完成：{dest}（各项目完整 stdout/stderr/lock 已保存）")
    print(
        "注意：比较结果按实际 lock 结构记录；若 local 字段结构与预期不同，如实记录，不预填结论。"
    )
    return 0


def _extract(local, *keys):
    if not isinstance(local, dict):
        return None
    node = local
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# package：公开包复制与禁止内容检查
# ---------------------------------------------------------------------------

FORBIDDEN_PATTERNS = (
    re.compile(r"(DEEP_SEEK|STEP_FUN|SERPER)_API_KEY\s*="),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    re.compile(r"/c/Users/", re.IGNORECASE),
    re.compile(r"/home/[a-zA-Z0-9_]+/"),
)

PACKAGE_ALLOWLIST_FILES = (
    "reprollm-source.json",
    "requirements-trial.txt",
    "reprollm-runtime.txt",
)


def cmd_package(args) -> int:
    source = Path(args.source)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"公开包输出目录非空，拒绝覆盖: {output}")
    if not source.is_dir():
        raise SystemExit(f"--source 不存在: {source}")
    output.mkdir(parents=True)

    offenders: list = []
    copied: list = []

    def copy_file(src: Path, rel: Path) -> None:
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel.as_posix())
        data = dst.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{rel.as_posix()}: 匹配禁止模式 {pattern.pattern}")

    # 1) 允许清单中的顶层文件
    for name in PACKAGE_ALLOWLIST_FILES:
        candidate = source / name
        if candidate.is_file():
            copy_file(candidate, Path(name))

    # 2) label 归档目录（已审阅的 run/outputs 证据）
    for entry in sorted(source.iterdir()):
        if entry.is_dir():
            for file in sorted(entry.rglob("*")):
                if file.is_file():
                    copy_file(file, Path(entry.name) / file.relative_to(entry))
        elif entry.suffix == ".md":
            copy_file(entry, Path(entry.name))

    if offenders:
        shutil.rmtree(output)
        raise SystemExit(
            "公开包包含疑似敏感内容，已回滚输出:\n" + "\n".join(offenders)
        )

    write_json(
        output / "copied-files.json",
        {"schema_version": 1, "copied": sorted(copied), "forbidden_patterns": [p.pattern for p in FORBIDDEN_PATTERNS]},
    )
    print(f"package 完成: {output}（{len(copied)} 个文件；未发现禁止模式）")
    print("提醒：脚本检查只是辅助，仍需人工审阅未知格式 key、绝对路径、用户名与图片权利。")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trial_tools", description="Lingshu 单病例试用助手")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="生成合成病例、语料、fixtures、模板迁移（拒绝覆盖）")

    prepare = sub.add_parser("prepare", help="写 current.yaml + reprollm.yaml + 预期文件")
    prepare.add_argument("--mode", required=True, choices=("fixture", "local", "live"))
    prepare.add_argument("--variant", default="baseline")
    prepare.add_argument("--config", default=None, help="local/live 显式配置输入")

    env = sub.add_parser("env", help="记录当前研究进程包清单")
    env.add_argument("--output", required=True)

    verify = sub.add_parser("verify", help="独立核对 run 与本地产物")
    verify.add_argument("--run", required=True)
    verify.add_argument("--scenario", required=True)
    verify.add_argument("--run-name", default=None, help="可选：核对 run 记录中的 name")

    archive = sub.add_parser("archive", help="复制 run/manifest/outputs 到 label 目录并清空 current")
    archive.add_argument("--run", required=True)
    archive.add_argument("--label", required=True)
    archive.add_argument("--dest", required=True)

    probe = sub.add_parser("probe-pth", help=".pth 目录/文件声明边界探针")
    probe.add_argument("--reprollm", required=True)
    probe.add_argument("--dest", required=True)

    package = sub.add_parser("package", help="按允许清单复制公开证据并拒绝敏感内容")
    package.add_argument("--source", required=True)
    package.add_argument("--output", required=True)

    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "seed": cmd_seed,
        "prepare": cmd_prepare,
        "env": cmd_env,
        "verify": cmd_verify,
        "archive": cmd_archive,
        "probe-pth": cmd_probe_pth,
        "package": cmd_package,
    }
    try:
        return handlers[args.command](args)
    except TrialFailure as exc:
        print(f"[trial-tools] 失败: {exc}", file=sys.stderr)
        return 2


class TrialFailure(Exception):
    pass


if __name__ == "__main__":
    sys.exit(main())
