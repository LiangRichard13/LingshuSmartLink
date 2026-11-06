"""舌苔视觉诊断 Agent。

模型名只来自注入的 VisionInvoker（配置唯一来源）；请求 create 使用同一个值，
消除原代码 self.model 与实际请求 model 硬编码不一致的问题。
分类器由外部注入：fixture 模式注入 FixtureClassifier，不实例化 torch 模型。
"""

from __future__ import annotations

import base64
from pathlib import Path

from replay_support import TrialError, VisionInvoker, REASON_VISION_FAILURE

_SUPPORTED_IMAGE_TYPES = ("jpg", "jpeg", "png")


class TongueCoatingAgent:
    def __init__(self, vision_invoker: VisionInvoker, system_prompt: str) -> None:
        self.vision_invoker = vision_invoker
        self.system_prompt_template = system_prompt

    def build_vision_messages(self, tongue_coating_result: str, confidence_score: str, image_path: Path) -> list:
        """按原请求结构构造 openai chat 消息；图片以 base64 data URL 内联。"""
        suffix = image_path.suffix.lstrip(".").lower()
        if suffix not in _SUPPORTED_IMAGE_TYPES:
            raise TrialError(
                f"不支持的图片格式: {suffix}（仅支持 jpg/jpeg/png）",
                "image_corrupt",
            )
        base64_bytes = base64.b64encode(image_path.read_bytes())
        data_url = f"data:image/{suffix};base64,{base64_bytes.decode('utf-8')}"
        return [
            {
                "role": "system",
                "content": self.system_prompt_template,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"以下是分类结果:{tongue_coating_result}\n和置信度:{confidence_score}%\n请根据这些信息进行诊断。\n",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]

    def llm_call(self, tongue_coating_result: str, confidence_score: str, image_path: Path) -> str:
        messages = self.build_vision_messages(tongue_coating_result, confidence_score, image_path)
        try:
            completion = self.vision_invoker.create(messages)
        except TrialError:
            raise
        except Exception as exc:
            raise TrialError(f"视觉诊断请求失败: {exc}", REASON_VISION_FAILURE) from exc
        return completion.content
