"""报告生成 Agent：时间与输出位置注入，不再依赖 cwd 或墙钟。

meta 消息使用注入的固定报告时间；报告保存经由注入的 saver
（markdownTool 绑定输出目录与文件名）完成。
"""

from __future__ import annotations

from typing import Callable

from replay_support import Message, RoleInvoker


class ReportGenAgent:
    def __init__(
        self,
        meta_invoker: RoleInvoker,
        main_invoker: RoleInvoker,
        meta_template: str,
        main_template: str,
        report_time_text: str,
        saver: Callable,
    ) -> None:
        self.invoker = meta_invoker
        self.meta_invoker = meta_invoker
        self.main_invoker = main_invoker
        self.meta_template = meta_template
        self.main_template = main_template
        self.report_time_text = report_time_text
        self.saver = saver
        self.history: list = []

    def init_meta_data_system_message(
        self,
        patient_background_information: str,
        tongue_coating_image_path: str,
        tongue_coating_diagnosis: str,
    ) -> None:
        import os

        content = self.meta_template.format(
            patient_background_information=patient_background_information,
            date_time=self.report_time_text,
            tongue_coating_image_name="../images/" + os.path.basename(tongue_coating_image_path),
            tongue_coating_diagnosis=tongue_coating_diagnosis,
        )
        self.history = [Message("system", content)]
        self.invoker = self.meta_invoker

    def init_main_content_system_message(self, main_content: dict) -> None:
        content = self.main_template.format(main_content=main_content)
        self.history = [Message("system", content)]
        self.invoker = self.main_invoker

    def add_reference_to_history(self, index: int, reference_content: dict) -> None:
        if index == 0:  # 给出诊断与辨证结论
            reference_message = f"""
            {reference_content['terms_search_result']}\n
            如果内容中包含以上中医术语，请你在生成的markdown文档中添加包含以上中医术语的网络链接
            """
        elif index == 1:  # 给出基础调整方案
            reference_message = f"""
            {reference_content['keyword_search_result']}\n
            请在内容中添加以上内容以及内容的相关链接
            """
        elif index == 2:  # 给出中医治疗方案
            reference_message = f"""
            {reference_content['terms_search_result']}\n
            如果内容中包含以上中医术语，请你在生成的markdown文档中添加包含以上中医术语的网络链接\n
            {reference_content['herbs_search_result']}\n
            如果内容中包含以上中药材，请你在生成的markdown文档中添加包含以上中药材的网络链接\n
            {reference_content['herbs_image_search_result']}\n
            如果内容中包含以上中药材，请你在生成的markdown文档中添加包含以上中药材的图片链接,请按照![<药材名称>](<药材图片链接>)的格式添加\n
            {reference_content["acupoints_search_result"]}\n
            如果内容中包含以上穴位，请你在生成的markdown文档中添加包含以上穴位的网络链接\n
            """
        else:
            return
        self.history.append(Message("system", reference_message))

    def _extract_code_block(self, text: str) -> str:
        import re

        pattern = r"```(markdown|json)([\s\S]*?)```"
        if not re.search(pattern, text):
            return text
        return re.sub(pattern, r"\2", text)

    def get_content(self) -> str:
        response = self.invoker.invoke(self.history)
        return self._extract_code_block(response.content)

    def save_report(self, report_content: str) -> str:
        return self.saver(report_content)
