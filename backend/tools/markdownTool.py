"""Markdown 保存工具：输出目录、文件名与时钟全部注入，不再依赖 cwd。

trial 链路传入固定输出目录与固定文件名（report.md）；无注入时保留
原行为（cwd/static/report/markdown_<时间戳>.md）供 legacy 交互路径使用。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from tools.colorPrinter import ColorPrinter


class markdownTool:
    def __init__(self, output_dir: str | None = None, now: datetime | None = None) -> None:
        if output_dir is not None:
            self.file_path = str(Path(output_dir))
            self.now = now or datetime.now()
        else:
            self.file_path = os.getcwd() + "/static/report"
            self.now = None

    def save_markdown(self, markdown_content: str, filename: str | None = None) -> str:
        os.makedirs(self.file_path, exist_ok=True)
        if filename is None:
            timestamp = (self.now or datetime.now()).strftime("%Y%m%d_%H%M%S")
            filename = f"markdown_{timestamp}.md"
        filepath = os.path.join(self.file_path, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as handle:
                handle.write(markdown_content)
            ColorPrinter.yellow("SystemMessage:")
            ColorPrinter.red(f"Markdown文件已成功保存到: {os.path.abspath(filepath)}")
            return filepath
        except Exception as exc:  # legacy 行为：返回错误文本而不是抛出
            return f"保存报告文件时出错: {exc}"
