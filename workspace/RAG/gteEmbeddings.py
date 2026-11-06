"""GTE 中文句子 embedding（ModelScope pipeline）。

本模块只在 local/live 真实模式下被导入（fixture 链路不导入）。
langchain_core 的 Embeddings 基类在模块顶层导入；较重的 modelscope
在构造时延迟导入。模型路径参数化，不再硬编码 /home/ubuntu/...。
"""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings


class GTEEmbeddings(Embeddings):
    def __init__(self, model_path: str, sequence_length: int = 512) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.model_path = model_path
        self.sequence_length = sequence_length
        self.pipeline = pipeline(
            Tasks.sentence_embedding,
            model=self.model_path,
            sequence_length=self.sequence_length,
        )

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """批量文本嵌入，加入批次控制。"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = {"source_sentence": batch_texts}
            result = self.pipeline(input=inputs)
            results.extend([vec.tolist() for vec in result["text_embedding"]])
        return results

    def embed_query(self, text: str) -> List[float]:
        """查询单条文本嵌入。"""
        return self.embed_documents([text])[0]
