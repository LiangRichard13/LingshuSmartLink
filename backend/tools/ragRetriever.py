"""RAG 检索器：持久化目录、embedding 模型与 k 全部参数化。

导入本模块不构建索引、不导入 Chroma/GTE 重依赖；这些在构造时延迟导入。
返回结果保留文档身份（doc_id）、来源、排名与 score，供回执与 trace 使用。
"""

from __future__ import annotations

from replay_support import DocRecord, RetrieverBackend  # noqa: F401


class Retriever:
    def __init__(self, persist_directory: str, embedding_model, k: int = 2) -> None:
        from langchain_chroma import Chroma

        if k <= 0:
            raise ValueError("k 必须为正整数")
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.k = k
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
        )

    def get_relevant_documents(self, query: str) -> list:
        """返回按相似度排序的 DocRecord 列表（rank 从 1 开始）。"""
        docs_with_scores = self.db.similarity_search_with_score(query, k=self.k)
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
