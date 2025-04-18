from typing import List, Union
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from langchain_core.embeddings import Embeddings

class GTEEmbeddings(Embeddings):
    def __init__(self)->None:
        self.model_path="/home/ubuntu/MCM/backend/model/embedding_model/nlp_gte_sentence-embedding_chinese-base"
        self.pipeline = pipeline(Tasks.sentence_embedding,
                       model=self.model_path,
                       sequence_length=512,
                       )

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """批量文本嵌入，加入批次控制"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = {"source_sentence": batch_texts}
            result = self.pipeline(input=inputs)
            results.extend([vec.tolist() for vec in result["text_embedding"]])
        return results

    def embed_query(self, text: str) -> List[float]:
        """查询单条文本嵌入"""
        return self.embed_documents([text])[0]


