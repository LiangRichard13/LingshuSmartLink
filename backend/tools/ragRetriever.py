from model.embedding_model.gteEmbeddings import GTEEmbeddings
from langchain.vectorstores import Chroma

class Retriever:
    def __init__(self, persist_directory:str='/home/ubuntu/MCM/workspace/RAG/vector_storage')->None:
        """
        初始化检索器
        
        参数:
            persist_directory: 向量存储的持久化目录路径
        """
        self.persist_directory = persist_directory
        self.embedding_model = GTEEmbeddings()
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        self.retriever = self.db.as_retriever()
    
    def get_relevant_documents(self, query:str)->list[str]:
        """
        获取与查询语义最接近的文档
        
        参数:
            query: 查询文本
            
        返回:
            相关文档的内容列表
        """
        results = self.retriever.get_relevant_documents(query)
        return [result.page_content for result in results]