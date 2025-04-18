#DirectoryLoader用于加载目录下的文档，用于从指定目录中加载所有支持的文件(如.txt、.pdf等)
from langchain_community.document_loaders import DirectoryLoader,TextLoader

#CharacterTextSplitter用于文档切分，确保文档被切成适合嵌入和检索的大小
# from langchain.text_splitter import CharacterTextSplitter

#RecursiveCharacterTextSplitter适合保留段落或句子结构的文本
from langchain.text_splitter import RecursiveCharacterTextSplitter

#引入Chroma向量库，Chroma是一个轻量级的向量存储库，用于保存和检索向量
from langchain.vectorstores import Chroma

from gteEmbeddings import GTEEmbeddings

def load_documents(directory:str)->list:
    '''
    用于加载和分块文档
    参数: directory 用于rag的文档目录路径
    返回: 分块后的文档列表
    '''
    print('开始加载')
    
    #实例化一个loader通过load()方法加载目录中的文件，返回一个Document对象列表
    loader = DirectoryLoader(
        path=directory,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}  # 或者试试 "gbk"
        )
    documents = loader.load()
    print("文档数量:",len(documents))
    print("加载的第一个document对象:\n",documents[0])
    
    print('开始切分文档:')

    # 设置切分的大小和切分的重叠部分,块之间重叠 64 个字符，避免信息断裂保持语义连贯性
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, add_start_index=True, separators=["<目录>"])
    split_docs = text_splitter.split_documents(documents)

    #检查分块数量
    print(f"分块后的文档快数量: {len(split_docs)}")

    print(f"第一个切块:{split_docs[0].page_content}")

    return split_docs # 返回切分后的分块

def load_embedding_model():
    '''
    用于加载来自openai的embedding model
    返回一个OpenAIEmbeddings的实例
    '''
    embeddings=GTEEmbeddings()
    return embeddings

def store_chroma(docs,embeddings,persist_directory='/home/ubuntu/MCM/workspace/RAG/vector_storage'):
    '''
    1.将分块后的文本数据通过embedding model进行嵌入生成向量
    2.将向量数据持久化到磁盘
    3.返回Chroma数据库对象以供后续使用
    '''
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
    #这里的向量数据应该和分块后的文档块数量一致
    print(f"数据库中的向量数据: {db._collection.count()}")
    return db

# 引入embedding模型
embeddings=load_embedding_model()
print("embedding模型测试：",embeddings.embed_query("测试文本"))

# 切分文档块
chunks=load_documents(directory="/home/ubuntu/MCM/workspace/RAG/rag_data")
#做嵌入后存储到向量数据库,返回db对象
db=store_chroma(chunks,embeddings)
print("向量数据库构建完毕")