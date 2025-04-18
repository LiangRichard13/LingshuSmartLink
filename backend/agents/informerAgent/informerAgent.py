from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage,BaseMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
from agents.informerAgent.informer_config import system_message
import os
load_dotenv()

class InformerAgent:
    def __init__(self)->None:
        self.api_key=os.getenv("DEEP_SEEK_API_KEY") #从环境变量中获取API密钥
        self.llm=ChatOpenAI(base_url="https://api.deepseek.com",model="deepseek-chat",api_key=self.api_key) #用于创建LLM实例
        self.system_message=SystemMessage(content=system_message) #用于创建系统消息
        self.history=[self.system_message]  #用于存储对话历史, 使用系统消息初始化

    def init_message(self)->None:
        self.history=[self.system_message]  #使用系统消息初始化 

    def update_history(self,input_message:BaseMessage)->list[BaseMessage]:
        self.history.append(input_message) #更新对话历史
        return self.history #返回对话历史
    def get_response(self,input_message:HumanMessage)->AIMessage: #获取响应
        if self.history==[]:
            self.init_message()
        self.history=self.update_history(input_message)
        response=self.llm.invoke(self.history)
        self.update_history(response)
        return response