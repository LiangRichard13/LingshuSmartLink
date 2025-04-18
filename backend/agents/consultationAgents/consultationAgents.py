from langchain_openai import ChatOpenAI
from agents.consultationAgents.systemMessageGen import SystemMessageGenerator
from langchain.schema import SystemMessage,BaseMessage,HumanMessage,AIMessage
import os
from dotenv import load_dotenv
from typing import List,Dict
load_dotenv()

class BaseAgent:
    def __init__(self,system_message:SystemMessage)->None:
        self.system_message=system_message
        self.api_key=os.getenv("DEEP_SEEK_API_KEY") #从环境变量中获取API密钥
        self.llm=ChatOpenAI(base_url="https://api.deepseek.com",model="deepseek-chat",api_key=self.api_key) #用于创建LLM实例
        self.init_messages()
    
    def init_messages(self)->None:
        """
        初始化对话消息

        将stored_messages置为空列表并只添加最初的系统消息
        """
        self.stored_messages=[self.system_message]

    def update_message(self,message:BaseMessage)->list[BaseMessage]:
        """
        更新对话消息列表
        
        将新消息加入到stored_messages中
        """
        self.stored_messages.append(message)
        return self.stored_messages
    
    def step(self,input_message:HumanMessage)->AIMessage:
        """
        与大模型进行交互
        
        中间会将输入和输出存入到stored_messages中
        最后返回大模型的输出
        """
        messages=self.update_message(input_message)
        output_message=self.llm.invoke(messages)
        self.update_message(output_message)
        return output_message

class ActorAgent(BaseAgent):   
    def __init__(self,patient_background_information:str)->None:
        systemMessageGenerator=SystemMessageGenerator()
        actor_sys_msg=systemMessageGenerator.get_actor_sys_msg(patient_background_information) #生成actor的初始系统消息
        super().__init__(system_message=actor_sys_msg) #调用父类的构造函数

class InstructorAgent(BaseAgent):   
    def __init__(self,patient_background_information:str)->None:
        systemMessageGenerator=SystemMessageGenerator()
        instructor_sys_msg=systemMessageGenerator.get_instructor_sys_msg(patient_background_information) #生成instructor的初始系统消息
        super().__init__(system_message=instructor_sys_msg) #调用父类的构造函数

class SummaryAgent(BaseAgent):
    def __init__(self,one_task_dialogue:str,task:str)->None:
        systemMessageGenerator=SystemMessageGenerator()
        summary_sys_msg=systemMessageGenerator.get_summary_sys_msg(one_task_dialogue,task) #生成summary的初始系统消息
        super().__init__(system_message=summary_sys_msg) #调用父类的构造函数