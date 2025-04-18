from agents.consultationAgents.consultation_config import (
actor_role_name,
instructor_role_name,
actor_inception_prompt,
instructor_inception_prompt,
summary_inception_prompt
)
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import SystemMessage
from typing import List,Dict

class SystemMessageGenerator:
    def __init__(self):
        self.actor_role_name=actor_role_name
        self.instructor_role_name=instructor_role_name 
        self.actor_inception_prompt=actor_inception_prompt
        self.instructor_inception_prompt=instructor_inception_prompt
        self.summary_inception_prompt=summary_inception_prompt

    def get_actor_sys_msg(self,patient_background_information:str)->SystemMessage:  #生成actor的初始系统消息
        actor_sys_template = SystemMessagePromptTemplate.from_template(template=self.actor_inception_prompt)
        actor_sys_msg=actor_sys_template.format_messages(
            actor_role_name=self.actor_role_name,
            instructor_role_name=self.instructor_role_name,
            patient_background_information=patient_background_information
        )[0]
        return actor_sys_msg
    
    def get_instructor_sys_msg(self,patient_background_information:str)->SystemMessage: #生成instructor的初始系统消息
        instructor_sys_template = SystemMessagePromptTemplate.from_template(template=self.instructor_inception_prompt)
        instructor_sys_msg=instructor_sys_template.format_messages(
            actor_role_name=self.actor_role_name,
            instructor_role_name=self.instructor_role_name,
            patient_background_information=patient_background_information
        )[0]
        return instructor_sys_msg

    def get_summary_sys_msg(self,one_task_dialogue:List[Dict[str,str]],task:str)->SystemMessage: #生成summary的初始系统消息
        summary_sys_template = SystemMessagePromptTemplate.from_template(template=self.summary_inception_prompt)
        summary_sys_msg=summary_sys_template.format_messages(
            one_task_dialogue=one_task_dialogue,
            task=task
        )[0]
        return summary_sys_msg