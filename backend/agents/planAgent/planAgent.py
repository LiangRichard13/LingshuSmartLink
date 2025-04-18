from agents.planAgent.plan_config import system_prompt_template,treatment_plan
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from tools.colorPrinter import ColorPrinter
import os
from dotenv import load_dotenv
load_dotenv()

class PlanAgent:
    def __init__(self,patient_background_information:str)->None:
        self.llm = ChatOpenAI(base_url="https://api.deepseek.com", model="deepseek-chat", api_key=os.getenv("DEEP_SEEK_API_KEY"))
        system_message_template=SystemMessagePromptTemplate.from_template(system_prompt_template)
        self.system_message=system_message_template.format_messages(
        patient_background_information=patient_background_information,
        treatment_plan=treatment_plan)[0]
        self.history=[self.system_message]

    def get_plan(self)->list[str]: #获取计划
        while True:
            response=self.llm.invoke(self.history)
            plan_tasks = response.content.split("<TASK_SPLIT>")
            if len(plan_tasks)==len(treatment_plan):
                break
            else:
                ColorPrinter.red("SystemMessage:")
                ColorPrinter.yellow("任务数量和计划不一致，正在重新生成")
                self.history.append(HumanMessage(content=f"任务数量应该为{len(treatment_plan)},请重新生成"))
                continue
        return plan_tasks