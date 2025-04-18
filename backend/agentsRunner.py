from agents.informerAgent.informerAgent import InformerAgent
from agents.tongueCoatingAgent.tongueCoatingAgent import TongueCoatingAgent
from agents.planAgent.planAgent import PlanAgent
from agents.consultationAgents.consultationAgents import InstructorAgent,ActorAgent,SummaryAgent
from agents.consultationAgents.consultation_config import chat_turn_limit
from agents.reportGenAgents.linksAddAgent import LinksAddAgent
from agents.reportGenAgents.reporGenAgent import ReportGenAgent
from tools.colorPrinter import ColorPrinter
from tools.ragRetriever import Retriever
from langchain.schema import HumanMessage,SystemMessage
from typing import List, Dict

class Runner:
    def run_tongue_coating(self, image_path:str)->str: # 运行舌苔诊断
        self.tongue_coating_agent = TongueCoatingAgent()
        return self.tongue_coating_agent.tongue_coating_diagnosis(image_path)
    def initial_rag_retriever(self)->None:
        ColorPrinter.red("SystemMessage: ")
        ColorPrinter.yellow("正在初始化RAG检索器...")
        self.retriever=Retriever()
        ColorPrinter.red("已加载RAG检索器")
    def run_informer(self,tongue_coating_diagnosis_content:SystemMessage)->str: # 运行InformerAgent,用于收集用户背景信息
        self.informer_agent = InformerAgent()
        self.informer_agent.history.append(tongue_coating_diagnosis_content)
        response=self.informer_agent.get_response(HumanMessage(content="你好！"))
        ColorPrinter.green("InformerAgent: ")
        ColorPrinter.white(response.content)
        while True:
            user_input = input(ColorPrinter.color_text("User: ", "blue")) 
            if user_input=="exit":
                break
            input_message=HumanMessage(content=user_input)
            response=self.informer_agent.get_response(input_message)
            ColorPrinter.green("InformerAgent: ")
            ColorPrinter.white(response.content)
            if "<INFORMER_TASK_DONE>" in response.content:
                return response.content
    def run_plan_agent(self,patient_background_information:str)->list[str]: # 运行PlanAgent
        self.plan_agent=PlanAgent(patient_background_information)
        plan_tasks=self.plan_agent.get_plan()
        return plan_tasks

    def run_actor_agent(self,input_message:str)->HumanMessage:
        actor_ai_msg=self.actor_agent.step(HumanMessage(content=input_message))
        actor_msg=HumanMessage(content=actor_ai_msg.content)
        ColorPrinter.green("ActorAgent: ")
        ColorPrinter.white(actor_msg.content)
        return actor_msg
    def run_instructor_agent(self,input_message:str)->HumanMessage:
        instructor_ai_msg=self.instructor_agent.step(HumanMessage(content=input_message)) 
        instructor_msg=HumanMessage(content=instructor_ai_msg.content)
        ColorPrinter.green("InstructorAgent: ")
        ColorPrinter.white(instructor_msg.content)
        return instructor_msg   
    def run_consultation_agents(self,patient_background_information:str,plan_tasks:list,rag_information:list[str])->List[Dict[str, str]]: # actor和instructor的对话
        self.actor_agent=ActorAgent(patient_background_information)
        self.actor_agent.stored_messages.append(SystemMessage(content=f"以下是参考资料:\n{rag_information},如果在执行任务过程中用到了以上资料，请引用参考资料原文"))
        
        self.instructor_agent=InstructorAgent(patient_background_information)
        
        summary_content_list=[] #用于每次任务后总结后的完整方案
        for task in plan_tasks:
            ColorPrinter.red("SystemMessage:")
            ColorPrinter.yellow(f"开始任务:{task}")
            dialogue_content=[] #用于每次任务的对话内容
            actor_msg=self.run_actor_agent(f"现在开始我们的任务:{task}")
            dialogue_content.append({"role":"actor","content":actor_msg.content})

            n=0 # 当前交互次数
            while n<chat_turn_limit:
                instructor_msg=self.run_instructor_agent(f"这是我的任务:{task}\n\n这是我的内容{actor_msg.content}\n\n")
                dialogue_content.append({"role":"instructor","content":instructor_msg.content})

                if "<CAMEL_TASK_DONE>" in instructor_msg.content:
                    ColorPrinter.red("SystemMessage:")
                    ColorPrinter.yellow(f"任务完成:{task}")
                    summary_content_list=self.add_to_summary_list(task,dialogue_content,summary_content_list)
                    break

                actor_msg=self.run_actor_agent(instructor_msg.content)
                dialogue_content.append({"role":"actor","content":actor_msg.content})
                n+=1
            
            #超过交互次数限制，结束任务
            if n>=chat_turn_limit:
                ColorPrinter.red("SystemMessage:")
                ColorPrinter.yellow(f"任务完成:{task}")
                summary_content_list=self.add_to_summary_list(task,dialogue_content,summary_content_list)
                
        consultation_result=[{"task":task,"answer":one_turn_content} for task,one_turn_content in zip(plan_tasks,summary_content_list)]
        ColorPrinter.red("SystemMessage:")
        ColorPrinter.yellow(f"以下是所有任务执行内容:{str(consultation_result)}")
        return consultation_result
    def run_summary_agent(self,one_task_dialogue:List[Dict[str,str]],task:str)->str: 
        self.summary_agent=SummaryAgent(one_task_dialogue,task)
        input_message=HumanMessage(content=f"请输出:")
        summary_content=self.summary_agent.step(input_message)
        return summary_content.content
    def add_to_summary_list(self,task:str,dialogue_content:List[Dict[str,str]],summary_content_list:list[str])->list[str]:
        # self.actor_agent.init_messages() # actor_agent不需要清空对话，因为之后的任务需要前面的对话作为参考
        self.instructor_agent.init_messages()
        summary_content=self.run_summary_agent(dialogue_content,task)
        ColorPrinter.green("SummaryAgent: ")
        ColorPrinter.white(summary_content)
        summary_content_list.append(summary_content)
        return summary_content_list
    def run_links_add_agent(self,consultation_result:List[Dict[str,str]])->dict: 
        self.links_add_agent=LinksAddAgent()
        main_content=""
        for content in consultation_result:
            main_content += content["answer"]+"\n"
        search_reference=self.links_add_agent.add_links(main_content)
        return search_reference

    def run_report_gen_agent(
            self,
            patient_background_information:str, # 患者背景信息
            main_content:List[Dict[str,str]], # 主要内容
            tongue_coating_image_path:str, # 舌苔图片路径
            tongue_coating_diagnosis:str, # 舌苔诊断结果
            search_reference:Dict[str,str] # 搜索参考
            )->str: # 运行ReportGenAgent
        self.report_gen_agent=ReportGenAgent()
        self.report_gen_agent.init_meta_data_system_message(
            patient_background_information=patient_background_information,
            tongue_coating_image_path=tongue_coating_image_path,
            tongue_coating_diagnosis=tongue_coating_diagnosis
        )
        report_content=self.report_gen_agent.get_content() # 生成报告的第一步：患者的背景信息、舌苔诊断等
        ColorPrinter.green("ReportGenAgent:")
        ColorPrinter.white(report_content)
        for index, content in enumerate(main_content):
            self.report_gen_agent.init_main_content_system_message(main_content=content)
            self.report_gen_agent.add_reference_to_history(index,search_reference)
            new_report_content=self.report_gen_agent.get_content() # 生成报告的其他步骤报告中的其他内容
            ColorPrinter.green("ReportGenAgent:")
            ColorPrinter.white(new_report_content)
            report_content = report_content+'\n\n'+ new_report_content
        save_path=self.report_gen_agent.save_report(report_content)
        return save_path
    def run_chain(self)->str: # 运行整个链路
        # 初始化rag检索器
        self.initial_rag_retriever()

        ColorPrinter.green("TongueCoatingAgent:")
        ColorPrinter.white("请输入舌苔图片路径:")
        image_path = input(ColorPrinter.color_text("User: ", "blue"))
        
        # 运行舌苔诊断
        tongue_coating_diagnosis_content=self.run_tongue_coating(image_path)
        ColorPrinter.green("TongueCoatingAgent:")
        ColorPrinter.white(tongue_coating_diagnosis_content)

        # 运行InformerAgent
        patient_background_information=self.run_informer(
        SystemMessage(
            content=f"""
            以下是用户上传的舌苔图片的诊断结果：{tongue_coating_diagnosis_content}\n
            在经过多轮对话后，你需要在患者信息描述中加入患者舌苔的特征和可能的健康问题(结合患者信息进一步缩小范围)。\n
            """
            ))
        ColorPrinter.red("SystemMessage:")
        ColorPrinter.yellow(f"PatientBackgroundInformation:{patient_background_information}")

        # 运行rag检索
        ColorPrinter.red("SystemMessage:")
        ColorPrinter.yellow("正在RAG检索相关参考资料...")
        rag_information=self.retriever.get_relevant_documents(query=patient_background_information)
        ColorPrinter.red("SystemMessage:")
        ColorPrinter.yellow(f"已检索到以下相关资料:{rag_information}")

        # 运行PlanAgent
        plan_tasks=self.run_plan_agent(patient_background_information)
        ColorPrinter.green("PlanAgent:")
        ColorPrinter.white(str(plan_tasks))

        # 运行actor和instructor的对话
        consultation_result=self.run_consultation_agents(patient_background_information,plan_tasks,rag_information)
        
        # from testData import patient_background_information,tongue_coating_diagnosis_content,consultation_result

        ColorPrinter.red("SystemMessage:")
        ColorPrinter.yellow("正在网络搜索参考内容...")
        search_reference=self.run_links_add_agent(consultation_result=consultation_result)
        ColorPrinter.green("LinksAddAgent:")
        ColorPrinter.white(str(search_reference))

        ColorPrinter.red("SystemMessage:")
        ColorPrinter.yellow("正在生成报告...")
        # 生成报告
        save_path=self.run_report_gen_agent(
            patient_background_information=patient_background_information,
            main_content=consultation_result,
            tongue_coating_image_path=image_path,
            tongue_coating_diagnosis=tongue_coating_diagnosis_content,
            search_reference=search_reference
            )
        return save_path
    
if __name__ == "__main__":
    runner = Runner()
    runner.run_chain()