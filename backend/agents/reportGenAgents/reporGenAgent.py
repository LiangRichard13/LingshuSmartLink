from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv
from agents.reportGenAgents.reportGen_config import meta_data_system_prompt_template,main_content_system_prompt_template
from tools.markdownTool import markdownTool
from tools.process_code_blocks import check_and_process_code_blocks
from typing import List, Dict
load_dotenv()

class ReportGenAgent:
    def __init__(self)->None:
        self.llm = ChatOpenAI(base_url="https://api.deepseek.com", model="deepseek-chat", api_key=os.getenv("DEEP_SEEK_API_KEY"))
        self.history = []
    def init_meta_data_system_message(self,patient_background_information:str, tongue_coating_image_path:str, tongue_coating_diagnosis:str)->None:
        meta_data_system_message = SystemMessage(content=meta_data_system_prompt_template.format(
            patient_background_information=patient_background_information,
            date_time=datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒"),
            tongue_coating_image_name='../images/'+os.path.basename(tongue_coating_image_path),
            tongue_coating_diagnosis=tongue_coating_diagnosis
        ))
        self.history = [meta_data_system_message]
    def init_main_content_system_message(self,main_content:List[Dict[str,str]])->None:
        main_content_system_message= SystemMessage(content=main_content_system_prompt_template.format(
            main_content=main_content
        ))
        self.history = [main_content_system_message]
    def add_reference_to_history(self,index,reference_content:Dict[str,str])->None:
        if index==0: # 给出诊断与辨证结论
            reference_message=f"""
            {reference_content['terms_search_result']}\n
            如果内容中包含以上中医术语，请你在生成的markdown文档中添加包含以上中医术语的网络链接
            """
        elif index==1:# 给出基础调整方案
            reference_message=f"""
            {reference_content['keyword_search_result']}\n
            请在内容中添加以上内容以及内容的相关链接
            """
        elif index==2: # 给出中医治疗方案   
            reference_message=f"""
            {reference_content['terms_search_result']}\n
            如果内容中包含以上中医术语，请你在生成的markdown文档中添加包含以上中医术语的网络链接\n
            {reference_content['herbs_search_result']}\n
            如果内容中包含以上中药材，请你在生成的markdown文档中添加包含以上中药材的网络链接\n
            {reference_content['herbs_image_search_result']}\n
            如果内容中包含以上中药材，请你在生成的markdown文档中添加包含以上中药材的图片链接,请按照![<药材名称>](<药材图片链接>)的格式添加\n
            {reference_content["acupoints_search_result"]}\n
            如果内容中包含以上穴位，请你在生成的markdown文档中添加包含以上穴位的网络链接\n
            """
        else:
            return
        reference_system_message=SystemMessage(content=reference_message)
        self.history.append(reference_system_message)    
    def get_content(self)->str:
        response = self.llm.invoke(self.history)
        report_content = response.content
        processed_report_content = check_and_process_code_blocks(report_content, action="extract")
        return processed_report_content
    def save_report(self, report_content:str)->str:
        markdowntool=markdownTool()
        save_path=markdowntool.save_markdown(report_content)
        return save_path