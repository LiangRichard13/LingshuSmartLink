from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage,HumanMessage
from agents.reportGenAgents.reportGen_config import links_add_agent_system_message
from tools.process_code_blocks import check_and_process_code_blocks
from tools.serper_search import SerperSearch
from tools.baidu_search import BaiduSearch
from tools.colorPrinter import ColorPrinter
from typing import Dict
from dotenv import load_dotenv
import json
import os
load_dotenv()

class LinksAddAgent:
    def __init__(self)->None:
        self.llm = ChatOpenAI(base_url="https://api.deepseek.com", model="deepseek-chat", api_key=os.getenv("DEEP_SEEK_API_KEY"))
        self.system_message=SystemMessage(content=links_add_agent_system_message)
        self.history=[self.system_message]

    def add_links(self,main_content:str)->Dict[str,str]:
        self.history.append(HumanMessage(content=f"请你从以下内容中提取:{main_content}\n 下面请直接输出json格式内容，请直接输出可解析的json源码，不要用```json或```包裹输出。"))
        response=self.llm.invoke(self.history)
        response_content=response.content
        processed_content=check_and_process_code_blocks(text=response_content,action="extract")
        try:
            # 解析JSON字符串为Python字典
            data = json.loads(processed_content)

            # 提取herbs、acupoints和keywords
            terms=data['terms'] # 获取terms列表
            herbs = data['herbs']  # 获取herbs列表
            acupoints = data['acupoints']  # 获取acupoints列表
            keyword = data['keyword']  # 获取keywords字符串

            serper_searcher=SerperSearch()
            baidu_searcher=BaiduSearch()

            # terms_search_result=serper_searcher.serper_search(terms)
            # herbs_search_result=serper_searcher.serper_search(herbs)

            terms_search_result=baidu_searcher.baidu_search(terms)
            herbs_search_result=baidu_searcher.baidu_search(herbs)
            
            herbs_image_search_result=serper_searcher.serper_image_search(herbs)
            
            # acupoints_search_result=serper_searcher.serper_search(acupoints)
            
            acupoints_search_result=baidu_searcher.baidu_search(acupoints)
            
            keyword_search_result=serper_searcher.serper_keyword_search(keyword)

            all_search_result={
                "terms_search_result":f"以下是中医术语搜索结果:\n{terms_search_result}",
                "herbs_search_result":f"以下是中药材搜索结果:\n{herbs_search_result}",
                "herbs_image_search_result":f"以下是中药材图片链接的搜索结果:\n{herbs_image_search_result}",
                "acupoints_search_result":f"以下是穴位搜索结果:\n{acupoints_search_result}",
                "keyword_search_result":f"以下是关于\"{keyword}\"的搜索结果:\n{keyword_search_result}"
                }

            return all_search_result
        
        except Exception as e:
            ColorPrinter.red("SystemMessage:")
            ColorPrinter.yellow(f"解析响应结果出错: {e}")
            return {}
