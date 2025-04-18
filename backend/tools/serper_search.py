import http.client
import json
from dotenv import load_dotenv
from typing import List,Dict
import os 
load_dotenv()

class SerperSearch:
    def __init__(self)->None:
        self.api_key=os.getenv("SERPER_API_KEY")
    def serper_search(self,query_list:list[str])->List[Dict[str,any]]: # 用于执行serper批量搜索
        conn = http.client.HTTPSConnection("google.serper.dev")
        processed_query_list=[{
            "q": query,
            "gl": "cn",
            "hl": "zh-cn"
        } for query in query_list]
        payload = json.dumps(processed_query_list)
        headers = {
        'X-API-KEY': self.api_key,
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        processed_search_result=[]
        for search_result in json.loads(data.decode("utf-8")):
            processed_search_result.append(search_result['organic'][0])
        all_search_result=[{"item":item,"item_search_result":item_search_result} for item,item_search_result in zip(query_list,processed_search_result)]
        return all_search_result
    def serper_keyword_search(self,query:str)->List[Dict[str,any]]: # 用于执行serper关键字搜索
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps(
            {
            "q": query,
            "gl": "cn",
            "hl": "zh-cn"
            }
        )
        headers = {
        'X-API-KEY': self.api_key,
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        processed_search_result=[]
        response_data=json.loads(data.decode("utf-8"))
        for search_result in response_data['organic'][:5]: #获取前五个搜索结果
            processed_search_result.append(search_result)
        all_search_result=[{"item":query,"item_search_result":processed_search_result}]
        return all_search_result
    def serper_image_search(self,query_list:list[str])->List[Dict[str,any]]: # 用于执行serper图片搜索
        conn = http.client.HTTPSConnection("google.serper.dev")
        processed_query_list=[{
            "q": query,
            "gl": "cn",
            "hl": "zh-cn"
        } for query in query_list]
        payload = json.dumps(processed_query_list)
        headers = {
        'X-API-KEY': self.api_key,
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/images", payload, headers)
        res = conn.getresponse()
        data = res.read()
        processed_search_result=[]
        for search_result in json.loads(data.decode("utf-8")):
            processed_search_result.append(search_result['images'][0]['imageUrl'])
        all_search_result=[{"item":item,"item_image_search_result":item_search_result} for item,item_search_result in zip(query_list,processed_search_result)]
        return all_search_result