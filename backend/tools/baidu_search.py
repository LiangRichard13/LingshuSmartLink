import requests
import urllib.parse
from typing import List,Dict

class BaiduSearch:
    def __init__(self):
        self.api='https://baike.deno.dev'
    def baidu_search(self,query_list:list[str])->List[Dict[str,any]]:
        all_processed_search_result=[]
        # 通过 `词条名` 搜索并获取词条信息
        for item in query_list:
            item_url = f"{self.api}/item/{urllib.parse.quote(item)}"
            item_res = requests.get(item_url)
            item_data = item_res.json()
            if item_data['status']==200:
                all_processed_search_result.append(
                    {"item":item,"item_search_result":item_data['data']['link']}
                )
        return all_processed_search_result    
