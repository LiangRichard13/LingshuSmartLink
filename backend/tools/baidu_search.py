"""百度百科词条搜索客户端：API 地址与超时经构造器注入；无导入时副作用。"""

from __future__ import annotations

import urllib.parse
from typing import List

import requests


class BaiduSearch:
    def __init__(self, api: str = "https://baike.deno.dev", timeout_seconds: float = 30.0) -> None:
        self.api = api
        self.timeout_seconds = timeout_seconds

    def baidu_search(self, query_list: list) -> List[dict]:
        all_processed_search_result = []
        for item in query_list:
            item_url = f"{self.api}/item/{urllib.parse.quote(item)}"
            item_res = requests.get(item_url, timeout=self.timeout_seconds)
            item_data = item_res.json()
            if item_data.get("status") == 200:
                all_processed_search_result.append(
                    {"item": item, "item_search_result": item_data["data"]["link"]}
                )
        return all_processed_search_result
