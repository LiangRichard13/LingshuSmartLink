"""Serper 搜索客户端：API key、超时经构造器注入；无导入时副作用。"""

from __future__ import annotations

import http.client
import json
from typing import List


class SerperSearch:
    def __init__(self, api_key: str, timeout_seconds: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _post(self, endpoint: str, payload) -> bytes:
        conn = http.client.HTTPSConnection("google.serper.dev", timeout=self.timeout_seconds)
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            }
            conn.request("POST", endpoint, json.dumps(payload), headers)
            res = conn.getresponse()
            return res.read()
        finally:
            conn.close()

    def serper_search(self, query_list: list) -> List[dict]:
        payload = [{"q": query, "gl": "cn", "hl": "zh-cn"} for query in query_list]
        data = json.loads(self._post("/search", payload).decode("utf-8"))
        processed = [entry["organic"][0] for entry in data]
        return [
            {"item": item, "item_search_result": result}
            for item, result in zip(query_list, processed)
        ]

    def serper_keyword_search(self, query: str) -> List[dict]:
        payload = {"q": query, "gl": "cn", "hl": "zh-cn"}
        data = json.loads(self._post("/search", payload).decode("utf-8"))
        processed = [
            {"title": entry.get("title", ""), "link": entry.get("link", "")}
            for entry in data["organic"][:5]
        ]
        return [{"item": query, "item_search_result": processed}]

    def serper_image_search(self, query_list: list) -> List[dict]:
        payload = [{"q": query, "gl": "cn", "hl": "zh-cn"} for query in query_list]
        data = json.loads(self._post("/images", payload).decode("utf-8"))
        processed = [entry["images"][0]["imageUrl"] for entry in data]
        return [
            {"item": item, "item_image_search_result": result}
            for item, result in zip(query_list, processed)
        ]
