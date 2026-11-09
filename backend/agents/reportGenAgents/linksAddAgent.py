"""链接搜索 Agent：JSON 解析失败不再静默返回空 dict，而是抛出明确异常。

搜索客户端由外部注入（fixture 或真实百度/Serper 组合），超时来自配置。
"""

from __future__ import annotations

import json

from replay_support import Message, RoleInvoker, SearchBackend, TrialLinksError


class LinksAddAgent:
    def __init__(
        self,
        invoker: RoleInvoker,
        system_message_template: str,
        search: SearchBackend,
    ) -> None:
        self.invoker = invoker
        self.system_message = Message("system", system_message_template)
        self.history = [self.system_message]
        self.search = search

    def extract_terms(self, main_content: str) -> dict:
        self.history.append(
            Message(
                "user",
                f"请你从以下内容中提取:{main_content}\n 下面请直接输出json格式内容，请直接输出可解析的json源码，不要用```json或```包裹输出。",
            )
        )
        response = self.invoker.invoke(self.history)
        self.history.append(Message("assistant", response.content))
        processed_content = self._extract_code_block(response.content)
        try:
            data = json.loads(processed_content)
            return {
                "terms": data["terms"],
                "herbs": data["herbs"],
                "acupoints": data["acupoints"],
                "keyword": data["keyword"],
            }
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise TrialLinksError(f"链接提取响应解析失败: {exc}") from exc

    @staticmethod
    def _extract_code_block(text: str) -> str:
        import re

        pattern = r"```(markdown|json)([\s\S]*?)```"
        if not re.search(pattern, text):
            return text
        return re.sub(pattern, r"\2", text)

    def add_links(self, main_content: str) -> dict:
        extracted = self.extract_terms(main_content)
        terms = extracted["terms"]
        herbs = extracted["herbs"]
        acupoints = extracted["acupoints"]
        keyword = extracted["keyword"]

        terms_search_result = self.search.search_terms(terms)
        herbs_search_result = self.search.search_herbs(herbs)
        herbs_image_search_result = self.search.search_herb_images(herbs)
        acupoints_search_result = self.search.search_acupoints(acupoints)
        keyword_search_result = self.search.search_keyword(keyword)

        return {
            "terms_search_result": f"以下是中医术语搜索结果:\n{terms_search_result}",
            "herbs_search_result": f"以下是中药材搜索结果:\n{herbs_search_result}",
            "herbs_image_search_result": f"以下是中药材图片链接的搜索结果:\n{herbs_image_search_result}",
            "acupoints_search_result": f"以下是穴位搜索结果:\n{acupoints_search_result}",
            "keyword_search_result": f"以下是关于\"{keyword}\"的搜索结果:\n{keyword_search_result}",
        }
