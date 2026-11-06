"""信息收集 Agent：历史语义与原实现一致，LLM 调用经注入的 invoker。

无 load_dotenv、无导入时服务创建。
"""

from __future__ import annotations

from replay_support import LLMResponse, Message, RoleInvoker  # noqa: F401 (RoleInvoker re-export)


class InformerAgent:
    def __init__(self, system_message: Message, invoker: RoleInvoker) -> None:
        self.invoker = invoker
        self.system_message = system_message
        self.history = [self.system_message]

    def init_message(self) -> None:
        self.history = [self.system_message]

    def update_history(self, input_message: Message) -> list:
        self.history.append(input_message)
        return self.history

    def get_response(self, input_message: Message) -> Message:
        if self.history == []:
            self.init_message()
        self.history = self.update_history(input_message)
        response = self.invoker.invoke(self.history)
        response_message = Message("assistant", response.content)
        self.update_history(response_message)
        return response_message
