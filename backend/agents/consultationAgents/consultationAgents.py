from replay_support import LLMResponse, Message, RoleInvoker, VisionInvoker  # noqa: F401 (兼容再导出)


class BaseAgent:
    """协作基类：消息历史语义与原实现一致，LLM 调用经由注入的 invoker。"""

    def __init__(self, system_message: Message, invoker: RoleInvoker) -> None:
        self.system_message = system_message
        self.invoker = invoker
        self.init_messages()

    def init_messages(self) -> None:
        """将 stored_messages 置为只含系统消息的初始状态。"""
        self.stored_messages = [self.system_message]

    def update_message(self, message: Message) -> list:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(self, input_message: Message) -> Message:
        """追加输入、调用模型、记录输出并返回。"""
        messages = self.update_message(input_message)
        response = self.invoker.invoke(messages)
        output_message = Message("assistant", response.content)
        self.update_message(output_message)
        return output_message


class ActorAgent(BaseAgent):
    def __init__(
        self,
        patient_background_information: str,
        invoker: RoleInvoker,
        message_generator,
    ) -> None:
        system_message = message_generator.get_actor_sys_msg(patient_background_information)
        super().__init__(system_message=system_message, invoker=invoker)


class InstructorAgent(BaseAgent):
    def __init__(
        self,
        patient_background_information: str,
        invoker: RoleInvoker,
        message_generator,
    ) -> None:
        system_message = message_generator.get_instructor_sys_msg(patient_background_information)
        super().__init__(system_message=system_message, invoker=invoker)


class SummaryAgent(BaseAgent):
    def __init__(
        self,
        one_task_dialogue: list,
        task: str,
        invoker: RoleInvoker,
        message_generator,
    ) -> None:
        system_message = message_generator.get_summary_sys_msg(one_task_dialogue, task)
        super().__init__(system_message=system_message, invoker=invoker)
