from typing import List

from replay_support import Message


class SystemMessageGenerator:
    """协作角色模板渲染器；模板文本由配置注入（单一来源：trial/prompts/*.txt）。

    渲染行为与原 langchain SystemMessagePromptTemplate 一致：
    str.format 以 actor_role_name/instructor_role_name/patient_background_information 填充。
    """

    def __init__(
        self,
        actor_template: str,
        instructor_template: str,
        summary_template: str,
        actor_role_name: str,
        instructor_role_name: str,
    ) -> None:
        self.actor_role_name = actor_role_name
        self.instructor_role_name = instructor_role_name
        self.actor_template = actor_template
        self.instructor_template = instructor_template
        self.summary_template = summary_template

    def get_actor_sys_msg(self, patient_background_information: str) -> Message:
        content = self.actor_template.format(
            actor_role_name=self.actor_role_name,
            instructor_role_name=self.instructor_role_name,
            patient_background_information=patient_background_information,
        )
        return Message("system", content)

    def get_instructor_sys_msg(self, patient_background_information: str) -> Message:
        content = self.instructor_template.format(
            actor_role_name=self.actor_role_name,
            instructor_role_name=self.instructor_role_name,
            patient_background_information=patient_background_information,
        )
        return Message("system", content)

    def get_summary_sys_msg(self, one_task_dialogue: List[dict], task: str) -> Message:
        content = self.summary_template.format(
            one_task_dialogue=one_task_dialogue,
            task=task,
        )
        return Message("system", content)
