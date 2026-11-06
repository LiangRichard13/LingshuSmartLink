"""任务规划 Agent：重试次数有界（planner_max_attempts），超限明确失败。

原实现为无限 while True 重试；本改造在注入的 max_attempts 耗尽后抛出
TrialPlannerError，由编排层记录失败原因码 planner_exhausted。
"""

from __future__ import annotations

from replay_support import Message, RoleInvoker, TrialPlannerError


class PlanAgent:
    def __init__(
        self,
        patient_background_information: str,
        invoker: RoleInvoker,
        system_prompt_template: str,
        treatment_plan: list,
        max_attempts: int,
    ) -> None:
        self.invoker = invoker
        self.treatment_plan = treatment_plan
        self.max_attempts = max_attempts
        self.system_message = Message(
            "system",
            system_prompt_template.format(
                patient_background_information=patient_background_information,
                treatment_plan=treatment_plan,
            ),
        )
        self.history = [self.system_message]

    def get_plan(self) -> list:
        attempts = 0
        while True:
            attempts += 1
            if attempts > self.max_attempts:
                raise TrialPlannerError(
                    f"任务规划在 {self.max_attempts} 次尝试内未得到与计划数量一致的任务列表"
                )
            response = self.invoker.invoke(self.history)
            response_message = Message("assistant", response.content)
            self.history.append(response_message)
            plan_tasks = response_message.content.split("<TASK_SPLIT>")
            if len(plan_tasks) == len(self.treatment_plan):
                return plan_tasks
            self.history.append(
                Message("user", f"任务数量应该为{len(self.treatment_plan)},请重新生成")
            )
