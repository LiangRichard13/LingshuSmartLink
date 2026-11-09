"""协作角色结构常量。

提示模板已迁移到 trial/prompts/*.txt（actor.txt / instructor.txt /
summary.txt），由试用配置注入运行时；本模块不再保存模板文本，
也不再在导入时读取环境变量。
"""

chat_turn_limit = 5  # 每任务 Instructor 调用次数上限（legacy 默认值；试用以配置为准）

actor_role_name = "中医医生"
instructor_role_name = "任务指导者"
