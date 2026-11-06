# LingshuSmartLink × ReproLLM 单病例试用记录（fixture 层）

日期：2026-11-06（Asia/Shanghai）。执行者：贡献者本人（梁晨诞，liangcd@sscenter.sh.cn）与
编码代理协作完成；代理执行了全部命令与代码改造，人工审阅了关键产物。本文如实区分
「已完成 / 已观察 / BLOCKED」，不把计划、mock 通过或 child 成功表述为科学或临床效果。

## 1. 身份与起点

| 项 | 值 |
|---|---|
| Lingshu 分支 | `codex/single-case-replay`（自 `2d6f839c9d2400bfe9f836bd400085971c9a511f` 固定起点） |
| 下游改造 SHA | `2e97ca3`（fixture baseline 最终态；完整链见 `git log`） |
| ReproLLM 来源 | `EnumaElish123/ReproLLM` @ `dfb72eda57abb47cc0fc972982e770c6199c5bd2`（与当时 upstream main 一致） |
| ReproLLM 安装来源 | 本地构建 wheel `reprollm-0.7.0-py3-none-any.whl`，`sha256:ca65c2b2f99487746ffc6222ae75d14a793e243deaa4ab5c7799b605b397ef09`（来源记录 `lingshu-trial-evidence/private/reprollm-source.json`） |
| 试用水解释释器 | CPython 3.12.13（uv 管理）；`.venv-trial` 仅含 reprollm 运行时 + pytest 8.3.5，**不含 torch/langchain/openai/modelscope** |
| 环境清单 | `trial/environment.json`（importlib.metadata 生成，标注研究进程；不含解释器绝对路径） |
| 执行环境偏差 | 计划假设 Linux；实际为 Windows 10（Git Bash + uv Windows venv `Scripts/` 布局）。命令语义等价，路径适配已记录 |

第 2 步的 fork（`gh auth login --web`）BLOCKED：需要贡献者交互式登录。本试用改用
上游固定 SHA 直接只读克隆（`ReproLLM-contrib`），origin/upstream 验收项未满足，
后续 PR 流程在登录后按计划第 2/16 步补做。

## 2. 已完成：fixture 层单病例链路

**边界声明：以下所有结论仅适用于 fixture 模式**——分类/视觉/RAG/搜索全部是
人工构造的确定响应（cassette 明确标记 `synthetic:true`，不是历史录制），
LLM 为脚本响应。它证明的是：原业务编排被真实执行、配置真实被消费、
事件与产物按契约落盘、ReproLLM 记录与实际一致。**不构成任何模型质量或医疗效果证明。**

### 2.1 下游改造（LS01–LS09）

- LS01：`backend/agentsRunner.py` 的 Runner 接收 config/services/root；`run_case(case, image_path, output_dir)`
  按原顺序执行八阶段；`run_chain()` 仅保留交互适配并调用同一编排。
- LS02/LS03：`backend/replay_case.py`（三模式 CLI、stdin 关闭可运行、越界路径拒绝、
  非空输出目录拒绝）与 `backend/replay_support.py`（严格配置、服务协议、Fixture/Real
  工厂、注入时钟、全局预算、trace/call 录制器）。
- LS04–LS08：各 Agent 构造器注入 client/参数/模板/时钟；模板迁移至 `trial/prompts/*.txt`
  作为运行时单一来源（`*_config.py` 仅留结构性常量）；视觉模型只来自配置
  （消除原 `self.model` 与 `create(model=...)` 硬编码不一致）；RAG 检索器 k/路径参数化，
  返回 doc_id/source/rank/score；`processed_rag_data.py` 改为带 main guard 的确定性 CLI
  （`--corpus --embedding-model --persist-directory --manifest --chunk-size --chunk-overlap --separator`）。
- LS09：`tools/trial_tools.py` 七个子命令（seed/prepare/env/verify/archive/probe-pth/package）
  与 `tests/replay/` 20 项验收测试，全部离线通过（`pytest -q tests/replay -m "not local and not live"`）。

与原实现的有意差异（单独记录，不冒充“完全相同”）：分类从视觉 Agent 拆为独立阶段
（调用顺序不变，便于回执）；交互 Informer 以 `exit` 结束时按明确失败处理
（原实现返回 None 并在后续崩溃）；LLM/搜索重试不再无限（planner 有界、SDK retries=0）。

### 2.2 基线、漂移矩阵与负例的命令回执

所有命令的 stdout/stderr/退出码/耗时保存在本机
`lingshu-trial-evidence/private/`（原始 L0 基线、`capture` 回执、各变体 diff、
负例 expected/actual、归档副本）。要点：

- 原始 L0 基线（未改代码前）：audit exit 0，13 findings（4 WARNING / 2 INFO / 7 PASS）。
- 首次 fixture run `lingshu-fixture-a`（`20261106T070328Z-0ccdef`）：child exit 0，
  八阶段全部 success，模型调用 59/80（chat 58 + vision 1），search 5 次；
  `trial_tools verify --scenario baseline` 18 项检查全 PASS（bindings 观察
  chat_turn_limit=5 / rag_k=2 / temperature=0.2 与配置一致；模型来自配置；
  artifact hash 与实际输出一致；trace 129 事件连续且单源 fixture）。
- 同输入重复（干净树）：报告 sha256 逐字节相同；`diff --fail-on HIGH` exit 0，
  仅 1 条 MEDIUM（`code.commit`——两次 run 之间有一次工具修复提交，可解释）+ 4 条 NONE 元数据。
- 正向漂移矩阵（对 baseline A 的 diff，`--fail-on HIGH` 预期 exit 1）：

| 场景 | 观察到的 HIGH | 伴随变化 |
|---|---|---|
| rounds-3 | `custom.workflow.chat_turn_limit` 5→3 | `files.trial/configs/current.yaml.sha256` |
| prompt-v2 | `prompts.actor.sha256`、`files.trial/prompts/actor.txt.sha256` | manifest 实验名/code.commit |
| image-v2-c | `files.trial/cases/case-001.png.sha256` | 同上 |
| retrieval-reordered-c | `files.trial/fixtures/retrieval.json.sha256` | 同上 |
| search-v2-c | `files.trial/fixtures/external.json.sha256` | 同上 |

  spy/回执证明消费侧真实变化：rounds-3 的 Instructor 实际调用 12 次（4 任务×3）；
  prompt-v2 的 Actor 请求摘要改变；image-v2 的消费图片 hash 改变；retrieval-reordered
  的下游 Actor 上下文中文档顺序 B,A,C；search-v2 的修订文本进入报告输入。
- 诚实记录的过程污染：prompt-v2 之后的三个变体首跑（已归档为
  `fixture-image-v2` / `fixture-retrieval-reordered` / `fixture-search-v2`）携带了
  actor 模板残留的伴生 HIGH——`actor.txt` 自映射导致 v2 模板残留。修复
  （`actor-base.txt` 基线源，且必须取 run 快照字节而非 `git show` blob，行尾差异）
  后以 `-c` 后缀重跑，矩阵以 `-c` 归档为准。
- 负例 N01–N08 全部 PASS（一次性副本执行，expected/actual 与完整日志在
  `private/negatives/`）：N01 删除绑定字段→规则 CRITICAL、audit exit 1；
  N02 manifest 5/配置 3→spy 实际消费 3（Instructor 12 次）+ `consistency.custom_fields`
  CRITICAL；N03a 只改 prompt 字节→`lock --check` 仍 0 而 audit 发现漂移
  （lock 只比较 manifest/project-rules 摘要，不扫描文件字节）；N03b 改 manifest 不重
  lock→`lock --check` exit 1；N04 删图→child exit 2、无报告、失败 run 保留；N05 缺
  fixture 键→明确 `fixture_key_missing`；N06 预算 3→used==3、第 4 次请求未发出；
  N07 搜索超时→已完成阶段保留、`search_failed`、非零退出；N07b 秘密 canary 被
  package 拒绝并回滚；N08 stdin 传图→run 记录无 child 收图证据（仅 manifest declared），
  argv 入口后 `command.argv` 直接可观察。
- .pth 格式探针（`private/pth-probe/`，与模型效果无关，纯格式行为）：
  仅含 .pth 的目录 `weights.total_size_bytes=0`、`hashed=true` 且 A→B 空集合摘要不变；
  显式文件声明 hash 随字节变化（4216 字节，A→B sha256 改变）；固定 config.json 时
  `config_sha256` 稳定。支持 Issue A 提案（目录后缀集合加 `.pth`）的复现材料。

### 2.3 L2 审计 finding（未抑制，逐条保留）

fixture 基线的 L2 审计共 61 findings：2 CRITICAL
（`model.revision_pinned`：fixture 模型 provider=other 无不可变 revision——如实反映，
不用假 revision 消除）、13 WARNING（provider 不可解析、dtype/quantization/seed 未声明、
`env.llm_critical_deps_pinned` 对 flash_attn/sentencepiece——原项目 requirements 确实未
pin 这两项——以及 untracked 提示等）、14 INFO、32 PASS、首轮另有
`consistency.*` 与 profile 相关提示。完整 JSON 在 `private/fixture-a-l2.stdout`。

## 2.4 真实层进展（2026-11-06 下午更新）

凭据与资产到位后完成了 local 全链路与 live 首跑：

- **真实环境**：`.venv-live`（CPython 3.12.13，211 包，依赖检查通过）。
  无 NVIDIA GPU，按计划走官方 CPU 轮子：torch 2.6.0+cpu / torchvision
  0.21.0+cpu（版本与原 pins 一致，去除 nvidia-*/triton；uvloop 0.21.0
  无 Windows 轮子按平台事实跳过——完整记录见
  `lingshu-trial-evidence/private/requirements-live-cpu.txt`）。
- **checkpoint**：首份交付（checkpoints_resnet18/）经验证与应用预期不符
  （自定义 features/classifier 键名、输出头 5 类、best 骨干实为 resnet34、
  5 类标签名无处可查）——如实保留在原目录不使用。第二份交付
  `checkpoints_resnet18_6classes/tongue_coating_resnet18_6classes.pth`
  通过全部验收：非 LFS pointer、torchvision resnet18+fc→6 严格加载、
  CPU 前向 (1,6)、真实病例图片预测「黑苔 100.00%」（样例即黑苔）。
  复制为 `trial/assets-local/classifier/tongue_coating_resnet18.pth`
  （sha256 `b8ee07a926b12bc4322b81860fd09edbea6bb2343215fec680b67ca8b796deaa`，
  来源：维护者交付 checkpoints_resnet18_6classes/，附交付 test.py）。
- **GTE/索引**：modelscope 下载 iic/nlp_gte_sentence-embedding_chinese-base
  至 trial/assets-local/gte（floating 下载，字节身份见
  trial/assets-local/assets-manifest.json，22 文件哈希）；以改造后的构建
  CLI 生成 chroma-200 与 chroma-400 两个索引及 build-manifest.json；真实
  检索冒烟返回 doc 身份/来源/排名/得分。
- **local-baseline run**（`20261106T084948Z-adb8ed`，child=.venv-live）：
  八阶段全部成功；真实分类（torch CPU）+ 真实 GTE/Chroma 检索（L2 距离
  0.687/0.702）+ cassette 注入 LLM/搜索；verify 18 项全 PASS；trace 单源
  local；预算 59/80。L2 审计 61 findings、0 CRITICAL（fixture 基线的
  2 条模型 revision CRITICAL 在 local 真实资产声明下消失）。
- **chunk 200→400 对照**：重建 400/20 索引后
  `lingshu-local-chunk400-c`（`20261106T085519Z-b60f3a`）对照
  `diff --fail-on HIGH` 精确命中 `custom.rag.chunk_size 200→400`（HIGH）+
  配置 hash 伴随变化；真实检索返回**未变化**（3 条短语料各成一块，如实
  记录，不夸大）。首跑 chunk400（`20261106T085207Z-8229e7`）暴露了我方
  配置错误（路径切 400、chunk_size 仍 200）——已修正并新增 verify 交叉
  校验（config 分块参数 vs 索引构建清单），污染的首跑归档保留。
- **recorder/child 环境分离（记录在案的偏差）**：.venv-live 内以
  --no-deps 安装的 reprollm 与原项目旧 pins（typer 0.15.2 等）存在依赖
  冲突，lock 触发上游依赖的内部错误；故 recorder 统一用 .venv-trial
  （reprollm 自身锁定依赖，与 fixture 基线一致），child 用 .venv-live。
  该冲突本身是"依赖隔离"价值的真实案例，也是上游可关注的兼容性观察。
- **live 完成**（`lingshu-live-baseline-f`，`20261106T093532Z-40519b`，
  child exit 0，251 秒）：**八阶段全部真实成功**——torch ResNet18 真实
  分类、硅基流动 `Qwen/Qwen3-VL-32B-Instruct` 真实视觉诊断（base64
  data URL）、DeepSeek `deepseek-chat` 真实多轮问诊/规划/协作/总结、
  真实 GTE/Chroma 检索、真实 Serper 搜索 5 次、报告落盘；预算 58/80；
  trace 单源 live；L2 审计 0 CRITICAL（61 findings）。
  - 模型切换记录：StepFun key 402 配额耗尽后，经贡献者显式决策切换视觉
    provider（`models.vision.base_url/api_key_env` 可配置；键名为环境
    变量名，非密钥值）。首跑另暴露两处真实边界并按原语义处理：病例 3 轮
    消息对真实模型不够（`informer_exhausted`）→ 派生
    `case-001-live.json` 追加 2 轮收束对话（结束标记仍由模型决定）；
    `baike.deno.dev` 第三方百科镜像服务死亡（直连/代理均不可达）→ 词条
    搜索按贡献者授权改走原代码中已有但被注释的 Serper 路径
    （`search.mode=serper`）；真实模型任务分割格式漂移 →
    `planner_max_attempts` 2→4（纠正反馈循环沿用原机制）。真实协作中
    一个任务在第 4 轮 `task_done` 提前完成（其余 3 个达到轮数上限）——
    verify 为非 fixture 模式改用上界+终止原因校验，精确计数校验仅限
    fixture。首跑 402 与中间失败 run（informer_exhausted / links DNS /
    keyword batch bug / planner_exhausted）全部保留归档，未删除。
  - 凭据经 .env 注入环境变量，不入库；`.env` 已 gitignore。
- **.pth 复测**：fetch upstream 确认 main 仍为 dfb72eda（探针基线即最新
  main），探针三项目 audit 报告已补录（L0 无 .pth 相关 finding）；
  Issue A 草稿（含完整对照材料）已备于
  `lingshu-trial-evidence/public/issue-pth.md`，待人工审阅后提交。

## 3. BLOCKED / 未执行（原样列出）

- **fork 与 upstream remote**：需 `gh auth login --web`（交互式，仅贡献者可完成）。
- **local 模式**：仓库不含可用的 ResNet18 checkpoint（`backend/model/tongue_coating_classify/`
  只有 dataset/train/test 材料，无 .pth 权重）；GTE 模型与真实 Chroma 索引未下载/构建
  （重依赖 torch/CUDA 属独立资源步骤）。按计划保留 BLOCKED，不用合成权重替代。
- **live 模式**：已完成（见 2.4）；StepFun 凭据因配额耗尽被贡献者决策
  替换为硅基流动 Qwen3-VL（显式记录）。
- **issue/PR（ReproLLM）**：依赖 fork/登录与维护者确认，未提交；Issue A 的复现材料
  （探针矩阵）已就绪。
- **Gate A / 产品 PR 质量门**：无产品 PR，未执行。
- 一次性副本的 `lock --check` 在 `git archive` 行尾转换（autocrlf）下会误报
  project-rules 变化——副本负例先重新 lock 再施加漂移；主仓库不受影响
  （主仓库 `lock --check` 始终 fresh）。已作为环境观察记录。

## 4. 公开材料

`lingshu-trial-evidence/public/case-study/`：合成病例、prompt、配置、fixture、
选定 run 目录快照（manifest/lock）、四个产物文件、基线/变体 diff、失败用例与
探针摘要；`package` 按允许清单复制并扫描敏感模式（负例 N07b 验证拒绝行为）。
原始证据保留在 `private/`，未改写。
