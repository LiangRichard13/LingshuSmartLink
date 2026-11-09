# LingshuSmartLink：从 clone、fork 到单病例试用与上游贡献

日期：2026-11-05（Asia/Shanghai）。这是供后续执行的计划；本次交付没有替执行者创建 fork、安装 Lingshu 重依赖、调用模型或提交 issue/PR。源码依据见[前次分析](2026-11-05-lingshu-smartlink-assessment.md)。

适用环境：Linux、Bash、Git、GitHub CLI（gh）、uv、Python 3.12。下文先克隆 Lingshu，再 fork ReproLLM；代码块依顺序在同一个 Bash 会话执行。需要登录/准备资产的步骤明确写出，未准备的资源保留 BLOCKED，不填 PASS。新回放入口和 trial_tools.py 是本计划要求实施的下游代码，完成第 5–7 步验收后，才能执行引用它们的命令。

## 1. git clone LingshuSmartLink，固定起点

在自己选择的空工作目录中执行：

```bash
set -euo pipefail
git clone https://github.com/LiangRichard13/LingshuSmartLink.git LingshuSmartLink
cd LingshuSmartLink
git status --short
git checkout --detach 2d6f839c9d2400bfe9f836bd400085971c9a511f
git switch -c codex/single-case-replay
LS_ROOT="$PWD"
TRIAL_ROOT="$(dirname "$LS_ROOT")/lingshu-trial-evidence"
mkdir -p "$TRIAL_ROOT/private" "$TRIAL_ROOT/public" "$TRIAL_ROOT/dist"
git rev-parse HEAD
git remote -v
```

验收：起点 SHA 正确，最初 tracked/untracked 状态为空。已有同名目录时先核对其 origin、HEAD 和未提交修改，不覆盖、不强制 reset。后文的绝对路径变量只用于本机调用，不写入分享材料。

在专用Bash会话执行，出现未预期错误即停止；capture会单独接住预期非0。安装缺失工具后若重开终端，先回到两个仓库共同父目录，重新设置LS_ROOT、RP_ROOT、TRIAL_ROOT、LS_PY、RP_BIN及capture函数，再从对应步骤继续，不能假定旧变量仍存在。

若 Git 未安装，先按[Git 官方安装说明](https://git-scm.com/downloads)安装，然后重新执行上述 clone。仓库中的大文件/LFS pointer 不能当成已准备模型：当前阶段只读代码；第 12 步逐一检查所需权重，不在初次 clone 后默认下载所有资产。

**不要直接启动原 agentsRunner.py。** 原入口会交互读病例、初始化本地模型并请求外部服务；这里先保存原始行为，再建立有界回放。

## 2. 登录自己的 GitHub 账户，然后 fork ReproLLM

先检查工具：

```bash
git --version
gh --version
uv --version
```

缺少 gh 时按[GitHub CLI 安装说明](https://github.com/cli/cli#installation)安装；缺少 uv 时按[uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装。完成后重新检查版本。

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth status
gh auth setup-git
GH_LOGIN="$(gh api user --jq .login)"
printf '%s\n' "$GH_LOGIN"
```

在输出中确认这是实际贡献者的账户。若机器保存多个账户，先用 `gh auth switch --hostname github.com --user 实际登录名` 切到正确身份，再重新设置 GH_LOGIN。不要输出 token。

回到两个仓库的共同父目录，再 fork，显式管理 remote：

```bash
cd "$(dirname "$LS_ROOT")"
gh repo fork EnumaElish123/ReproLLM --clone=false --remote=false
git clone "https://github.com/$GH_LOGIN/ReproLLM.git" ReproLLM-contrib
cd ReproLLM-contrib
git remote add upstream https://github.com/EnumaElish123/ReproLLM.git
git fetch upstream
git remote -v
gh repo view "$GH_LOGIN/ReproLLM" --json nameWithOwner,isFork,parent
git switch -c codex/lingshu-trial-reference dfb72eda57abb47cc0fc972982e770c6199c5bd2
RP_ROOT="$PWD"
RP_SOURCE_SHA="$(git rev-parse HEAD)"
```

首次提交前用 `git -C "$LS_ROOT" var GIT_AUTHOR_IDENT` 和 `git var GIT_AUTHOR_IDENT` 核对署名。缺少配置时输入自己的公开贡献身份，并只设置这两个checkout：

```bash
read -rp "Public commit name: " CONTRIB_NAME
read -rp "Verified GitHub email or GitHub no-reply address: " CONTRIB_EMAIL
git -C "$LS_ROOT" config --local user.name "$CONTRIB_NAME"
git -C "$LS_ROOT" config --local user.email "$CONTRIB_EMAIL"
git config --local user.name "$CONTRIB_NAME"
git config --local user.email "$CONTRIB_EMAIL"
```

已经有 fork 时复用它，不重复创建；clone 失败若是同名目录已存在，先核查后复用或换新目录。验收：origin 是自己的 fork，upstream 是官方仓库，isFork=true，父仓库正确。fork 命令的布尔参数及 remote 行为依据[gh 官方文档](https://cli.github.com/manual/gh_repo_fork)。

本次试用先使用固定源码基准，避免公开包 0.6.1 与开发源码 0.7.0 混用。以后提交 PR 则从届时的 upstream/main 新建分支，并在该输入上复测，不能把旧基准的结果冒充最新 main。

## 3. 阅读规则，构建并安装固定来源的 ReproLLM

在 ReproLLM-contrib 中按顺序阅读：

1. AGENTS.md、CONTRIBUTING.md。
2. docs/plan/00_architecture_and_decisions.md。
3. docs/plan/01_specification.md，重点 manifest、local/API lock、run、bindings、project rules、diff。
4. docs/plan/README.md 和其当前执行包。
5. docs/plan/validation-workflow-2026-10-13.md、val.md。

本计划没有给 ReproLLM 新产品任务赋予已批准状态；第 15 步的格式扩展需要具体 spec 决定及上游任务安排。使用者侧的应用改造先推进。

```bash
uv python install 3.12
uv sync --locked --dev --python 3.12
uv run reprollm --version
uv export --locked --no-dev --no-emit-project --no-header --no-annotate \
  --format requirements.txt --output-file "$TRIAL_ROOT/reprollm-runtime.txt"
uv build --wheel --out-dir "$TRIAL_ROOT/dist"
RP_DEV="$RP_ROOT/.venv/bin/reprollm"
```

若 --locked 失败，记录失败并核对该提交的 uv.lock、Python 和 uv 版本；不通过自动升级依赖绕过。dist 必须是本轮新目录或已核对的空目录，防止误装旧 wheel。

在 Lingshu 内建立独立试用环境，将 recorder 与轻依赖回放子进程放在同一环境：

```bash
cd "$LS_ROOT"
printf '\n.venv-trial/\n.venv-live/\n' >> .git/info/exclude
uv venv --python 3.12 .venv-trial
LS_PY="$LS_ROOT/.venv-trial/bin/python"
RP_BIN="$LS_ROOT/.venv-trial/bin/reprollm"
uv pip install --python "$LS_PY" --require-hashes -r "$TRIAL_ROOT/reprollm-runtime.txt"
WHEEL="$("$LS_PY" - "$TRIAL_ROOT/dist" <<'PY'
from pathlib import Path
import sys
wheels = list(Path(sys.argv[1]).glob("reprollm-*.whl"))
assert len(wheels) == 1, "dist must contain exactly one ReproLLM wheel"
print(wheels[0])
PY
)"
uv pip install --python "$LS_PY" --no-deps "$WHEEL"
uv pip install --python "$LS_PY" "pytest==8.3.5"
uv pip check --python "$LS_PY"
"$RP_BIN" --version
"$LS_PY" --version
```

生成不含本机路径的工具来源记录：

```bash
export RP_SOURCE_SHA WHEEL TRIAL_ROOT
"$LS_PY" - <<'PY'
import hashlib, json, os
from importlib.metadata import version
from pathlib import Path
wheel = Path(os.environ["WHEEL"])
record = {
    "schema_version": 1,
    "repository": "EnumaElish123/ReproLLM",
    "source_commit": os.environ["RP_SOURCE_SHA"],
    "distribution_version": version("reprollm"),
    "wheel_name": wheel.name,
    "wheel_sha256": "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest(),
}
path = Path(os.environ["TRIAL_ROOT"]) / "private" / "reprollm-source.json"
path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
PY
uv pip freeze --python "$LS_PY" --exclude reprollm > "$TRIAL_ROOT/private/requirements-trial.txt"
```

检查导出的 requirements-trial.txt 不含本机 file URL、凭据或绝对路径；第6步再把它复制进Lingshu，保留原始L0源码输入。ReproLLM 本身由上面的 source SHA + wheel hash 单独恢复，不能把未发布的 0.7.0 当作 PyPI 安装命令。此阶段不安装 backend/requirements.txt 中的 torch/CUDA/模型依赖。第 12 步才准备真实应用环境。

## 4. 保存原始 Level 0 审计与命令回执

先定义本机命令采集函数；stdout、stderr、退出码、时间分别保存。该函数不把 audit 退出 1 当内部崩溃：

```bash
capture() {
  local expected="$1" label="$2"
  shift 2
  local started ended rc
  started="$(date +%s)"
  if "$@" >"$TRIAL_ROOT/private/$label.stdout" 2>"$TRIAL_ROOT/private/$label.stderr"; then
    rc=0
  else
    rc=$?
  fi
  ended="$(date +%s)"
  "$LS_PY" - "$TRIAL_ROOT/private/$label.receipt.json" "$label" "$expected" "$rc" "$started" "$ended" <<'PY'
import json, sys
from pathlib import Path
path, label, allowed, actual, start, end = sys.argv[1:]
record = {
    "schema_version": 1, "label": label,
    "allowed_exit_codes": [int(x) for x in allowed.split(",")],
    "exit_code": int(actual), "elapsed_seconds": int(end) - int(start),
}
Path(path).write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
assert int(actual) in record["allowed_exit_codes"], record
PY
}
capture 0,1 original-l0 "$RP_BIN" audit . --level 0 --format json
```

把日志里的实际 finding、profile、hint、路径和诊断作为原始观察保留。允许 0/1 只表示采集命令正常完成，不代表所有规则通过；2/3 必须调查。不要预填“依赖未固定”：原项目已经有大量 == pins。开发前 baseline 应在新增代码尚未落盘时完成；第3步只增加了本机环境及本地exclude，tracked源码保持原始SHA，环境观察的来源另行注明。

## 5. 实施下游改造：明确文件和接口

在 Lingshu 分支内实施；不要把这些文件放进 ReproLLM 的 src/。每项先写有独立预期的验收，再实施，以 Conventional Commit 保存。

| ID | 文件/接口 | 必须完成的改动 | 验收 |
|---|---|---|---|
| LS01 | backend/agentsRunner.py | Runner 构造器接收 config、services、clock；新增 run_case(case, image_path, output_dir)；run_chain 保留交互适配并调用同一业务编排 | 非交互与交互共用业务链，fixture 并未另造 toy pipeline |
| LS02 | 新增 backend/replay_case.py | CLI：--mode fixture/local/live、--case、--image、--config、--cassette、--output-dir；main 返回退出码 | stdin 关闭仍可运行；输入缺失、模式冲突、输出目录非空、越界路径均明确失败 |
| LS03 | 新增 backend/replay_support.py | 严格配置读取、服务工厂、FixtureServices、RealServices、输入摘要、事件回执、可注入时钟 | fixture 无网络与重依赖导入；不静默回退 fake |
| LS04 | 各 Agent 构造器及 *_config.py/systemMessageGen.py | 注入 LLM client/参数/模板；单一 prompt 来源；去除导入时创建服务和 load_dotenv 副作用 | spy client 看到真正 invoke/create 的 model、参数和消息摘要 |
| LS05 | tongueCoatingAgent.py、tongueCoatingClassifier.py | 模型名只来自配置；create 使用同一值；权重路径、map_location/device 可配置 | 消除 self.model 与实际请求硬编码不一致；fixture 不实例化 torch 模型 |
| LS06 | ragRetriever.py、gteEmbeddings.py、processed_rag_data.py | 路径参数化；构建脚本加 main guard；显式分块/检索参数；保留 doc ID、来源、排名、score可用性 | import 不构建索引；实际查询返回的文档身份和顺序进入回执 |
| LS07 | planAgent.py、LinksAddAgent、搜索工具 | planner_max_attempts 有界；全局 model_call_limit；HTTP timeout；注入搜索 client | 限额前检查；超限不发下一请求；格式错误/超时非0退出并留失败回执 |
| LS08 | ReportGenAgent、markdownTool | 时间与输出位置注入；报告、trace、consumed、result写到指定目录 | 不依赖 backend cwd；fixture 固定输入报告可逐字节复现 |
| LS09 | 新增 tools/trial_tools.py、tests/replay/、trial/ | 输入生成、manifest准备、环境清单、证据验证/归档、.pth探针；最小测试夹具 | 见第6–11步；只写本计划约定的 trial 文件和传入的证据目录 |

原 Runner 顶层导入 Agent、LangChain、torch 的路径必须重构：轻量编排只依赖接口，真实服务在 RealServices 工厂内延迟导入。仅给 Runner 增加 fake 参数、仍在 import 时加载重依赖，不满足 LS03。

LS06还需给构建脚本实现确定的CLI：`--corpus`、`--embedding-model`、`--persist-directory`、`--manifest`、`--chunk-size`、`--chunk-overlap`、`--separator`。这组参数在原脚本中尚不存在；第12步的构建命令以该改造验收为前提，禁止导入脚本即构建数据库。

### 5.1 单病例输入与执行语义

trial/cases/case-001.json 使用 schema_version:1，包含 case_id、synthetic:true、固定的消息列表、图片相对路径、报告固定时间。只用新建的合成病例及可分享图片；不把项目公开样例中的人物信息自动认定为可再分发试用数据。

run_case 按原顺序执行：分类/视觉→信息摘要→检索→规划→Actor/Instructor→Summary→链接搜索→报告。Informer 从病例消息列表消费输入，无 input()；消息耗尽但无结束标记时失败。Actor 跨任务保留历史、Instructor 任务间重置等原语义保持并测试。若有意调整，单独写明，不与“纯接入 ReproLLM”混为一项。

协作轮数定义为每任务最多 Instructor 调用次数，初次 Actor 不计入该数。定义清楚初次 Actor、修改轮数、提前结束标记的关系；不要改变原 5 轮边界后仍声称行为完全相同。

### 5.2 三种模式，统一业务编排

| mode | 分类/embedding/RAG | LLM/搜索 | 可声称的证据 |
|---|---|---|---|
| fixture | 注入的固定结果，不读真实权重 | 人工构造的确定响应 | 实际编排、配置消费、文件记录与回放接线 |
| local | 真实 ResNet18、GTE、Chroma | 注入的固定响应 | 真实本地组件和检索；外部部分仍为模拟 |
| live | 真实本地组件 | 真正 provider/API 与搜索 | 单病例完整在线运行，记录实际模型和返回 |

人工 fixture 明确标记 synthetic，不叫“历史录制”。live 产生的 cassette 若用于严格重放，应额外实现 exact-request 校验：请求消息/图片摘要不同就报 miss，不能把旧答案当作新 prompt 的模型输出。变更 prompt 的 fixture 测试仍可使用人工响应，但只证明状态变化可记录。

### 5.3 事件与产物契约

每次输出固定四个文件，其他文件须显式列入产物清单：

- report.md：最终报告；fixture 使用固定时间，不含执行环境信息。
- trace.jsonl：每条 schema_version:1，含 sequence、stage、task/turn、parent事件、fixture/local/live来源、实际 model、request/response摘要、文档顺序、终止原因；保留有意义的顺序。
- consumed.json：schema_version:1；代码实际使用的配置、输入/模板/权重摘要、实际请求构造值、每阶段调用计数、研究进程环境清单。不能只是复制 manifest。
- result.json：schema_version:1；case_id、mode、各阶段成功/失败、报告相对路径、失败原因码。

trace 不保存鉴权 header、密钥、图像 base64；合成输入的请求/响应正文可另列文件，引用其 hash。真实 provider 的 response.model、request ID、usage 若可用则保存；不可用写 null及原因，不编造。

验证配置消费时，用独立 spy client 检查真正传入 invoke/create 的参数、消息和调用次数；仅比较同一函数生成的“配置副本”和 manifest 没有证明力。

## 6. 实施 trial_tools.py 的有限命令集合

这是下游试用助手，不新增 ReproLLM 子命令：

| 子命令 | 必需行为 |
|---|---|
| seed | 生成合成病例/有效PNG、小语料、人工LLM与搜索fixtures、原始模板迁移文件及变体；已存在文件拒绝覆盖 |
| prepare --mode --variant [--config FILE] | baseline、rounds-3、prompt-v2、image-v2、retrieval-reordered、search-v2六种输入；显式config时按该文件准备；写current.yaml与reprollm.yaml；真实模式核验资产；同步config_files与bindings |
| env --output | 用 importlib.metadata 记录当前研究进程的 Python和已安装包版本，稳定排序，无sys.executable绝对路径 |
| verify --run --scenario | 读取本模式独立预期；验证run状态、输入指纹、bindings、应用实际消费见证、artifact hash、阶段计数与报告 |
| archive --run --label --dest | 复制完整ReproLLM run目录、manifest/lock与应用输出到该label目录；校验副本hash，再移走current输出，为下一次运行腾空；拒绝覆盖归档 |
| probe-pth --reprollm --dest | 生成独立一次性项目；目录/文件两种声明、变更前后各lock一次，保存完整stdout/stderr/lock及比较结果 |
| package --source --output | 按公开allowlist复制审阅过的合成材料、来源清单和结果；原始证据不改写；拒绝秘密/本机路径等禁止内容 |

prepare 不是 ReproLLM 自动修复功能，而是试用者主动准备自己的实验声明。它只能复制选择的 trial 资产、写 current.yaml/manifest；不随意改 Agent 代码或旧 run。env 的包清单包含 langchain、langchain-openai、chromadb、modelscope、torchvision 等，弥补默认 recorder 包清单并非完整应用环境的边界。

```bash
cd "$LS_ROOT"
mkdir -p trial tools tests/replay
cp "$TRIAL_ROOT/private/requirements-trial.txt" requirements-trial.txt
"$LS_PY" tools/trial_tools.py seed
printf '3.12\n' > .python-version
```

将 .venv-trial/、.venv-live/、trial/assets-local/、trial/outputs/、.reprollm/runs/、__pycache__/ 追加到已有 .gitignore，保留其原内容。合成输入、prompt、配置、fixtures、测试和源码应跟踪；真实权重不提交。

### 配置样例：trial/configs/current.yaml

下面是 prepare 生成的 fixture baseline 的最低字段形状。generation 是本试用新选定的请求设置，不宣称原代码原先已显式使用这些值：

```yaml
schema_version: 1
trial:
  mode: fixture
  case_id: case-001
  report_time: "2026-11-05T00:00:00Z"
workflow:
  chat_turn_limit: 5
  planner_max_attempts: 2
  model_call_limit: 80
generation:
  temperature: 0.2
  top_p: 1.0
  max_tokens: 512
rag:
  chunk_size: 200
  chunk_overlap: 20
  separators: ["<目录>"]
  k: 2
  corpus_manifest: trial/corpus/manifest.json
  index_manifest: trial/fixtures/index-manifest.json
classifier:
  mode: fixture
  checkpoint: null
search:
  mode: fixture
  cassette: trial/fixtures/external.json
```

k=2 是本次试用显式选择，不是对原 SDK 默认值的追认。live/local 扩展配置包含本机已准备资产的仓库内相对路径；各角色模型配置、超时和预算必须显式写全。若 provider 不支持 seed，不发送也不假称采样确定性。

## 7. 接入前先完成下游验收

至少完成以下 tests/replay 用例；人工服务全程禁止 socket/HTTP 出站，真实组件测试另加 local 标记且不会默认下载：

- 无 stdin、无 torch/LangChain 安装时，fixture 能导入并执行真实 Runner 编排。
- 固定输入连续两次 report 字节相同；事件顺序相同。运行时间字段单列，不用于冒充整份 run.json 字节相同。
- spy 验证每个角色实际 model/参数/消息摘要；视觉请求不再另用硬编码 model。
- 轮数5与3分别真正消费对应次数；提前完成、达到轮数、规划重试达到上限都留明确原因。
- 病例缺失、图片损坏/缺失、fixture键缺失、非法配置、输出非空、预算超限、HTTP失败均非0退出。
- 检索doc身份与排序、搜索回执变化确实进入实际下游消息，而非只写进trace。
- 模板改变后，实际消息摘要变化；图片改变后，实际消费摘要变化。
- 模拟失败保留已完成阶段，不生成成功result；不得悄悄用fixture替代失败的live请求。

```bash
"$LS_PY" -m pytest -q tests/replay -m "not local and not live"
```

把这些验收实现完并通过后，提交下游代码和合成材料；记录实际 Lingshu adaptation SHA。提交前检查 git diff 与 staged 文件，不使用 git add . 包含真实资产。推荐拆为“接口/配置”“回放与回执”“夹具与测试”三个任务提交；超过约400行的上游PR也按独立可验收范围拆分。

## 8. 初始化 ReproLLM、声明 manifest 和项目规则

先确认目录和命令：

```bash
cd "$LS_ROOT"
"$RP_BIN" init . --profiles inference
"$LS_PY" tools/trial_tools.py prepare --mode fixture --variant baseline
"$LS_PY" tools/trial_tools.py env --output trial/environment.json
```

init 只生成初稿。prepare 填写后的 manifest 必须人工检查：模型身份、profiles、实际文件、所有绑定值与程序一致。不要用 --force 覆盖已整理的声明。

fixture manifest 的可校验示例：

```yaml
schema_version: 1
project:
  name: lingshu-single-case
experiment:
  name: case-001-fixture
  profiles: [inference, trial_policy]
models:
  primary:
    provider: other
    id: fixture-scripted-v1
prompts:
  vision:
    path: trial/prompts/vision.txt
  actor:
    path: trial/prompts/actor.txt
  instructor:
    path: trial/prompts/instructor.txt
  planner:
    path: trial/prompts/planner.txt
  informer:
    path: trial/prompts/informer.txt
  summary:
    path: trial/prompts/summary.txt
  links:
    path: trial/prompts/links.txt
  report_meta:
    path: trial/prompts/report_meta.txt
  report_main:
    path: trial/prompts/report_main.txt
generation:
  temperature: 0.2
  top_p: 1.0
  max_tokens: 512
inference:
  backend: other
execution:
  command: python backend/replay_case.py --mode fixture --case trial/cases/case-001.json --image trial/cases/case-001.png --config trial/configs/current.yaml --cassette trial/fixtures/external.json --output-dir trial/outputs/current
  cwd: "."
  config_files:
    - trial/cases/case-001.json
    - trial/cases/case-001.png
    - trial/configs/current.yaml
    - trial/corpus/manifest.json
    - trial/fixtures/index-manifest.json
    - trial/fixtures/external.json
    - trial/environment.json
bindings:
  generation.temperature:
    config: trial/configs/current.yaml:generation.temperature
  generation.top_p:
    config: trial/configs/current.yaml:generation.top_p
  generation.max_tokens:
    config: trial/configs/current.yaml:generation.max_tokens
custom:
  trial:
    mode: fixture
  workflow:
    chat_turn_limit: 5
    planner_max_attempts: 2
  rag:
    chunk_size: 200
    chunk_overlap: 20
    k: 2
    corpus_manifest: trial/corpus/manifest.json
    index_manifest: trial/fixtures/index-manifest.json
artifacts:
  outputs:
    - trial/outputs/current/report.md
    - trial/outputs/current/trace.jsonl
    - trial/outputs/current/consumed.json
    - trial/outputs/current/result.json
```

fixture primary 如实标为 fixture；不伪造 provider endpoint 或不可验证 revision。local/live 由 prepare 生成不同manifest：DeepSeek/StepFun 使用 provider:other+真实endpoint，primary指实际主要生成模型，vision指实际视觉模型，embedding在真正使用时声明local目录；角色参数放 models.<role>.params 或 custom。API alias 保留 unpinnable 与unsupported-provider提示。

辅助 ResNet 分类器可先以 custom.assets.classifier_file + execution.config_files 记录实际 .pth文件，这是辅助资产记录；另做第13步的ModelSpec格式探针。如果选择 models.classifier.provider:local 且id为具体.pth，虽然权重能hash，当前规则仍可能报缺少local.config_sha256的CRITICAL；不能承诺“文件路径绕过后审计全绿”，也不制造无实际含义的config只为消除finding。

创建 .reprollm/profiles/trial_policy.yaml，作为项目自己的漂移策略：

```yaml
schema_version: 1
name: trial_policy
description: Single-case application trial drift policy.
extends: []
rules: []
required_fields: []
severity_overrides: {}
drift_overrides:
  custom.workflow.chat_turn_limit: HIGH
  custom.workflow.planner_max_attempts: HIGH
  custom.rag.chunk_size: HIGH
  custom.rag.chunk_overlap: HIGH
  custom.rag.k: HIGH
detect: {}
```

该用户profile只配置漂移级别，不是新增内置Agent/RAG profile。通配符只匹配一个点分段，因此不用 custom.* 覆盖嵌套所有叶子。

添加三个实际需求规则；rules命令无项目位置参数，必须在LS_ROOT执行：

```bash
"$RP_BIN" rules add --id project.chat_turn_limit --field custom.workflow.chat_turn_limit \
  --severity CRITICAL --reason "Bound the actor and instructor iteration budget." \
  --config trial/configs/current.yaml:workflow.chat_turn_limit
"$RP_BIN" rules add --id project.corpus_manifest --field custom.rag.corpus_manifest \
  --severity CRITICAL --reason "Identify the corpus build used by retrieval." \
  --config trial/configs/current.yaml:rag.corpus_manifest
"$RP_BIN" rules add --id project.rag_k --field custom.rag.k \
  --severity CRITICAL --reason "Record the number of retrieved documents." \
  --config trial/configs/current.yaml:rag.k
capture 0,1 prepared-l1 "$RP_BIN" audit . --level 1 --format json
```

项目规则检查present/non-null；binding提供配置观察。它们不会自行计算语料hash、验证Chroma内容或证明参数进入请求；独立消费见证仍必需。Informer/Instructor是生成链的一部分，本轮不选judge_only。每轮规则增加只执行一次，重复运行遇到ID占用应检查现状，不删旧规则重建历史。

## 9. 第一次 lock、预演、实际 run、证据核对

先把源码、测试、依赖清单、合成输入、profile、项目规则、manifest纳入明确的下游提交；排除运行产物与真实权重。然后：

```bash
capture 0 fixture-lock "$RP_BIN" lock . --offline
capture 0 fixture-lock-check "$RP_BIN" lock . --check
git add reprollm.lock
git commit -m "chore(trial): lock the single-case fixture baseline"
git status --porcelain=v1 --untracked-files=all
capture 0 fixture-preview "$RP_BIN" run --dry-run -- \
  "$LS_PY" backend/replay_case.py --mode fixture \
  --case trial/cases/case-001.json --image trial/cases/case-001.png \
  --config trial/configs/current.yaml --cassette trial/fixtures/external.json \
  --output-dir trial/outputs/current
capture 0 fixture-a "$RP_BIN" run --name lingshu-fixture-a --capture-output -- \
  "$LS_PY" backend/replay_case.py --mode fixture \
  --case trial/cases/case-001.json --image trial/cases/case-001.png \
  --config trial/configs/current.yaml --cassette trial/fixtures/external.json \
  --output-dir trial/outputs/current
"$RP_BIN" runs list --name lingshu-fixture-a --status completed --outcome success --json \
  > "$TRIAL_ROOT/private/fixture-a-list.json"
RUN_A="$("$LS_PY" - "$TRIAL_ROOT/private/fixture-a-list.json" <<'PY'
import json, sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text())
assert len(rows) == 1, "use a fresh campaign or inspect duplicate run names"
print(rows[0]["run_id"])
PY
)"
"$RP_BIN" runs show "$RUN_A" --json > "$TRIAL_ROOT/private/fixture-a-run.json"
"$LS_PY" tools/trial_tools.py verify --run "$RUN_A" --scenario baseline
capture 0,1 fixture-a-l2 "$RP_BIN" audit . --format json
capture 0 fixture-a-export "$RP_BIN" export . --run "$RUN_A" \
  --output "$TRIAL_ROOT/private/fixture-a-REPRODUCIBILITY.md"
"$LS_PY" tools/trial_tools.py archive --run "$RUN_A" --label fixture-a \
  --dest "$TRIAL_ROOT/private"
```

执行run前，git status必须为空；若不为空，解释并提交明确的输入改动或停止，不能强制清理。run --dry-run不执行、不写run，不算模型试用。run无项目位置参数；显式 --cwd只控制child工作目录，本计划统一从根目录执行，避免backend隐式cwd。

verify应独立检查：

1. 只有预期的新run，status完成且exit_code=0；result各阶段也成功，不能只凭child退出0。
2. case、图片、配置、模板、外部fixture、环境清单的输入hash与实际字节相等；图片二进制snapshot为空是正常现象。
3. bindings_observed中的轮数/k/采样值正确，consumed/spy见证与真正调用一致。
4. schema-2输入前后复查未发生未预期变化；代码SHA正确，dirty=false。
5. artifacts四个路径及hash与应用实际输出相等；ReproLLM仅记录输出hash，不会替你保存输出正文，archive必须在下一次覆盖前执行。
6. 不存在未解释的capture warning。fixture里的unsupported model/backend等finding逐条记录，不用suppression制造全绿。

首次完整run之后生成的L2报告还可能因源码扫描看到未执行的其他任务而提示profile不匹配；把仓库级信号与本次真实调用区分。审计的完整finding集合保留，初次适配发现新CRITICAL要定位，不静默排除。

## 10. 正向漂移矩阵：必须展示应用级差异

至少执行以下六个正向场景；只有baseline/同输入重跑可作为相同输入对。科学变量的变更会带来配置hash、manifest hash或code.commit等伴随变化，列全并解释，不能声称diff只有一条。

| 场景 | 改变什么 | 必验的实际消费/输出 | ReproLLM证据 |
|---|---|---|---|
| baseline重复 | 同一已提交输入，不改配置 | fixture report字节相同、事件顺序一致 | self-diff为空；两次run的实际完整diff另审阅 |
| rounds-3 | 轮数5→3 | 人工fixture在第5轮前不提前结束，spy分别观察5/3次Instructor调用 | custom.workflow.chat_turn_limit=HIGH |
| prompt-v2 | Actor模板增加可识别测试标记 | 实际Actor请求消息摘要改变 | prompts.actor.sha256=HIGH及相关文件变化 |
| image-v2 | 同一路径换有效PNG字节 | 输入消费hash改变；fixture不声称真实视觉模型响应改变 | files.<图片相对路径>.sha256=HIGH |
| retrieval-reordered | 固定文档集合改变返回顺序 | 下游Actor实际上下文顺序改变 | 已声明retrieval fixture文件hash=HIGH；trace展示语义顺序 |
| search-v2 | 固定搜索返回改变一项 | Report实际收到新返回，相关内容/摘要变化 | 已声明外部fixture文件hash=HIGH |

baseline重复：归档A后，不改输入、不prepare、不重新commit，以新name再次执行第9步run，得到RUN_REPEAT；verify并归档，然后执行 `capture 0 baseline-repeat-diff "$RP_BIN" diff "$RUN_A" "$RUN_REPEAT" --format json --fail-on HIGH`。完整报告可能仍含NONE元数据差异，逐项记录；不把self-diff代替两次实际执行。

其余五个真正改变输入的变体逐个执行完整循环：

1. `trial_tools.py prepare --mode fixture --variant 对应名称`。
2. 检查只改该变体计划内的输入；env未变则不重复生成。
3. `lock . --offline`，随后 `lock . --check`。
4. 提交变体实际改变的配置、manifest、lock、模板/fixture/图片；确保运行前干净。
5. 按第9步同一显式child命令执行，run name改为 `lingshu-fixture-变体名`，保存完整命令回执。
6. 用runs list精确名称提取该run ID；保存为本机 `RUN_B`；重复verify、export和archive。
7. 执行下列比较，审阅完整JSON并断言表中的路径、级别和消费证据。

```bash
capture 0 baseline-self-diff "$RP_BIN" diff "$RUN_A" "$RUN_A" --format json --fail-on HIGH
capture 1 variant-diff "$RP_BIN" diff "$RUN_A" "$RUN_B" --format json --fail-on HIGH
```

实际JSON在capture的stdout文件；diff没有--output选项。每个场景用不同label防止覆盖前面的回执。--fail-on HIGH的1是预期门控结果；2/3是失败。归档后的run仍保留在.reprollm/runs中，使用显式ID避免“最新两个成功run”选错配对。

分块200→400另列到真实local RAG场景：必须重新构建索引并保留构建清单，在真实查询返回与下游消息中观察影响。fixture中只改chunk_size可以测binding/drift，不能称为真实RAG行为验证。

## 11. 负例：证明边界与失败可被观察

全部在独立的一次性Lingshu副本或专用测试临时目录中做，主试用输入保持可恢复。每项保存expected/actual和完整日志。

| ID | 操作 | 独立预期 |
|---|---|---|
| N01 | 删除custom.workflow.chat_turn_limit | project.chat_turn_limit产生CRITICAL，audit默认退出1 |
| N02 | manifest仍声明5，配置实际读3，重新run | spy消费3；consistency.custom_fields报告冲突，保留其他文件hash finding |
| N03a | 只修改prompt字节，manifest和rules字节不变 | lock --check仍退出0；audit的consistency.file_hashes发现实际文件漂移；freshness检查不是文件完整性证明 |
| N03b | 修改manifest或project-rules后不重新lock | lock --check退出1；审计保留lock freshness等finding |
| N04 | 删除图片/坏PNG | child非0，失败result，无成功报告；ReproLLM保留失败run |
| N05 | 缺fixture键/严格录制请求hash不符 | 明确miss失败，不请求真实API补救 |
| N06 | 超过planner尝试或全局请求预算 | 到界限即停止，下一次请求未发出；trace记录原因 |
| N07 | 人工服务抛超时或解析错误 | 保留已完成阶段，非0退出；秘密canary不进入公开包 |
| N08 | 图片路径只经stdin传入、未显式声明 | 对照说明原交互入口不能自动建立图片文件证据；新入口显式argv后可捕获 |

对N02不要只用config observation自证消费；必须读实际调用spy/回执。不要放宽全局schema、改通配符、降低上游audit阈值或伪造model revision来让负例通过。

N03依据当前[check_lock实现](../../src/reprollm/lock/writer.py)：它比较manifest与project-rules摘要，不扫描所有已锁文件字节。

## 12. 准备真实资产，完成local和可选live试用

### 12.1 单独恢复真实应用环境

A层fixture通过后再准备。真实阶段的软件、模型下载空间和CPU/GPU资源由执行者确认；未具备时保留BLOCKED，仍可完成fixture和格式探针。

在Lingshu新建真实环境，依据原项目依赖完整恢复，而不是把它们安装进ReproLLM-contrib：

```bash
cd "$LS_ROOT"
uv venv --python 3.12 .venv-live
LIVE_PY="$LS_ROOT/.venv-live/bin/python"
uv pip install --python "$LIVE_PY" -r backend/requirements.txt "$WHEEL" "pytest==8.3.5"
uv pip check --python "$LIVE_PY"
```

这一步包含torch及原仓库CUDA等重依赖，耗时/空间显著；是独立资源步骤，不属于轻依赖fixture前置条件。本手册未实测该组合的当前可安装性。失败时保存解析/构建错误，优先确认系统、Python、原pins；如必须改依赖，列为下游兼容性任务，记录原/新版本，重跑fixture，不静默升级全部包。CPU专用路线应另用官方CPU轮子建立被记录的依赖方案，参考[uv/PyTorch官方说明](https://docs.astral.sh/uv/guides/integration/pytorch/)，不混入原CUDA环境结果。

接着逐项准备在仓库内、被gitignore排除的trial/assets-local：

- ResNet18真实checkpoint：来自项目维护者交付或有来源记录的训练结果；验证非LFS pointer、文件字节hash、六类标签顺序与网络结构；在CPU/device配置下实际load并完成一次前向。不得用合成.pth替代。缺失则先交付权重，不能称local完成。
- GTE embedding：准备源码所用的iic/nlp_gte_sentence-embedding_chinese-base，记录下载来源/revision若可得、文件清单及hash，路径改为仓库内相对路径；不能仅把一条Hub ID当作本地模型已存在。
- 小语料：用本试用合成语料或有明确再分发依据的片段，固定内容、文件清单和源版本；完整古籍库不是首次试用必需。
- Chroma索引：经改造后的构建脚本，从固定语料按200/20分块、显式embedding构建；输出index-manifest（schema_version、语料/模板/embedding摘要、分块参数、collection、文档ID与数量）。库文件字节hash不是语义等价证明。

具体执行顺序如下；原仓库本身未提供本次可直接使用的分类checkpoint，若没有维护者交付的可信文件，就在此保留BLOCKED，不跳到“真实组件通过”。本轮不扩展为重新训练分类模型的研究任务。

```bash
mkdir -p trial/assets-local/classifier trial/assets-local/gte
read -rp "Path to the provided ResNet18 checkpoint: " CHECKPOINT_SOURCE
test -f "$CHECKPOINT_SOURCE"
cp -- "$CHECKPOINT_SOURCE" trial/assets-local/classifier/tongue_coating_resnet18.pth
"$LIVE_PY" - <<'PY'
from pathlib import Path
import torch
from torchvision.models import resnet18
p = Path("trial/assets-local/classifier/tongue_coating_resnet18.pth")
assert not p.read_bytes()[:200].startswith(b"version https://git-lfs.github.com/spec/")
state = torch.load(p, map_location="cpu", weights_only=True)
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 6)
model.load_state_dict(state, strict=True)
model.eval()
with torch.no_grad():
    output = model(torch.zeros(1, 3, 224, 224))
assert tuple(output.shape) == (1, 6)
print("checkpoint structure and CPU forward smoke passed")
PY
"$LS_ROOT/.venv-live/bin/modelscope" download --help
read -rp "GTE provider revision (empty means floating download): " GTE_REVISION
if [ -n "$GTE_REVISION" ]; then
  "$LS_ROOT/.venv-live/bin/modelscope" download \
    --model iic/nlp_gte_sentence-embedding_chinese-base \
    --revision "$GTE_REVISION" --local_dir trial/assets-local/gte
else
  "$LS_ROOT/.venv-live/bin/modelscope" download \
    --model iic/nlp_gte_sentence-embedding_chinese-base \
    --local_dir trial/assets-local/gte
fi
"$LIVE_PY" - <<'PY'
import hashlib, json
from pathlib import Path
base = Path("trial/assets-local")
items = []
for root in (base / "classifier", base / "gte"):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            assert not path.is_symlink(), "copy actual assets inside the repository"
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            items.append({"path": path.as_posix(), "size_bytes": path.stat().st_size,
                          "sha256": "sha256:" + digest.hexdigest()})
Path("trial/assets-local/assets-manifest.json").write_text(
    json.dumps({"schema_version": 1, "files": items}, sort_keys=True, indent=2) + "\n")
PY
"$LIVE_PY" workspace/RAG/processed_rag_data.py \
  --corpus trial/corpus/texts --embedding-model trial/assets-local/gte \
  --persist-directory trial/assets-local/chroma-200 \
  --manifest trial/assets-local/chroma-200/build-manifest.json \
  --chunk-size 200 --chunk-overlap 20 --separator '<目录>'
```

ModelScope参数已对照其[v1.25.0源码](https://github.com/modelscope/modelscope/blob/v1.25.0/modelscope/cli/download.py)，实际安装版本的help仍须核对。空revision只表明下载时获取了一个内容快照；assets-manifest中的实际字节身份可核验，不能据此宣称已经证明不可变Hub revision。另在sources记录交付者/下载来源、请求revision及实际可得的provider标识，避免仅有hash而无来源说明。checkpoint的零张量前向只是结构smoke，真实case的预处理、标签顺序和预测另由local测试验证。

将LS_PY/RP_BIN切换为同一真实环境，随后采集新env，明确开始新的mode/campaign：

```bash
LS_PY="$LIVE_PY"
RP_BIN="$LS_ROOT/.venv-live/bin/reprollm"
"$LS_PY" -m pytest -q tests/replay -m "not local and not live"
"$LS_PY" -m pytest -q tests/replay -m local
"$LS_PY" tools/trial_tools.py prepare --mode local --variant baseline
"$LS_PY" tools/trial_tools.py env --output trial/environment.json
```

local smoke失败先修下游，不进入live。按第9步对local执行锁定、提交、预演、run、verify、export、archive；显式child的--mode改local，run name/日志label也改为local。classifier/embedding实际调用和RAG返回必须标local，LLM/搜索保持fixture标记。

然后执行chunk200→400，建立独立索引：

```bash
"$LIVE_PY" workspace/RAG/processed_rag_data.py \
  --corpus trial/corpus/texts --embedding-model trial/assets-local/gte \
  --persist-directory trial/assets-local/chroma-400 \
  --manifest trial/assets-local/chroma-400/build-manifest.json \
  --chunk-size 400 --chunk-overlap 20 --separator '<目录>'
```

手工把local配置的chunk_size、persist_directory和index_manifest切到400版本，prepare读取并保留这份明确配置（实现时提供 `--config trial/configs/local-chunk400.yaml` 输入），同步manifest，重新lock和提交、运行并比较。记录真实doc数量/顺序变化以及可能未变化的结果，不预先承诺一定改变答案。

### 12.2 在线单病例live

只有真实组件已通过且维护者准备了目的地、凭据和本轮预算才执行。必须在trial/configs/live.yaml明确：

- DeepSeek/StepFun实际endpoint与model ID；StepFun旧别名若不可用就报告，不静默换模型。
- 每角色temperature、max_tokens等受支持参数；HTTP timeout、SDK retry次数、planner最大尝试、协作轮数与全局请求上限。全部attempt包括SDK重试都受预算约束。
- Serper搜索次数/返回数量限制；模型和搜索请求的可用花费上限由资源提供者填写；缺值不启动live。
- 用同一个合成case和有使用依据的图片，日志只记录必要内容。

凭据不写进配置、manifest或命令参数；使用终端静默输入或已有本机凭据管理：

```bash
read -rsp "DeepSeek API key: " DEEP_SEEK_API_KEY
printf '\n'
export DEEP_SEEK_API_KEY
read -rsp "StepFun API key: " STEP_FUN_API_KEY
printf '\n'
export STEP_FUN_API_KEY
read -rsp "Serper API key: " SERPER_API_KEY
printf '\n'
export SERPER_API_KEY
```

确认live.yaml已满足预算前置条件，再执行prepare --mode live，采集env，锁定、提交、预演；按第9步运行一次--mode live并完整验证、归档。ReproLLM的lock --offline不会产生模型请求，但包裹的live child会；other provider的lock --verify-api没有额外校验transport，不用它冒充目的地验证。

完成后清理当前会话的key变量：

```bash
unset DEEP_SEEK_API_KEY STEP_FUN_API_KEY SERPER_API_KEY
```

如做第二次live，只能在本轮剩余预算内；两次输出不同不自动说明ReproLLM失败，需对照模型不可固定性、检索/搜索返回及真实请求证据。只有local完成，才能说真实本地组件试用；只有live完成，才能说完整在线病例链路已验证。fixture文档本身可以成立，但必须如实限定。

## 13. 独立复现.pth目录边界，保留完整CLI对照

格式探针与真实模型效果分开，不污染单病例主manifest：

```bash
cd "$LS_ROOT"
"$LS_PY" tools/trial_tools.py probe-pth --reprollm "$RP_BIN" \
  --dest "$TRIAL_ROOT/private/pth-probe"
```

probe-pth的实现必须生成两个独立最小项目，各自schema_version:1的manifest，并保存如下矩阵：

| 项目 | models.primary.id | 内容变更 | 必录字段 |
|---|---|---|---|
| directory | weights | 目录内classifier.pth，A→B | local.weights.hashed/total_size_bytes/sha256，完整revision |
| file | weights/classifier.pth | 同一文件A→B | 同上 |
| directory+config补充 | weights | 固定有效config.json，.pth A→B | 分别比较config_sha256与weights.sha256 |

每次执行实际CLI `lock . --offline --hash-large-files`，保存stdout、stderr、退出码、完整lock及输入摘要。当前基准预期：仅.pth目录识别权重数为0、hashed=true且空集合摘要不变；显式文件hash变化。带固定config时revision可能优先采用config_hash，因此“权重变化”测试断言weights.sha256，不能强求revision值随权重改变。

补充实际audit报告：无config的local声明可能报缺少配置摘要的CRITICAL，带config的目录未必有该finding；当前没有专门的“.pth未识别”finding，不能预填它。lock退出0不代表audit全绿。探针中的任意字节只是格式测试，不称可加载ResNet checkpoint。

提交前在届时upstream/main重新构建wheel并安装到新的探针环境复测。若问题已修复，报告修复SHA并停止重复实现；历史基准结果仍保留。

## 14. 整理可公开的试用证据

在Lingshu下游新增 docs/reprollm-single-case-trial.md，记录两仓库SHA、wheel hash、mode、输入/模型来源、命令/退出码/耗时、每个矩阵expected/actual、所有finding与限制。环境清单单独标研究进程，不把另一venv的recorder观察当成child证明。

每个公开包至少包括：

- schema_version:1的sources.json/results.json；角色与阶段来源矩阵。
- 可分享的合成病例、图片、prompt、配置、人工/真实录制身份明确的cassette。
- 选定run目录及其manifest/lock快照；必要输入的完整正文或可独立获取的固定来源。
- 对应应用report、trace、consumed、result；与artifact hash相符。
- baseline/变体完整diff、默认selected-run export、失败用例和.pth CLI证据。
- 研究者独立检查说明及Agent协助披露；未执行/阻塞项原样列出。

```bash
"$LS_PY" tools/trial_tools.py package --source "$TRIAL_ROOT/private" \
  --output "$TRIAL_ROOT/public/case-study"
"$RP_ROOT/.venv/bin/python" "$RP_ROOT/scripts/check_no_leaks.py" \
  "$TRIAL_ROOT/public/case-study"
```

既有检查脚本只是辅助：另人工审阅未知格式key、绝对路径、用户名、病例字段、图片权利及请求正文。输出快照/日志经脱敏后不一定等于原始输入字节，验证原始输入hash不能靠重新hash脱敏文本替代。原始证据保持不变，公开包有单独清单。没有真实采用者试用时，只写“由贡献者执行的适配/案例”。

若需要在Lingshu发表下游PR：先用自己的账户fork Lingshu，origin/upstream按第2步同样明确配置；原项目有写权限的维护者按其流程提交。只推适配代码、测试、可分享夹具与案例，不推权重、凭据、完整本机原始日志。记录实际PR URL，不用计划中的名称假装已经提交。

## 15. 向ReproLLM提issue：先去重，再提交具体范围

先检查最新条目：

```bash
cd "$RP_ROOT"
gh issue list --repo EnumaElish123/ReproLLM --state all --limit 100
gh pr list --repo EnumaElish123/ReproLLM --state all --limit 100
gh issue view 11 --repo EnumaElish123/ReproLLM --comments
gh pr view 12 --repo EnumaElish123/ReproLLM
gh issue view 13 --repo EnumaElish123/ReproLLM --comments
gh issue view 14 --repo EnumaElish123/ReproLLM --comments
```

### Issue A：.pth目录权重格式支持，优先

建议标题：`spec(lock): include .pth files in local directory weight inventory`。

正文先写可复现的当前行为，然后提出精确窄提案：规范§4.3目录后缀集合增加.pth，保持现有大小阈值、排序、流式hash与显式文件行为；不改变config优先revision策略、不改零权重目录政策、不承诺读取checkpoint语义。

必填：最新ReproLLM SHA、环境/安装来源、最小项目正文、四次CLI完整对照及带config补充、预期/实际、显式文件绕过与audit限制、Lingshu真实使用触发、兼容性成本、测试列表。现行spec本来不列.pth，应归支持扩展，不写成已违反现行合同的bug。建议标签spec；若上游选择不支持，也接受明确诊断/文档作为另一个独立提案。

### Issue B：应用回放教程，第二项

建议标题：`docs: add a single-case multimodal agent workflow with retrieval evidence`。

先链接Lingshu已公开的固定提交与证据，列明实际完成fixture/local/live哪层；提议在docs/tutorials新增教程并在docs/run.md、docs/index.md导航、CHANGELOG加入口。若导航文件是生成文件，按仓库机制更新来源，不直接改生成结果。

范围是：图像显式输入、实际角色参数、依赖注入接回原编排、检索与搜索回执、辅助资产、失败/预算边界和相应diff。位置/范围由维护者确认后按其指定任务实施。它不重复独立工作区教程，不以judge_only或ACL/NeurIPS导出为主题。

### Issue C：后续Agent/RAG状态提案，按需

只有案例显示现有custom/文件声明确实不足时再提。一次只讨论一种缺失表达，例如检索结果身份/顺序的可移植表示；附现有方案、剩余摩擦与兼容性。D-06/D-22/D-18的边界不能被本计划解除。没有具体不足就不制造第三个issue。

准备每份issue正文为本机文件，人工审阅后分别提交，不把计划里的多个方向塞入一个issue：

```bash
gh issue create --repo EnumaElish123/ReproLLM --label spec \
  --title "spec(lock): include .pth files in local directory weight inventory" \
  --body-file "$TRIAL_ROOT/public/issue-pth.md"
gh issue create --repo EnumaElish123/ReproLLM \
  --title "docs: add a single-case multimodal agent workflow with retrieval evidence" \
  --body-file "$TRIAL_ROOT/public/issue-recipe.md"
```

这两份正文由上述事实准备，不能预填未发生的run ID、PASS或维护者决定。保存返回的issue URL/编号。如果不是仓库协作者、无法加标签，在正文说明请求spec分类并使用网页spec模板提交；不要因标签失败重复创建issue。

## 16. 两个ReproLLM PR分别实施，禁止混合

本计划明确要求走fork/issue/PR评审流程。分支使用codex/前缀，每个PR从届时upstream/main开始，避免带入另一个PR的未合并改动。

### PR A：只改.pth格式集合与相应规范/回归

依赖：Issue A的精确提案被接受，任务进入指定执行文档。进入之前可以完善复现，不能把尚未接受的行为修改当作已批准开发。

允许修改：src/reprollm/lock/local_resolver.py、tests/unit/lock/test_local_resolver.py及必要CLI回归、规范§4.3后缀说明、CHANGELOG。禁止顺便更改local模型config要求、revision优先级、默认漂移、redaction/engine、schema或dependencies；“零权重目录提示”另议。

验收先RED再实现：

- 仅含.pth目录被纳入size/hash清单，且字节变化使weights.sha256改变。
- 固定config+变化.pth，config_sha256不变、weights.sha256变；revision原策略保留。
- 混合现有后缀与.pth，大小/排序/摘要确定；大文件默认跳过hash、显式开关后hash。
- 显式.pth文件继续正常；不支持后缀及空/config-only目录保留旧行为。
- 路径边界不放宽，不反序列化checkpoint，不导入torch，测试全离线。
- 本计划不预定大小写.pth策略；实现须明确是否沿用现有suffix.lower()，并覆盖它。
- 所有既有schemas字节不变；无snapshot变化则明确记录“无”。

```bash
cd "$RP_ROOT"
git fetch upstream
git switch -c codex/local-pth-weights upstream/main
uv sync --locked --dev
uv run pytest -q tests/unit/lock/test_local_resolver.py
```

上面先运行既有测试；随后加入被接受的验收测试，保存RED，再实现并GREEN，不能仅展示最后一次通过。一任务一Conventional Commit，引用真实任务ID、issue和spec§4.3。

### PR B：只交付已验证的多模态回放教程

允许修改：一个新教程Markdown、必要导航链接、CHANGELOG。真实Lingshu代码和大型证据留在下游固定提交；教程提供必要片段及恢复步骤，单PR尽量控制约400行。若必须新增可执行fixture/测试/工具，按代码/工具会话处理或拆PR，不能套纯文稿例外。

内容必须出现：实际入口/源码pin、fixture与真实模式区别、图像argv、模板/渲染消息区别、配置消费见证、检索/搜索证据、.pth支持边界、完整默认finding、正常/失败/漂移实例。只有已完成的层才写“验证通过”。

从最新main新建独立分支，不复制PR A尚未合并的产品更改：

```bash
cd "$RP_ROOT"
git fetch upstream
git switch -c codex/multimodal-case-recipe upstream/main
```

如果另一个分支有未提交改动，先保存该任务再切换；不stash隐藏相关证据，不force reset。PR A是否合并决定教程应写当前限制还是新支持，明确最低源码/发行版本，不要求PR A必须先合并才能提交准确教程。

## 17. 提交前验证：产品PR与文稿PR分别要求

### 17.1 产品PR完整质量门

在最终输入上运行一次，不在每项修改后重复全量。以下与当前CI相匹配；执行时再读当前quality.yml，不复制旧测试数：

```bash
cd "$RP_ROOT"
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ skills/reprollm/scripts/
uv run pytest -q --cov=reprollm --cov=skills/reprollm/scripts \
  --cov-report=xml --cov-report=term --cov-fail-under=85
uv run coverage report --fail-under=100 --include='src/reprollm/core/redaction.py'
uv run pytest -q -m security
uv run python scripts/gen_rules_doc.py --check
uv run python scripts/gen_profiles_doc.py --check
uv run python scripts/gen_cli_doc.py --check
uv run python scripts/gen_quickstart.py --check
for example in examples/*/; do
  uv run reprollm audit "$example" --level 1 --fail-on never --format json > /dev/null
  if [ "$example" = "examples/lm_eval_replay/" ]; then
    test ! -e "$example/reprollm.lock"
  else
    uv run reprollm lock --check "$example"
  fi
done
uv run reprollm audit . --format github --fail-on never
uv run python - <<'PY'
from pathlib import Path
import subprocess, tempfile
with tempfile.TemporaryDirectory(prefix="reprollm-schema-") as temp:
    target = Path(temp)
    subprocess.run(["reprollm", "schema", "export", "--out", str(target)], check=True)
    expected = {p.relative_to("schemas"): p.read_bytes() for p in Path("schemas").rglob("*.json")}
    actual = {p.relative_to(target): p.read_bytes() for p in target.rglob("*.json")}
    assert actual == expected, "inspect schema changes; do not overwrite tracked baseline"
PY
```

上面包含当前CI的examples有效性/lock freshness和self-audit；官方Action与跨平台矩阵在PR CI核对。首次fork PR可能等待维护者批准CI，状态记action_required，不写PASS。环境不支持所需Git选项时按验证流程先诊断，不删除失败测试。

### 17.2 产品PR：fresh五项目Gate A及受影响lock

按val.md核对五个目标，不把Lingshu加成自动必跑的第六套gold：

| checkout | repository | pin |
|---|---|---|
| lm-evaluation-harness | EleutherAI/lm-evaluation-harness | b954108c9baaaa934b4ad842033b31a97ee30816 |
| FastChat | lm-sys/FastChat | 587d5cfa1609a43d192cedb8441cac3c17db105d |
| LlamaFactory | hiyouga/LlamaFactory | 673048c6a543cbbeaed5b8444b8223dc4e23c721 |
| HarmBench | centerforaisafety/HarmBench | 8e1604d1171fe8a48d8febecd22f600e462bdcdd |
| llm-dp-finetune | jyhong836/llm-dp-finetune | 7f8b5dff4b92aae90ceccce3ec959b48307bed9e |

首次准备每个目标：在独立dogfooding目录git clone其表列repo，git checkout --detach表列pin；不要修改原checkouts。每个都记录git rev-parse HEAD、git status --porcelain=v1 --untracked-files=all、git remote get-url origin；任一缺失/脏状态停止该目标验收。

以下给出首次准备和fresh L0采集的完整循环，使用当前产品PR的开发环境，不能继续用第3步旧baseline wheel：

```bash
GATE_ROOT="$TRIAL_ROOT/gate-a"
CANDIDATE_RP="$RP_ROOT/.venv/bin/reprollm"
mkdir -p "$GATE_ROOT"
while read -r name repository pin; do
  CHECKOUT="$GATE_ROOT/$name"
  if [ ! -d "$CHECKOUT" ]; then
    git clone "https://github.com/$repository.git" "$CHECKOUT"
    git -C "$CHECKOUT" checkout --detach "$pin"
  fi
  test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$pin"
  test -z "$(git -C "$CHECKOUT" status --porcelain=v1 --untracked-files=all)"
  test "$(git -C "$CHECKOUT" remote get-url origin)" = "https://github.com/$repository.git"
  capture 0 "gate-a-$name" "$CANDIDATE_RP" audit "$CHECKOUT" \
    --level 0 --format json --fail-on never
done <<'TARGETS'
lm-evaluation-harness EleutherAI/lm-evaluation-harness b954108c9baaaa934b4ad842033b31a97ee30816
FastChat lm-sys/FastChat 587d5cfa1609a43d192cedb8441cac3c17db105d
LlamaFactory hiyouga/LlamaFactory 673048c6a543cbbeaed5b8444b8223dc4e23c721
HarmBench centerforaisafety/HarmBench 8e1604d1171fe8a48d8febecd22f600e462bdcdd
llm-dp-finetune jyhong836/llm-dp-finetune 7f8b5dff4b92aae90ceccce3ec959b48307bed9e
TARGETS
```

该循环只完成身份核对与采集，还没有完成gold比较或受影响lock扩展层。已有正式checkout可改用其位置，但必须保留同样的身份与干净断言。待变更命令的测试使用另建的一次性副本，不能直接初始化这些原checkout。

每个目标在原干净checkout执行本候选：
`reprollm audit CHECKOUT --level 0 --format json --fail-on never`，保留完整stdout/stderr及诊断；受影响lock在各自一次性副本上按现有固定输入执行offline lock/check及本地权重回归，不改正式checkouts。具体manifests、独立预期取val.md§8.1.1/§8.1.2及其版本化input catalog，逐个保留模型文件来源和hash。

先写“改动→受影响命令→独立预期”：本PR为local目录扩展，L0无预期delta；lock对含.pth的合格输入只增加该类权重清单及相应hash/size，对不含.pth输入不应改变。实际catalog若没有.pth，单列本任务夹具，不伪造五项目都消费了该后缀。

使用已有独立gold JSON或依据val.md和源文件审阅后冻结的expected；expected缺失须先补独立审阅，不能复制当前输出。可用：

```bash
uv run python scripts/compare_gold.py --audit-metadata EXPECTED_AUDIT ACTUAL_AUDIT
```

两个参数在执行前替换成该目标实际证据文件。比较完整rule/profile/hint/path/line/diagnostics和生成文件，计数相同不够；脚本不替代文本/文件/退出码和checkout身份核对。记录每条命令、目标pin、候选SHA、耗时、expected/actual、所有差异。任何目标缺失或未解释delta阻止Gate A完成；资源不具备记BLOCKED。

普通产品PR不自动触发整套发布Gate B；如本次被指定里程碑/发布，再执行val.md当前全部适用资源矩阵。此计划不请求发版，也不拿本试用的local/live替代五项目正式Gate B。

### 17.3 纯文稿PR

检查所有新增链接/anchor、恢复命令、版本和源码pin、生成导航freshness；在一次性目录实际重跑教程所称步骤，检查run/diff/export例子。未改变生成来源时对应freshness记NOT_APPLICABLE并解释。完整产品suite/Gate A/B按文稿例外记NOT_APPLICABLE，不抄历史PASS；一旦新增脚本/测试/fixture或产品改动重新分类。

两类PR都在docs/dogfooding的会话报告中记录真实验证状态、任务提交hash、例外/阻塞；PR正文按模板填写，不把计划清单勾为已完成。

## 18. 推送自己的分支、创建PR、等待CI并回应审查

每个PR各自执行；先确认当前分支、staged diff、质量门/适用Gate A与报告完整。示例是产品PR，文稿PR替换对应分支和正文：

```bash
cd "$RP_ROOT"
git diff --check
git diff --stat upstream/main...HEAD
git status --short
git push -u origin codex/local-pth-weights
gh pr create --repo EnumaElish123/ReproLLM --base main \
  --head "$GH_LOGIN:codex/local-pth-weights" \
  --title "feat(lock): include .pth files in local model directories" \
  --body-file "$TRIAL_ROOT/public/pr-pth.md"
```

正文写真实issue引用、accepted task ID/spec、范围、RED/GREEN、完整门和五项目结果、schema/snapshot状态、Lingshu证据链接；不在未确定关闭关系时写Closes。文稿PR用独立pr-recipe.md。提交多行正文使用文件，不在shell命令里拼接包含敏感内容的字符串。[gh PR创建文档](https://cli.github.com/manual/gh_pr_create)、[issue创建文档](https://cli.github.com/manual/gh_issue_create)。

取得真实PR编号后：

```bash
gh pr checks PR_NUMBER --repo EnumaElish123/ReproLLM --watch
gh pr view PR_NUMBER --repo EnumaElish123/ReproLLM \
  --json url,headRefOid,state,mergeStateStatus,statusCheckRollup
```

PR_NUMBER替换为返回编号。核对当前head SHA的五平台CI；旧head成功不替代新head，pending/action_required/failed均保留。按审查追加相关提交，不擅自合并或发布；产品改动使相关验证失效时重跑最终门。上游merge后记录merge SHA/CI与issue状态，必要时补教程版本说明。

若通过Codex实际创建PR，应按宿主要求把返回URL附到当前任务；本计划没有创建PR，所以没有预造附件。

## 19. 与Mosionca2的实质差异及最终交付清单

截至本次核查，Mosionca2的[#11](https://github.com/EnumaElish123/ReproLLM/issues/11)/[PR #12](https://github.com/EnumaElish123/ReproLLM/pull/12)覆盖本地研究条件隔离、offline lock、bindings、diff、历史export；[#13](https://github.com/EnumaElish123/ReproLLM/issues/13)是输入候选文档澄清；[#14](https://github.com/EnumaElish123/ReproLLM/issues/14)是本地Qwen3 judge_only已有答案评分及教程建议。不要重复归为API judge。

本计划必须交出以下独立证据，不能只换模型/仓库名：

| 差异点 | 最低可审阅交付 |
|---|---|
| 多模态输入 | 显式图片文件证据、实际消费hash、原stdin边界对照 |
| 多阶段协作 | 共用原编排的run_case；角色真实调用、迭代上限、停止/失败见证 |
| 外部状态 | RAG文档身份/排序、索引构建来源、搜索返回实际进入报告的证据 |
| 辅助模型格式 | .pth目录/文件/固定config的完整CLI对照及精确支持提案 |
| 贡献范围 | Lingshu适配PR、ReproLLM独立教程PR、经接受的窄格式PR，状态分别记录 |

最终按实际完成填写，不预勾：

- [ ] 两仓库身份/起点/改造SHA、wheel hash、环境恢复材料齐全。
- [ ] LS01–LS09已实现，fixture真实编排通过；未用独立toy替代业务链。
- [ ] baseline、正向矩阵、负例、失败记录与完整finding已归档。
- [ ] local真实组件/分块比较已完成，或明确列出缺失资产和恢复动作。
- [ ] live已完成或BLOCKED，人工fixture与真实调用数量分别报告。
- [ ] .pth问题在最新upstream/main复测；若已修复，不再重复实现。
- [ ] 可公开案例与来源/独立性披露已审阅，未发布私密资产。
- [ ] issue提案有具体范围，维护者决定与上游任务号已记录。
- [ ] 文稿PR/产品PR分开；各自适用验证及当前head CI有证据。
- [ ] 没有把仅计划、mock通过、child成功或本工具记录成功写成科学/临床效果证明。

本手册自身的交付验证是命令接口核对、Markdown链接/格式、示例schema及Bash/Python片段检查；并不把未来执行步骤记为已完成，也不替代上游接受提案与本轮真实资源试用。

本次实际完成：32个Bash块语法检查、7个内嵌Python片段语法检查、3个YAML示例解析、manifest/profile的当前schema验证、用户profile实际继承解析（core/inference/trial_policy）、2个本地链接和空白检查。另在临时合成项目执行5条CLI，确认prompt-only时lock check仍为0且文件一致性审计报告CRITICAL，manifest改变后lock check为1。fork、安装Lingshu真实依赖、下游实现和资源场景均未代执行。
