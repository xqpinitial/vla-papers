# WALL-X 精读：WALL 系列的开源工程栈，不是一篇论文

WALL-X 不是论文，是 X Square Robot 给 WALL 系列模型提供的开源工程栈：下载 checkpoint、准备 LeRobot 数据、写 YAML、跑后训练、起 policy server、做 LIBERO / 真机评估的代码入口。本地章节给出的可核验数字都是工程口径：支持 4 个 checkpoint 变体、Python 3.10、LeRobot 固定 commit `c66cd401767e60baece16e1cf68da2824227e076`、`MAX_JOBS=8` 编译 CUDA op、Apache-2.0 license。章节没有给任何模型性能、显存或延迟数字；本文按“工程导航”写，不按“论文方法”写。

> 来源约定：本文事实只取自本仓库 [WALL-X 导读](../../08WALL-X开源工程框架导航/README.md)，没有用模型记忆补写仓库细节。
>
> 标注约定：`【章节】` 指本地导读自己的归纳；`【章节转述】` 指导读对官方 README、workspace guide 或模型页的转述。WALL-X 自身 commit 章节未给。
>
> 版本：无 arXiv；WALL-X commit 章节未给；LeRobot 固定 commit `c66cd401…`；workspace README 当前面向 Wall-OSS-0.5；查阅日期 2026-11-01。

## 1. 仓库和资源状态

| 资源 | 入口 | 说明 |
| :--- | :--- | :--- |
| 论文 | 无独立论文 | 章节明确说明 WALL-X 是工程框架；方法读 [WALL-OSS 精读](wall-oss.md)，不把工程栈当论文 |
| 源码 | [X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x) | 训练、推理、评估、serving 与 CUDA op |
| 权重 | [x-square-robot/wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5)、[wall-oss-flow](https://huggingface.co/x-square-robot/wall-oss-flow)、[wall-oss-fast](https://huggingface.co/x-square-robot/wall-oss-fast) | 当前 README 列出 4 个变体；`wall-oss-flow-0.1` 章节只给名称、未给链接 |
| 项目页 | 章节未给 | 本地章节没有出现项目页 |
| 官方文档 | [workspace/README.md](https://github.com/X-Square-Robot/wall-x/blob/main/workspace/README.md) | Wall-OSS-0.5 的 fine-tune、simulation evaluation、real robot deployment |
| License | Apache-2.0 | 章节转述仓库 license |

**版本提示：**WALL-X README 的 News 同时提到 WALL-OSS、WALL-OSS-0.5 和 WALL-WM，容易让人误以为“WALL-WM 完整代码已能一键复现”。更谨慎的理解是：WALL-X 当前明确提供的是 WALL 系列开源模型的训练与推理栈；WALL-WM 的论文和 PDF 已公开，但完整事件级 world action model 的权重、事件数据和训练 recipe 不能直接从 WALL-X 模型列表中确认。【章节】

## 2. 核心问题：复现从哪一步开始

一句话：WALL-X 重新分配的是“复现工作的起点”——不是从论文方法读起，而是从 checkpoint、LeRobot 数据、YAML 配置、训练脚本和 policy server 这条工程链路跑起。【章节】

![WALL-X 工程链路](assets/wall-x-architecture.svg)

图 1：WALL-X 从 checkpoint 与数据到训练、服务和评估的工程链路。自绘，依据本地导读；不是论文原图。

对比两种入口：论文复现包（方法 + 权重 + 数据 + 训练 recipe 一起给）→ 工程栈（把已有 checkpoint 接到数据、配置、训练和服务上）。【章节】

- 节省了什么：不用自己搭训练/推理脚手架，直接用官方 YAML、serving 与评估入口。
- 保留了什么：WALL-OSS 的模型结构与动作分支；WALL-X 只负责把它们组织成可运行的工程。
- 代价是什么：环境版本、CUDA 编译、LeRobot commit、checkpoint 路径、数据格式都要对齐；任一项错都可能跑不起来。
- 读者该学什么：先判断“我要跑哪个 checkpoint、用哪条入口”，再谈复现；不要把工程链路通当成论文复现。

## 3. 仓库里有什么

### 3.1 六项能力

| 能力 | 章节给出的说明 | 对复现的意义 |
| :--- | :--- | :--- |
| LeRobot data preparation | 把机器人轨迹组织成 LeRobot 可读格式 | 降低接入自有数据的成本 |
| model configuration | 用 YAML 指定 checkpoint、processor、动作维度、数据路径与训练超参 | 配置是主要“接口”，不是代码 |
| flow-matching action branch | 面向连续动作生成，适合高频控制 | 对应 `wall-oss-flow` 一类 checkpoint |
| FAST action branch | 面向更快的 action token / action prediction | 对应 `wall-oss-fast` |
| serving and evaluation utilities | 把 policy 作为服务启动，由仿真或机器人客户端调用 | 模型进程与控制进程解耦 |
| CUDA operator sources | 安装时编译自定义 CUDA op | 要求 CUDA / PyTorch / flash-attn 匹配 |

这也说明复现 WALL-X 的难点不在“命令多”，而在环境版本、CUDA 编译、LeRobot 版本、checkpoint 路径和数据格式。【章节】

### 3.2 在整条链路中的位置

| 阶段 | 输入 | 输出 | 章节给出的入口 |
| :--- | :--- | :--- | :--- |
| 取模型 | HF checkpoint | 本地模型目录 | `huggingface-cli download` 示例 |
| 取数据 | LeRobot 格式数据（LIBERO 或自有） | 本地数据目录 | LeRobot 数据准备 |
| 写配置 | checkpoint + data + 动作维度 + 超参 | `workspace/*.yml` | `workspace/example/*.yml` |
| 训练 | 配置 + checkpoint + 数据 | fine-tuned checkpoint | training script |
| 服务 | fine-tuned checkpoint | policy server | serving utilities |
| 评估 / 部署 | policy server | LIBERO 结果或机器人动作 | evaluation utilities |

章节的原话判断是：WALL-X 不是只提供一个模型定义文件，而是提供一套能围绕 WALL-OSS 跑实验的工程框架。【章节】

### 3.3 与 WALL-WM 的边界

WALL-X 的 News 提到 WALL-WM，但章节把边界写得很清楚：

- WALL-X 当前明确提供的是 WALL 系列开源模型的训练和推理栈。
- `workspace/README.md` 当前重点是 Wall-OSS-0.5 的 fine-tune、simulation evaluation 和 real robot deployment。
- WALL-WM 的论文与 PDF 已公开，但完整事件级 world action model 的权重、事件数据和训练 recipe 还不能直接从 WALL-X 模型列表中确认。

因此，WALL-X 不能自动等同于 WALL-WM 的完整论文复现包。【章节】

## 4. 评估入口与可核验状态

章节没有给任何 benchmark 分数、任务数、trials 或种子；下表只整理“有哪些评估入口、各自缺什么”。【章节】

| 入口 | 前置条件 | 章节未给 | 读法 |
| :--- | :--- | :--- | :--- |
| LIBERO simulation evaluation | LIBERO 环境、对应 checkpoint、配置 | suite 选择、任务数、trials、成功率 | 最接近标准评测，但仍需自己记录协议 |
| 自有 LeRobot 数据后训练 + validation | LeRobot 数据、动作维度匹配 | 数据规模、验证指标、早停策略 | 只能验证链路与数据适配 |
| fake inference / serving smoke test | policy server 能起 | 动作质量、延迟 | 证明服务连通，不证明策略有效 |
| 真实机器人部署 | 硬件接口、安全控制 | 任务、演示数、测试次数、成功率 | 章节明确说不应在通用教程里默认可用 |

读法：这张表能回答“从哪开始跑”，不能回答“跑出来多少分”。如果报告里要写性能，必须补自己的协议与数字，不能引用本章。

来源：本地 WALL-X 导读 §3、§8。

## 5. 选择矩阵：checkpoint、分支与路线

本章是工程导航，没有消融实验；下面的“对照”是章节给出的选择建议，不是实验结果。

| 目标 | 推荐选择 | 原因 | 风险 |
| :--- | :--- | :--- | :--- |
| 第一次跑通 WALL-X | `wall-oss-0.5` + 当前 main 分支 | workspace README 明确面向 Wall-OSS-0.5 | 其他 checkpoint 可能不兼容当前代码 |
| 测试 flow / FAST 老模型 | 按官方说明切到对应历史 commit | 官方建议使用旧版本代码 | commit 与依赖需一起回退 |
| 接入 LeRobot 社区工具 | `policy.type=wall_x` | 已有 LeRobot 数据与 `lerobot-train` 经验时更顺 | LeRobot 版本与 WALL-X 依赖可能冲突 |
| 复现 WALL-WM 论文 | 暂不建议 | 缺权重、事件数据与训练 recipe | 容易把“News 提到”误当成“已开源” |

读者该不该复用：如果目标是“在自己的数据上后训练一个 WALL-OSS”，选 `wall-oss-0.5` + 当前代码最稳；如果目标是复现论文方法，应该先读 WALL-OSS 论文，不要把 WALL-X 当成方法论文。章节没有给不同 checkpoint 的成功率对照，因此无法判断哪个模型“更好”，只能判断哪个“更匹配当前代码”。【章节】

来源：本地 WALL-X 导读 §4、§9。

## 6. 环境与延迟怎么读

章节没有给模型延迟、显存或控制频率，只给了安装约束与部署形态。

| 检查项 | 章节给出的要求 | 为什么重要 |
| :--- | :--- | :--- |
| Python | 3.10 | 官方 conda 环境即 `python=3.10` |
| 依赖 | `pip install -r requirements.txt` | 基础包版本由仓库锁定 |
| dmuon | `pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"` | 从 git 安装，版本随上游变化 |
| LeRobot | checkout `c66cd401767e60baece16e1cf68da2824227e076` | 章节给出的固定 commit |
| LeRobot 安装方式 | `pip install --no-deps -e .` | 避免覆盖 WALL-X 依赖 |
| CUDA op 编译 | `MAX_JOBS=8 pip install --no-build-isolation -e .` | 安装时编译自定义算子 |
| CUDA / PyTorch / flash-attn | 必须匹配 | 章节把版本冲突列为最常见失败原因 |

读法：

- 这张表能判断“环境是否能装上”，不能判断“推理有多快”。章节没有给任何计时。
- policy server 把模型进程与仿真/机器人控制进程分开，这有利于部署，但不等于低延迟；网络与序列化开销章节未测。
- `1000 / 延迟` 只说明策略调用频率，不等于动作执行频率或端到端反馈频率。
- 需要显存数字时只能本机测：分别记录加载显存、单次前向峰值、完整请求延迟。

来源：本地 WALL-X 导读 §5、§8。

## 7. 训练配置与 smoke test

章节没有给 GPU 数、学习率、warmup、EMA 或默认训练步数，也没有给损失或动作维度的公式；训练配置的核心是 YAML 字段。

| 配置字段 | 作用 | 章节是否给默认值 |
| :--- | :--- | :--- |
| `model.config_path` | 指向 WALL-OSS checkpoint 里的 `config.json` | 未给 |
| `model.processor_path` | 指向 Qwen2.5-VL processor | 未给 |
| `model.pretrained_path` | 指向预训练权重 | 未给 |
| dataset path | 指向 LeRobot 格式数据 | 未给 |
| robot DOF | 动作维度必须与机器人数据匹配 | 未给 |
| action branch | flow / FAST / 其他预测模式 | 未给 |
| batch size / steps | 根据显存调整 | 未给 |

如果把复现目标设成 smoke test 级别，章节给的验收标准是：

- 模型和 processor 能加载。
- LeRobot 数据能读取。
- 训练 forward/backward 能跑几步。
- 输出 checkpoint 能保存。
- evaluation 或 fake inference 能返回动作。

这类 smoke test 证明的是工程链路通了，不证明模型达到论文性能。【章节】章节没有给有效 batch 的组成，因此无法用 `N_GPU × B_micro × N_accum` 反推；也没有给单卡配方，不能把示例 YAML 当成论文训练配置。

三级复现的可行性：

1. **理解与评测**：可行。按 §9 安装、下载 checkpoint，先跑 fake inference 或单任务 LIBERO。
2. **训练链路**：可行但需自定超参。用少量 LeRobot 数据跑通 forward/backward 与保存。
3. **完整训练**：章节未给配方，且明确说“当前不写手把手复现”，不能声称可复现论文训练。

来源：本地 WALL-X 导读 §7、§10。

## 8. 代码阅读顺序

| 顺序 | 路径 / 入口 | 阅读重点 |
| ---: | :--- | :--- |
| 1 | 仓库根 `README` | 支持的模型、能力列表、News 与边界 |
| 2 | `workspace/README.md` | Wall-OSS-0.5 的 fine-tune、sim evaluation、real robot deployment |
| 3 | `workspace/example/*.yml` | 模型路径、processor、数据、动作维度、batch/steps |
| 4 | `requirements.txt` | 基础依赖与版本约束 |
| 5 | CUDA operator sources | 安装时编译哪些算子、依赖什么 CUDA 版本 |
| 6 | serving / evaluation utilities | policy server 的请求-响应格式与客户端调用方式 |
| 7 | LeRobot checkout | 固定 commit `c66cd401…` 与 `policy.type=wall_x` 的接口 |

坑（章节明确提示）：

- **LeRobot 必须用指定 commit**：`c66cd401767e60baece16e1cf68da2824227e076`，并且要 `--no-deps` 安装，否则会覆盖 WALL-X 依赖。
- **CUDA op 要能编译**：`MAX_JOBS=8 pip install --no-build-isolation -e .` 不是可选步骤；编译失败通常不是模型问题。
- **workspace README 只面向 0.5**：使用 FLOW / FAST 时按官方说明切旧版本代码，不要拿 0.5 的 YAML 直接套。
- **路径不要写死作者机器**：章节建议用 `MODEL_ROOT`、`DATA_ROOT`、`WALLX_ROOT` 三个环境变量，YAML 里引用它们。

## 9. 从安装到 smoke test

以下命令来自章节转述的官方 README，未在本机验证；路径与版本需按自己的机器替换。

### 9.1 创建环境

```bash
conda create --name wallx python=3.10
conda activate wallx

pip install -r requirements.txt
pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"

git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout c66cd401767e60baece16e1cf68da2824227e076
pip install --no-deps -e .
cd -

MAX_JOBS=8 pip install --no-build-isolation -e .
```

### 9.2 下载模型与 processor

```bash
huggingface-cli download x-square-robot/wall-oss-0.5 \
  --local-dir /path/to/wall-oss-0.5

huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

### 9.3 固定路径

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export WALLX_ROOT=/path/to/wall-x
```

### 9.4 先跑 smoke test，再谈评估

```bash
# 1. 确认模型与 processor 能加载
# 2. 确认 LeRobot 数据能读取
# 3. 跑几步 forward/backward，确认 checkpoint 能保存
# 4. 起 policy server，做 fake inference 或单任务 LIBERO
# 具体命令章节未给，按 workspace/README.md 与仓库示例填写
```

能证明：环境装好、模型能加载、数据能读、服务能通。不能证明：任何 benchmark 分数、真机安全、论文训练可复现。

## 10. 如何用到自己的项目

WALL-X 的价值是把“模型 → 数据 → 配置 → 训练 → 服务 → 评估”这条链路固定下来，让 WALL-OSS 可以被后训练和部署，而不是停留在论文权重。

- **接自己的 LeRobot 数据**：按 YAML 字段填 `model.*`、数据路径与 robot DOF，先跑 smoke test；代价是动作维度与归一化必须和机器人一致。
- **把 policy server 接进自己的仿真/真机**：复用请求-响应接口，把模型进程与控制进程分开；代价是要处理通信延迟与异常重启。
- **用 WALL-X 做后训练基线**：固定 `wall-oss-0.5` 与 LeRobot commit，只改数据与训练步数，得到可复现的工程基线；代价是不能把结果当成论文复现。

相关章节：[WALL-OSS 精读](wall-oss.md)、[EventVLA 精读](eventvla.md)、[WALL-X 导读](../../08WALL-X开源工程框架导航/README.md)、[01 技术专题](../01-技术专题.md)、[02 论文与评测](../02-论文与评测.md)。
