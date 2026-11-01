# VLA 论文精读索引

> 按 [`paper-deep-read`](../../../../.claude/skills/paper-deep-read/SKILL.md) 结构生成：一句话结论 → 资源 → 问题 → 结构/张量 → 主实验 → 消融 → 延迟 → 训练资源 → 代码顺序 → 复现 → 应用。
>
> 本批范围：本目录 19 个章节目录覆盖的 ~22 个工作。外部补充论文（3DGS/轻量化/harness/backbone）暂不单独成文，索引见 [02-论文与评测.md](../02-论文与评测.md) §2。
>
> 状态：22 篇全部生成（3 篇样稿 v3 + 19 篇批量）。每篇 10 节、无缺口章节、一张 `assets/*.svg` 架构图；资源表只保留论文/源码/权重/项目页（G0.5 另含数据集）。

## 1. 状态总表

| # | 论文 | 本地章节 | 状态 | 文件 |
| ---: | :--- | :--- | :--- | :--- |
| 1 | RT-1 | [RT 系列](../../03RT系列论文解读与代码分析/01RT系列论文解读与代码分析.md) | 已生成 | [rt-1.md](rt-1.md) |
| 2 | RT-2 | [RT 系列](../../03RT系列论文解读与代码分析/01RT系列论文解读与代码分析.md) | **样稿** | [rt-2.md](rt-2.md) |
| 3 | RT-X | [RT 系列](../../03RT系列论文解读与代码分析/01RT系列论文解读与代码分析.md) | 已生成 | [rt-x.md](rt-x.md) |
| 4 | OpenVLA | [OpenVLA 导读](../../02OpenVLA复现/02openvla复现.md) | **样稿** | [openvla.md](openvla.md) |
| 5 | SmolVLA | [SmolVLA 导读](../../01SmolVLA-LIBERO/01SmolVLA-libero.md) | 已生成 | [smolvla.md](smolvla.md) |
| 6 | ACT | [ACT 结构](../../04mujoco复现ACT、Pi0、SmolVLA/ACT_模型结构与数据维度.md) | 已生成 | [act.md](act.md) |
| 7 | π0 | [LeRobot 教程](../../04mujoco复现ACT、Pi0、SmolVLA/README.md) | 已生成 | [pi0.md](pi0.md) |
| 8 | DiT4DiT | [DiT4DiT 导读](../../05DiT4DiT-LIBERO/01DiT4DiT-LIBERO训练与评估.md) | 已生成 | [dit4dit.md](dit4dit.md) |
| 9 | EventVLA | [EventVLA 导读](../../06EventVLA视觉证据记忆导读/README.md) | 已生成 | [eventvla.md](eventvla.md) |
| 10 | WALL-OSS | [WALL-OSS 导读](../../07WALL-OSS开源VLA模型导读/README.md) | 已生成 | [wall-oss.md](wall-oss.md) |
| 11 | WALL-X | [WALL-X 导读](../../08WALL-X开源工程框架导航/README.md) | 已生成 | [wall-x.md](wall-x.md) |
| 12 | 3DVLA | [3DVLA 导读](../../09-3DVLA三维空间实例增强VLA导读/README.md) | 已生成 | [3dvla.md](3dvla.md) |
| 13 | PhysBrain 1.0 | [PhysBrain 导读](../../10-PhysBrain物理常识增强VLA导读/README.md) | 已生成 | [physbrain.md](physbrain.md) |
| 14 | PRTS | [PRTS 导读](../../11-PRTS强化学习原生VLA导读/README.md) | 已生成 | [prts.md](prts.md) |
| 15 | Galaxea G0.5 | [G0.5 导读](../../12-Galaxea-G0.5自回归VLA导读/README.md) | **样稿** | [galaxea-g0-5.md](galaxea-g0-5.md) |
| 16 | Dexora | [Dexora 导读](../../13-Dexora高自由度双臂灵巧VLA导读/README.md) | 已生成 | [dexora.md](dexora.md) |
| 17 | LWD | [LWD 导读](../../14-LWD真机机群强化学习导读/README.md) | 已生成 | [lwd.md](lwd.md) |
| 18 | VisualThink-VLA | [VisualThink-VLA 导读](../../15-VisualThink-VLA视觉证据推理导读/README.md) | 已生成 | [visualthink-vla.md](visualthink-vla.md) |
| 19 | Agentic-VLA | [Agentic-VLA 导读](../../16-Agentic-VLA在线适应导读/README.md) | 已生成 | [agentic-vla.md](agentic-vla.md) |
| 20 | Dexbotic-RLinf | [Dexbotic-RLinf 导读](../../17-Dexbotic-RLinf工程化VLA后训练导读/README.md) | 已生成 | [dexbotic-rlinf.md](dexbotic-rlinf.md) |
| 21 | DM0.5 / OpenDM | [DM0.5 导读](../../18-DM0.5高性能推理与OpenDM导读/README.md) | 已生成 | [dm0-5.md](dm0-5.md) |
| 22 | EVA-Client | [EVA-Client 导读](../../19-EVA-Client真机部署与评测工程导航/README.md) | 已生成 | [eva-client.md](eva-client.md) |

## 2. 三篇样稿的差异

- **RT-2**：本地章节有范式、参数量、动作 token 化与定性能力。资源表给 arXiv、项目页与 RT-1 参考源码；第 8–9 节写成“不可复现/替代读法”；新增训练/推理 SVG。
- **OpenVLA**：本地章节有架构、权重、LIBERO 评测与排错。资源表给 arXiv、源码、权重、项目页；新增组件表、归一化公式、LIBERO 协议、排错与命令骨架；新增架构 SVG。
- **G0.5**：本地章节最完整，有 27 维 ActionCodec、RVQ、记忆、五组 benchmark 与真机结果。资源表给论文 PDF、源码、权重、项目页、数据集；新增 27 维动作表、RVQ/记忆/CoT、benchmark 协议与架构 SVG。

## 3. 批量完成说明

- 资源表：论文 / 源码 / 权重 / 项目页（G0.5 另含数据集）。带 `*` 的是公开入口，未在本地章节核验；写“章节未给”的是本地导读也没有给。
- 篇幅：19 篇批量文档 208–300 行；3 篇样稿 213–261 行。全部在 skill 的 150–350 行区间。
- 图表：每篇一张 900×430 SVG，风格统一；图注标明“自绘，依据本地导读；不是论文原图”。
- 缺口：不再单列缺口章节，必要处用一行注记；未用模型记忆补写。

确认后按 #1–#22 批量生成，并回填本索引。
