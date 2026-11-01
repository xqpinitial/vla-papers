# 🤖 Harness Papers — 2026 具身智能 Harness 范式精读库

> **冻结底座，进化外壳。** 25 篇论文/综述的六站深读（动机 → 架构 → 机制 → 基准 → 信息流 → 例子 → 局限），覆盖 VLA、世界模型与通用 Agent 的组织层（Harness）研究全景。
>
> 🌐 **在线阅读**：`https://<YOUR_USERNAME>.github.io/harness-papers/`（启用 Pages 后）
> 📮 同步发布于知乎专栏：[@initial_dream](https://www.zhihu.com/people/initial_dream)

所有数字均对照论文 arXiv 原文（LaTeX 源）核实；每篇附论文原图与原始材料索引。

## 🗓 时间线总表（2026）

| 日期 | 论文 | arXiv | 一句话 | 类别 |
|---|---|---|---|---|
| 03-23 | CaP-X | [2603.22435](https://arxiv.org/abs/2603.22435) | Code-as-Policy 研究平台四件套 | 评测与基准 |
| 06-17 | RATs | [2606.19419](https://arxiv.org/abs/2606.19419) | 任务到来前先"玩"出技能库 | 技能库与探索 |
| 06-30 | ASPIRE | [2607.00272](https://arxiv.org/abs/2607.00272) | trace + 技能库 + verifier 门控 | 技能库与探索 |
| 07-03 | DreamSteer | [2607.02865](https://arxiv.org/abs/2607.02865) | 潜空间世界模型预演排序 | 世界模型与想象 |
| 07-06 | GaP | [2607.05369](https://arxiv.org/abs/2607.05369) | 策略 = 可自学习的计算图 | 编码 Agent 与系统 |
| 07-09 | **Harness VLA** | [2607.08448](https://arxiv.org/abs/2607.08448) | 冻结 VLA 当原语 + 双层记忆 | 冻结模型编排 |
| 07-15 | HarnessBank | [2607.13683](https://arxiv.org/abs/2607.13683) | harness 自进化：基因库 + 门控 | 通用 Agent |
| 07-20 | RoboHarness | [2607.18060](https://arxiv.org/abs/2607.18060) | 异构策略编排 + Memory Bridge | 冻结模型编排 |
| 07-27 | HumanCLAW | [2607.27180](https://arxiv.org/abs/2607.27180) | VLM 具身决策解耦评测 | 评测与基准 |
| 08-16 | Zetta ζ | [2608.16590](https://arxiv.org/abs/2608.16590) | 代码化 critic 动作频率监控 | 冻结模型编排 |
| 08-17 | HarnessEval-W | [2608.16859](https://arxiv.org/abs/2608.16859) | 世界模型评测 agentic 化 | 评测与基准 |
| 08-29 | EMERGE-Policy | [2608.29896](https://arxiv.org/abs/2608.29896) | 图结构多智能体系统级涌现 | 冻结模型编排 |
| 09-01 | EmbodiedSkills | [2609.01281](https://arxiv.org/abs/2609.01281) | skill 契约 + guarded runtime | 冻结模型编排 |
| 09-22 | RoboDawn | [2609.22966](https://arxiv.org/abs/2609.22966) | 冻结 VLM + 离散命令 + ICL | 智能迁移与 ICL |
| 09-23 | EmbodiedSWE | [2609.27308](https://arxiv.org/abs/2609.27308) | 编程 Agent 当"教师"产数据 | 编码 Agent 与系统 |
| 09-24 | HarnessPAI | [2609.29166](https://arxiv.org/abs/2609.29166) | 代码接口组织冻结 VLA/WAM 原语 | 冻结模型编排 |
| 09-27 | RoboHarn-Evo | [2609.37583](https://arxiv.org/abs/2609.37583) | 双层物理知识自演化 | 知识与记忆 |
| 09-28 | RoboFoundry | [2609.32862](https://arxiv.org/abs/2609.32862) | System-as-Policy 文件系统进化 | 编码 Agent 与系统 |
| 09-29 | AdaHVLA | [2609.29204](https://arxiv.org/abs/2609.29204) | 修订图驱动的自适应 harness | 冻结模型编排 |
| 10-09 | Harness² | [2610.14224](https://arxiv.org/abs/2610.14224) | 谷歌递归 agent harnessing | 通用 Agent |
| 10-13 | Orchestra | [2610.18352](https://arxiv.org/abs/2610.18352) | 规划/执行管理/动作生成三分 | 冻结模型编排 |
| 10-25 | RAX | [2610.22496](https://arxiv.org/abs/2610.22496) | 在线探索 scaling law | 技能库与探索 |

## 📚 专题综述

- [2026 具身智能 Harness 全景：18 篇时间线与分类学](papers/overview-timeline.html)
- [六篇详细对比：RAX · HarnessPAI · ASPIRE · CaP-X · RATs · Zetta](papers/comparison.html)
- [VLA 的 Action 表示方式全景（AR/VQ/FAST/扩散/流匹配/CVAE/EBM）](papers/action-representations.html)

## 🧭 六条光谱轴

1. **底座**：VLA / VLM / coding agent / 通用 LLM
2. **外层介质**：记忆与知识 · 代码与技能 · 想象力（WM）· 评测器
3. **学习时机**：不学 → 探索期 → 跨 rollout → 反哺权重
4. **动作来源**：冻结 VLA 出动作 ↔ 代码出动作 ↔ 混合
5. **门控哲学**：无人值守成功 / verifier 复现 / 统计门控 / 契约前置 / 行为对比
6. **提升 vs 测量**：增强能力 vs 评测能力

## 📁 仓库结构

```
├── index.html            # 论文库首页（搜索/分类/时间线/卡片）
├── papers/               # 25 个详情页（每篇附论文原图）
├── data/papers.json      # 结构化索引（供搜索与筛选）
├── assets/               # 样式、脚本、论文原图
└── deploy.sh             # 一键部署 GitHub Pages
```

## 🚀 部署

```bash
GH_USER=<你的GitHub用户名> GH_TOKEN=<repo权限PAT> ./deploy.sh
```

自动：创建仓库 → 推送 → 启用 GitHub Pages → 输出在线地址。

## 📮 相关

- 知乎专栏同步发布：[@initial_dream](https://www.zhihu.com/people/initial_dream)
- 相关 awesome 列表：[awesome-agentic-robotics](https://github.com/w-xb/awesome-agentic-robotics)

## License

内容 CC BY 4.0；引用请注明出处。
