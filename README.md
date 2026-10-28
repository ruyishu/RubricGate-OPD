# RubricGate-OPD

偏序 Rubric 门控的动作级 On-Policy Distillation。

**用带前置依赖的得分点图（DAG-Rubric）当中间奖励，在 rollout 过程中决定「放行 / 改造 / 早停」，而不是在每个 token 上和老师对齐。**

例子：任务是「先查库存 a，再查价格 b 和查促销 c（两者顺序无关），最后下单 d」。老师走 `a→b→c→d`，学生走 `a→c→b→d`。
token 级 OPD 会在学生迈出 `c` 的那一刻因偏离老师的顺序偏好而重罚，并让后续前缀 `[a,c]` 落入老师分布的尾部；
而基于 token 分歧的早停/接管方法（STRIDE、TISD）会把这次合法换序直接当成错误掐掉或改掉。
本仓库要验证的是：用 Rubric 判「这一步有没有推进」，能不能把「良性换序」和「关键错步」分开。

## 当前状态

仓库处于**设计阶段**：只有研究计划与方法文档，**没有任何训练代码、GPU 实验或分数**。
方法里的每一条规则都还只是待验证的假设，`docs/EXPERIMENTS.md` 里的数字全是预期指标，不是结果。
状态与证据记录在 [STATUS.md](docs/STATUS.md)。

| 内容 | 入口 |
|---|---|
| 研究计划与里程碑 | [plan.md](plan.md) |
| 方法与形式化定义 | [METHOD.md](docs/METHOD.md) |
| 与 2026 年同类工作的差异 | [RELATED_WORK.md](docs/RELATED_WORK.md) |
| 实验设计与预期结果 | [EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| 第一个可执行任务（1–2 天 pilot） | [START_HERE.md](docs/START_HERE.md) |
| 状态与证据 | [STATUS.md](docs/STATUS.md) |
| 开发约定 | [AGENTS.md](AGENTS.md) |

## 方法一句话

对每个任务构造一个 DAG-Rubric：节点是可机器验证的子目标，边是硬性先后依赖。
任意时刻用「前置已完成的节点」算出已解锁池 `U_t`。
学生在 `U_t` 内无论怎么排序都算推进（放行并给正向奖励）；碰到未解锁节点或不可逆错误才介入（老师代打一步）或早停。

## 与最近工作的关系

- **ActionOPD（2610.19820）**：把粒度从 token 提到单步动作语义，但仍是「同一状态下单步匹配」，处理不了跨步换序。
- **MENTOR（2510.18383）/ ROPD（2605.07396）**：用整轨迹工具集合或整轨迹 rubric 总分，不怕换序但丢了因果依赖，退回轨迹级稀疏奖励。
- **STRIDE（2609.14636）/ TISD（2609.30878）/ TRD（2606.08432）**：会早停或接管，但判断依据是师生 token 分歧度，分不清「合法换序」和「关键错步」。

细节与引用见 [RELATED_WORK.md](docs/RELATED_WORK.md)。

## 开工

先读 [plan.md](plan.md) 和 [METHOD.md](docs/METHOD.md)，然后按 [START_HERE.md](docs/START_HERE.md) 里的 M0 做 toy pilot。
