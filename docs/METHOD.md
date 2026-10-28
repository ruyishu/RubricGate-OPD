# 方法：偏序 Rubric 门控的动作级 OPD

本文件只写设计与定义。**没有实现，也没有验证过任何一条。** 存疑处标了「风险」或「待定」。

## 1. 记号

- 任务 $x$，学生策略 $\pi_\theta$，冻结教师 $\pi_T$。
- 第 $t$ 步状态 $s_t = (x, a_{<t}, o_{<t})$，学生动作 $a_t^S \sim \pi_\theta(\cdot\mid s_t)$，教师参考动作 $a_t^T \sim \pi_T(\cdot\mid s_t)$。
- 动作 = 一次工具调用（工具名 + 参数）或一条面向用户的消息。一次动作在 token 序列上占一个 span。

## 2. DAG-Rubric

对每个任务构造有向无环图 $\mathcal{G}=(\mathcal{V},\mathcal{E})$：

- 节点 $v\in\mathcal{V}$：一个**可机器验证**的子目标，带
  - 验证谓词 $P_v$：在当前环境状态（或工具返回、或已提交的调用集合）上可判真假；
  - 分值 $r_v>0$；
  - 前置集合 $\mathrm{Pa}(v)\subseteq\mathcal{V}$。
- 边 $(u\to v)\in\mathcal{E}$ 表示 $u$ 必须在 $v$ 之前完成，否则 $v$ 的动作无效（例如未查库存就下单）。
- 无边的同层节点即**可交换动作**：`b` 与 `c` 的 $\mathrm{Pa}$ 相同且互不为前置，谁先谁后都合法。

已完成集合与已解锁池：

$$C_t=\{v\in\mathcal{V}: P_v \text{ 在 } \le t \text{ 的某时刻为真}\},\qquad
\mathcal{U}_t=\{v\in\mathcal{V}\setminus C_t:\ \mathrm{Pa}(v)\subseteq C_t\}$$

**构造来源**（待定，按可行性排序）：
1. 由工具 schema 与任务的参考解自动推导，凡是能被环境断言验证的才建节点；
2. 教师轨迹 + LLM 生成候选节点，再用教师 rollout 实跑一遍过滤掉永远为假的节点；
3. 兜底：退化为线性链（= 顺序严格），用于对照。

风险：DAG 画错会直接教错。需要单独统计「rubric 与真实可解路径不一致」的比例，并在实验里作为上界参考。

## 3. 三态门控

每一步先走**快路径**：轻量语义匹配（工具名 + 规范化参数）判断 $a_t^S\approx a_t^T$。
一致 → 直接给正向动作奖励，更新 $C_{t+1}$，不调用 rubric 检查器。

只有**师生动作不一致**时才触发 rubric 检查，并按结果分三种走法：

### (1) 放行 Continue
条件：$a_t^S$ 命中了 $\mathcal{U}_t$ 中的某个（些）节点，即 $V_\text{hit}=\{v\in\mathcal{U}_t: a_t^S \text{ 满足 } P_v\}\neq\varnothing$。

- 不给教师惩罚（尽管 $a_t^S\neq a_t^T$），给 $a_t^S$ 的分值 $\sum_{v\in V_\text{hit}} r_v$。
- $C_{t+1}=C_t\cup V_\text{hit}$，rollout 继续，学生顺着自己的路径走。
- 这一步是**本方法相对 token-OPD / STRIDE 唯一的关键差异**：它把「跟老师不一样」和「没推进」解绑了。

### (2) 改造 Intervene
条件：$V_\text{hit}=\varnothing$，不是不可逆错误，且该轨迹介入次数 $<M$。

- 学生这一步记负优势（用于训练惩罚）；
- 环境里**丢弃** $a_t^S$，用教师动作 $a_t^T$（要求 $a_t^T$ 落在 $\mathcal{U}_t$ 内）替换执行，把状态掰回正轨；
- 之后把控制权交还学生继续采样第 $t+1$ 步。

风险（**主要未解问题**）：被替换的那一步由教师产生，后续学生 token 处在一个「拼接前缀」上，对当前策略是 off-policy 的。
待定的处理方式：对拼接轨迹重算 $\log\pi_\theta$；loss mask 只覆盖学生自己生成的 span；对拼接点做截断重要性采样（参考 DAgger 类做法）。
是否稳定、是否需要限制 $M=1$，都还没试过。

### (3) 早停 Early-Stop
条件：出现不可逆错误（破坏性/越权动作、违反硬约束）、或 $M$ 已用尽、或连续 $K$ 步零进展。

- 立即终止该 rollout，最后一步记惩罚，省掉后续全部生成与评分开销。

## 4. 优化目标

放弃逐 token 的 KL 惩罚，改在**动作 span** 上聚合：同一 span 内所有 reasoning + action token 共享该步优势 $A_t$。

$$A_t=\mathrm{Norm}\Big(\gamma^{t}R_t + \gamma^{T-t}R_\text{outcome}\Big)_{\text{group}}$$

其中 $R_t$ 来自第 3 节的门控（放行得正分、改造记负、早停记罚），$R_\text{outcome}$ 是环境终局 0/1，$\mathrm{Norm}$ 是组内标准化（同题多条 rollout 或同状态下多候选，二选一，见 [EXPERIMENTS](EXPERIMENTS.md) 的消融）。

梯度形式与 ActionOPD / GRPO 同族（clipped surrogate + 组内基线），差别只在于**优势的来源是 rubric 门控 + 终局结果**，而不是教师 token 分布。

## 5. 成本

- 快路径（师生一致）不调用评判器，是主要省算力的地方；
- 慢路径只在分叉点触发，最坏情况是每步都分歧 —— 需要在 pilot 里量分歧率；
- 改造会产生额外一次环境交互，早停会少跑若干步，净开销待实测。

## 6. 明确不做的事

- 不比对教师的思考文本（reasoning 不作为模仿目标）；
- 不要求师生同词表（只用工具名 + 参数 + 环境状态）；
- 不在 rubric 与真实可解路径冲突时假装没事——冲突率要单独报。
