# 状态

最后更新：2026-10-28

## 当前

叙事改成多轮 agent 上的逐步评分点蒸馏。仓库里有设计文档和 CPU 检查，没有训练器，没有模型分数。

## 已核对

- `docs/METHOD.md`：状态评分点、逐步奖励、KL 只在出错步打开、黑盒时的非负一致奖励、一次纠正后的回报切断。
- `docs/RELATED_WORK.md`：ROPD（2605.07396）已占「用评分点代替 logit」；DARS（2610.01207，预印本）已占逐步依赖奖励。DARS 附录 E 有逐轮读前缀的设置。
- 奖励与接管记账：`python -m unittest research.test_mechanism`，以及 `python -m research.run_m0` 写出的 `results/m0-mechanism-001/`。
- 损失开关和回报切断：`python -m unittest research.test_loss_spec`。

## 这些检查说明什么

符号和公式按设计实现了。脚本动作匹配的低分不能外推成真实误罚。损失检查里的 0.2、1.0、0.1 只是用来断言规则的输入，不是调过的超参。

## 未验证

- 六类「不一样」里，哪些真的被 token OPD 或 ActionOPD 压分。
- 老师写出的评分点是否够准，写错时会误伤多少。
- 按步打分是否优于整卷打一次分，以及是否能替代 DARS 的手写图。
- KL 开关是否优于全程 KL。
- 按评分点纠正或早停是否优于按分歧干预。

## 证据

没有成功率，没有损失曲线。
