# 状态

最后更新：2026-10-28

## 当前

设计已按 GraphOPD、DARS、GEAR、CD-OPD 收紧。仓库里有文档和一个 CPU 机制夹具，没有训练器，没有模型分数。

## 已核对

- 方法边界写在 `docs/METHOD.md`：第一版只打分；清单外动作不是错误；环境结果才算完成；接管是第二阶段，代打进展不进学生回报。
- 相关工作补了 GraphOPD（2610.08959）、DARS（2610.01207）、GEAR（2606.03361）、CD-OPD（2610.25359）。DARS 附录 E 有逐轮读前缀的设置。
- 机制夹具：`python -m unittest research.test_mechanism`，12 项通过；`python -m research.run_m0` 写出 `results/m0-mechanism-001/`。

## 夹具结论

规则自检通过，包括错误 DAG 和接管记账。v1 奖励和简化势能差的符号一致，所以依赖打分本身没有显示出相对 DARS 的新差异。动作匹配的低分来自脚本分布，不能外推成真实教师的误罚率。

## 未验证

- 真实 token-OPD 或 ActionOPD 是否显著误罚合法换序。
- 减少错误干预能否提高成功率。
- 自动构造的 DAG 是否够用。
- 接管后的截断回报能否训稳。

## 证据

只有 `results/m0-mechanism-001` 的规则表。没有成功率。
