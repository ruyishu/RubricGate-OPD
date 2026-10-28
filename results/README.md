# 结果记录

每个实验一子目录：`results/<experiment_id>/`。

- `metrics.csv`：实测或按设计算出的表，不填猜测。
- `notes.md`：这张表能说明什么、不能说明什么。
- 有配置的实验再放 `config.json`。

`m0-mechanism-001` 是脚本规则检查。它的动作匹配数字是给定分布下的期望，不是模型评测。KL 开关和回报切断在 `research/test_loss_spec.py`，不另建结果目录。
