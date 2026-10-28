# 现在能跑的检查

```text
python -m unittest research.test_mechanism research.test_loss_spec
python -m research.run_m0
```

`results/m0-mechanism-001/notes.md` 只解释奖励符号和脚本动作匹配。`test_loss_spec` 只解释 KL 开关和回报切断。两个都通过，也不说明真实老师会误罚，更没有训练收益。

## 下一件该做的事

用真实学生轨迹和真实老师，把师生不一致的步标成六类：顺序不同、换工具、粒度不同、多了无害步骤、真错、关键步走错。先量「不同但没错」的步被 token KL 和动作匹配压分的比例。

比例很小就停，不写训练器。比例很大，再做评分点质量和只打分、不干预的训练。纠正、早停和 `train/` 放在那之后。
