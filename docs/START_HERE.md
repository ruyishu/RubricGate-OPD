# 现在能跑的检查

```text
python -m unittest research.test_mechanism research.test_loss_spec
python -m research.run_m0
```

`results/m0-mechanism-001/notes.md` 只解释奖励符号和脚本动作匹配。`test_loss_spec` 只解释老师何时出现、KL 系数、候选奖励和回报切断。都通过，也不说明真实老师会误罚，更没有训练收益。

## 下一件该做的事

按 [EXPERIMENTS.md](EXPERIMENTS.md) 里写死的 P0 协议做：在 \(\tau^2\)-bench retail 上收学生轨迹，在学生经过的状态上让老师贪心一次、采样四次，把不一致的步标成七类，再算 token 侧和动作侧的压分比例。

门槛不过就停，不写训练器。过了再做 P1 评分点质量。纠正、早停和 `train/` 放在那之后。
