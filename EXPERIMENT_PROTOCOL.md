# 当前实验协议

原始设计快照：[时间戳协议](protocols/EXPERIMENT_PROTOCOL_20261017T040502Z.md)，[时间戳计划](protocols/CH2_GAIN_GATE_PLAN_20261017T040502Z.md)。窗口末完成成本已由[单次终点剩余工作修订](protocols/AMENDMENT_SINGLE_TERMINAL_PENALTY_20261017T045635Z.md)升版；[尝试后冷却强基线修订](protocols/AMENDMENT_ATTEMPT_COOLDOWN_BASELINE_20261017.md)保留原 `ExistingET` 并新增 `ExistingETAttemptCooldown`，两者均纳入后续比较。

在正式采集前增补[attempt-cooldown ET 基线及主比较族](protocols/AMENDMENT_ATTEMPT_COOLDOWN_BASELINE_20261017.md)：保留原 ExistingET，并新增每次普通重构尝试后冷却4分钟的强基线。该修订尚待主协调者只读审核。

状态：协议修订待主协调者只读审核。尚未开始正式配对数据采集、训练或测试；修订版的实现和哈希须在采集前冻结。本文件为当前协议入口；历史版本保留供审计。
