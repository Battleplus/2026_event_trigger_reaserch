# V3.3 Final Diagnostics

采集快照包含 168/168 个已提交场景，marker 清单 SHA-256 为 `9f6963b87c796714bc7c3001f88310c340f60d88e5b1bcf2c49f02986041f9b3`。普通有效的 417 个状态全部来自 56 个 `Periodic-4min` 场景；`Existing-ET` 与 `Random-Allow-Suppress` 各 56 个场景均没有普通有效标签。来源限制意味着本结果不能代表后两种轨迹来源。

168 个场景都有 trajectory trace，未解决处置回执共 343 条，header 摘要数完全匹配。逐事件原因均为 `safety_receipt_pre_public_hash_mismatch`，即处置前公开快照哈希验证不匹配。该原因使状态验证无法成立，但不等价于 343 次已证实的安全违规。另有 2 个 validation gain 状态因同一原因保留为 unknown 并排除。安全达标仍未证实，需修复并审查前态哈希链。

收益门槛结果为 3 正、0 零、414 负，正例占普通有效状态 0.72%，未达到 30/30 门槛。三个正例都是本地 insert，且均非 NOOP-only；其中第三个状态的 suffix gain 为 `-0.20151` 和 `+1.31886`，符号不稳定。另有 3 个 NOOP 单 suffix gain 为正，但对应状态均值仍负，且两臂 `J_exec` 相同；正差来自 CPU 计时差抵消一次 0.05 调用费用，不计为正状态。391 个 NOOP 状态中没有正状态标签。390 个状态只有 NOOP 是合法候选。方法状态为 `INCONCLUSIVE`，数据门槛为 `NOT_MET`，不支持训练或部署收益门控。正例稀少且执行与规划计时存在波动，不能仅凭当前样本确立稳定方法效果。

普通有效集合中选派跨区的 12 个状态包含 11 个 insert 与 1 个 direct-dispatch action。它们的 paired `J_total(replan−keep)` 为 `+0.184677`，只描述被选中子集的观测全局成本增加，不是随机跨区与本地动作对照；执行、重规划和安全成本分项应分别解读。平台 home 区的区域后果是未加任务权重的 endpoint 描述，不证明 donor 放弃/牺牲任务。

首轮采集有 4 项失败记录（communication degradation 退出码 `1`；environment/obstacle change 退出码 `4294967295`）；已知原始 registry 106 记录包含 MemoryError，另有重试进度回执记录 3 项失败。重试日志有部分被覆盖，完整历史归因应结合单独保存的原始失败归档，不能只从当前 `COLLECTION_PROGRESS` 还原。最终 168 个场景均已提交。协调器重复启动/终止影响失败归因，因此不能把历史失败都称为模拟器自然失败。中断及重复进程的完整累计 CPU 消耗，以及真实峰值内存均为 `UNKNOWN`；已记录分片 CPU 仍按原 receipt 报告。最终审阅状态为 `WARN`。GPU、世界模型和 PPO 均为 `NOT_RUN`。Existing-ET 只作为比较基线，不是最佳方法或安全背书；本快照仅作实验参考基线，安全达标与方法有效性尚未证实。

继续研究前须修复并审查安全处置前态哈希链，并在采集前冻结、平衡各轨迹来源的公开任务时序与决策机会覆盖；不得按观测到的正收益筛选场景。机器可读明细见 `FINAL_DIAGNOSTICS.json`。本诊断不修改原始 shard、冻结配置、成本或采集结果。
