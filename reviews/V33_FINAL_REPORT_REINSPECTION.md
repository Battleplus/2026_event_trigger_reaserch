# V33 final report reinspection

**Verdict: WARN — limited report reinspection.** Reviewer: GPT-6 Astra ultra, fresh same-family reviewer, provisional. Evaluation: simulation_only.

本轮只复查归档目录的 FINAL_CONCLUSION.md、conclusion.md、FINAL_DIAGNOSTICS.md/json，以及项目内 V33_FINAL_COORDINATOR_DECISION.md 的修订表述。未重新读取原始分片、运行模拟、测试、oracle 或重新计算实验统计。数值证据依据上一轮已记录的有限审核；本轮确认报告与其边界相符，不重新认证实验正确性。

## 六项 E 问题

| 项目 | 复查状态 | 修订后的证据与边界 |
|---|---|---|
| E1 来源覆盖 | ADDRESSED | FINAL_CONCLUSION.md:16、FINAL_DIAGNOSTICS.md:3 明确 417 个普通有效状态均来自 56 个 Periodic-4min 场景，其他两来源各 56 场景没有普通有效标签。 |
| E2 343 条 safety 回执含义 | ADDRESSED | FINAL_CONCLUSION.md:7、:16 与 FINAL_DIAGNOSTICS.md:5 明确全部为 pre-public hash mismatch 导致无法验证，不等同 343 次物理违规，不证明安全达标；工程状态保留 REPAIR_REQUIRED。 |
| E3 正状态与 NOOP 计时 | PARTIALLY_ADDRESSED | FINAL_CONCLUSION.md:18 明确三个正状态为本地 insert；FINAL_DIAGNOSTICS.json:38 明确 positive_states_selected_noop=0，并披露第三个正状态 suffix 符号不稳。仍缺少对另三个 NOOP 正单 suffix 的显式说明，见下文；没有错误增加正状态数。 |
| E4 跨区子集与 donor 推断 | ADDRESSED | FINAL_CONCLUSION.md:5 已将“派出区域等合计导致变差”改为观测全局配对成本增加，明确区域 endpoint 未加任务权重且不证明 donor 放弃/牺牲任务；:18 限定 11 个 insert、1 个 direct-dispatch action。FINAL_DIAGNOSTICS.md:9 保留描述性、非因果边界。 |
| E5 30/30 口径与陈旧下一步 | ADDRESSED | conclusion.md:20 保留 train+validation 合计且每 split 各类覆盖多个场景的规则；:213 已改为门槛未达，不再说“仍需检查”。唯一 state 才计一次，suffix 不扩大样本。 |
| E6 冻结绑定范围 | ADDRESSED | FINAL_DIAGNOSTICS.json:71–73 及 V33_FINAL_COORDINATOR_DECISION.md:26 区分 package/config/baseline marker 绑定和后处理单独哈希；本复查另记录所读五个文件的准确 SHA-256。 |

剩余一项披露建议：在 FINAL_DIAGNOSTICS.md/json 补充“另有 3 个 NOOP 单 suffix 的 gain 为正，但各自状态均值仍负；这些 suffix 的 J_exec 相同，正差来自 CPU 计时差抵消一次 0.05 调用费用。391 个 NOOP 状态中没有正状态标签。”这来自上一轮已记录的核查，本轮没有重新运算。该遗漏不改变 3 正、414 负或 NOT_MET，也不是当前报告夸大正例数的证据。

## 其他边界

- V33_FINAL_COORDINATOR_DECISION.md:24 准确限定 exactly-once 为定向 split/cooperate 生命周期夹具及发现后的 NOOP pair，不覆盖配对分支内新发生的整个发现链。
- FINAL_CONCLUSION.md:22 已将后续场景设计限定为采集前冻结公开时序与决策机会，明确不得按已观测正收益筛选场景。
- Existing-ET 仅为比较基线；修订报告没有以本实验为依据授予最优方法、部署或安全背书。
- 保留原审核的 WARN、same-family、provisional；报告措辞修正不修复冻结采集的前态哈希问题，不证明方法有效性、安全达标或真实世界外部有效性。
- 本轮读取路径为归档目录的四份报告和项目内协调器决策。项目同步由协调器报告；本复查不额外认证目录同步。输入快照与准确哈希保存在本次 trace 和 JSON 中。
