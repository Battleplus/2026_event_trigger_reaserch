# V33 结论分析（COMPLETE_168_GATE_NOT_MET）

快照：2026-10-21T05:34:55.766707+00:00；168/168 场景、168 个已提交分片；采集状态 `complete_168`。marker 清单 SHA-256：`9f6963b87c796714bc7c3001f88310c340f60d88e5b1bcf2c49f02986041f9b3`。非 complete_168 时，结论仅为诊断。

## 前页结论

- 跨区有效否：当前证据不足以作正式结论。发现跨区候选 action 9352 个、选派 action 1145 个；候选/选派唯一状态数按 split 见下表。这些不是跨区相对本地策略的因果效应。
- 门控有效否：3 个正增益、0 个零增益、414 个负增益唯一普通状态。冻结门槛（总正≥30、总非正≥30，且 train 与 validation 各自两类均覆盖多个场景）未达。
- 推荐方法：当前 168 场景均已提交且无失败分片，可进入门槛评估；本快照不据此选定模型或部署门控。
- WM 值不值得：当前数据没有 WM 与轻量方法/无模型的闭环对照，不能回答；不应据此训练或推荐 WM。
- 适用范围：合成 100×100、3×3 九区、36 平台、search-rescue 场景与已覆盖扰动；不能外推真实灾害或未覆盖任务。

## 状态与覆盖

| split | 唯一状态 | forced | empty | unknown | 可评估 | 正 | 非正 | 正类场景 | 非正场景 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1475 | 1251 | 11 | 0 | 213 | 1 | 212 | 1 | 28 |
| validation | 1405 | 1181 | 18 | 2 | 204 | 2 | 202 | 2 | 28 |

类按每个唯一 `state_id` 的 oracle 重算均值划分；同一状态的匹配后缀只贡献一个状态。forced、empty obligation 和 unknown 均排除。30/30 为 train+validation 合计门槛，并要求两个 split 中正类与非正类分别覆盖多个场景；没有单独的 train≥30 要求。

区间按独立单位 `scenario_instance` 计算：先在场景内平均普通有效 state gain，再 bootstrap 场景 10,000 次。区间描述本批采集标签，不是策略或跨区处理效应 CI。

| split | 场景数 | 场景均值的等权平均 gain | 场景 bootstrap 95% CI |
|---|---:|---:|---:|
| train | 28 | -0.063330 | [-0.068499, -0.058707] |
| validation | 28 | -0.066026 | [-0.079365, -0.053297] |

## Live Safety Receipt

从 header `trajectory_trace` 独立计数 `safety_disposition.resolved == false`：已知场景 168/168，实际未解决处置数合计 343；摘要计数合计 343。缺失 trajectory trace 的场景 0 个，不按零处理；逐场景计数、原因和摘要匹配结果在 `analysis.json.scenarios.*.live_safety_receipt_audit`。摘要不一致场景：`[]`。
强制状态实际计数 2432；header 摘要已知计数合计 2432，缺摘要 0 个场景；不一致场景：`[]`。这是运行时安全处置回执核验，不证明全部硬约束已满足或方法安全；branch safety 仍由 gain oracle 的配对标签单独复算，最终安全结论限于协议和实际核验范围。


## 跨区动作与后果

候选回执排序复核 2537 条、排序不匹配 0 条。跨区 action 以每个候选/选中动作计数，唯一状态按 `state_id` 去重：候选 9352 actions，选派 1145 actions。按 queue operation 的原始语义分别计数，insert=7524、transfer=533、split=538、cooperate=529；split/cooperate 不计作 donor transfer。含跨区 direct dispatch 的候选 action=228、选派 action=213，即使没有 queue operation 也纳入 action 总数。
| action / state | 候选 action 数 | 候选唯一状态数 | 选派 action 数 | 选派唯一状态数 |
|---|---:|---:|---:|---:|
| any | 9352 | 1732 | 1145 | 1145 |
| insert | 7524 | 1456 | 932 | 932 |
| transfer | 533 | 176 | 0 | 0 |
| split | 538 | 174 | 0 | 0 |
| cooperate | 529 | 171 | 0 | 0 |
| direct_dispatch | 228 | 228 | 213 | 213 |

上述 action/state 统计另在 JSON 按 ordinary/forced 拆分；forced 动作不带有效 gain 标签。任务/平台 ID 中 `00`..`08` 是 row-major 区域 ordinal，映射契约区域 `r00,r01,r02,r10,r11,r12,r20,r21,r22`；完整映射见 JSON `region_id_mapping`。区域后果与成本分解只汇总普通、有效标签的匹配 suffix；每个 suffix 的明细见 `state_detail.csv`。区域进度变化是 endpoint 结果，不能单独识别跨区动作的因果效果。

| 区域 | paired suffix 数 | keep 完成任务数均值 | replan 完成任务数均值 | 完成数差均值 | keep 剩余比例和均值 | replan 剩余比例和均值 |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 834 | 4.7182 | 4.7182 | 0.0000 | 1.2769 | 1.2772 |
| 01 | 834 | 4.5827 | 4.5839 | 0.0012 | 1.4152 | 1.4128 |
| 02 | 834 | 4.4892 | 4.4928 | 0.0036 | 1.5095 | 1.5056 |
| 03 | 834 | 4.7386 | 4.7398 | 0.0012 | 1.2596 | 1.2584 |
| 04 | 834 | 4.5671 | 4.5695 | 0.0024 | 1.4303 | 1.4279 |
| 05 | 834 | 4.3597 | 4.3669 | 0.0072 | 1.6387 | 1.6315 |
| 06 | 834 | 4.4017 | 4.4089 | 0.0072 | 1.5935 | 1.5863 |
| 07 | 834 | 4.3621 | 4.3657 | 0.0036 | 1.6347 | 1.6311 |
| 08 | 834 | 4.2542 | 4.2590 | 0.0048 | 1.7407 | 1.7359 |

### 普通有效且选派跨区状态

唯一状态 12 个，覆盖 12 个场景；场景均值的等权平均 gain 为 -0.18467739695820773，描述性 95% CI=[-0.2344692901176115, -0.1406057550421071]（descriptive_estimable）。路线按平台 home 区与任务区区分：home 区跨区 insert、显式 donor transfer、其他跨区操作及 direct dispatch 分开计数；insert 路由本身不证明平台空闲。这里只是被选中跨区状态的配对结果，不是跨区动作相对本地动作的因果效应。

| 路线类别 | 被选路由数 |
|---|---:|
| direct_cross_region_dispatch | 2 |
| home_region_insert | 11 |

| 区域（all-region endpoint） | paired suffix 数 | keep 完成数 | replan 完成数 | 完成数差 | keep 剩余比例和 | replan 剩余比例和 |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 24 | 4.3333 | 4.2500 | -0.0833 | 1.6667 | 1.7500 |
| 01 | 24 | 3.9583 | 4.0417 | 0.0833 | 2.0208 | 1.8958 |
| 02 | 24 | 3.8750 | 3.8750 | 0.0000 | 2.1250 | 2.1146 |
| 03 | 24 | 4.2917 | 4.2500 | -0.0417 | 1.7083 | 1.7500 |
| 04 | 24 | 3.8333 | 3.9583 | 0.1250 | 2.1667 | 2.0417 |
| 05 | 24 | 3.8333 | 3.8750 | 0.0417 | 2.1667 | 2.1250 |
| 06 | 24 | 3.7917 | 4.0000 | 0.2083 | 2.2083 | 2.0000 |
| 07 | 24 | 3.9167 | 4.0000 | 0.0833 | 2.0625 | 1.9792 |
| 08 | 24 | 3.8333 | 3.8750 | 0.0417 | 2.1667 | 2.1250 |

角色分组按每个唯一 state×suffix 中的路线区域去重；同一区域可同时属于多个角色，重叠 state 数保存在 JSON。`other_region` 是当前状态路线未涉及的区域。

| 路由角色 | 区域 | paired suffix 数 | 完成数差均值 | 剩余比例差均值（keep−replan） |
|---|---|---:|---:|---:|
| other_region | 00 | 6 | 0.0000 | 0.0000 |
| other_region | 01 | 18 | 0.1111 | 0.1111 |
| other_region | 02 | 24 | 0.0000 | 0.0104 |
| other_region | 03 | 14 | -0.0714 | -0.0714 |
| other_region | 04 | 20 | 0.0500 | 0.0500 |
| other_region | 05 | 22 | 0.0455 | 0.0455 |
| other_region | 06 | 22 | 0.2273 | 0.2273 |
| other_region | 07 | 20 | 0.0000 | 0.0000 |
| other_region | 08 | 18 | 0.0556 | 0.0556 |
| platform_home_region | 00 | 18 | -0.1111 | -0.1111 |
| platform_home_region | 01 | 2 | 0.0000 | 0.0000 |
| platform_home_region | 03 | 4 | 0.0000 | 0.0000 |
| platform_home_region | 07 | 2 | 0.0000 | 0.0000 |
| recipient_task_region | 01 | 4 | 0.0000 | 0.2500 |
| recipient_task_region | 03 | 6 | 0.0000 | 0.0000 |
| recipient_task_region | 04 | 4 | 0.5000 | 0.5000 |
| recipient_task_region | 05 | 2 | 0.0000 | 0.0000 |
| recipient_task_region | 06 | 2 | 0.0000 | 0.0000 |
| recipient_task_region | 07 | 2 | 1.0000 | 1.0000 |
| recipient_task_region | 08 | 6 | 0.0000 | 0.0000 |

| selected cross-region 全局成本分项均值 | keep | replan | replan − keep |
|---|---:|---:|---:|
| C_replan | 0.515788 | 0.615345 | 0.099557 |
| C_safety | 0.053607 | 0.055169 | 0.001562 |
| J_exec | 3.200232 | 3.283790 | 0.083558 |
| J_total | 3.769627 | 3.954304 | 0.184677 |
| execution_components.completion_cost | 0.528809 | 0.515062 | -0.013746 |
| execution_components.delay_cost | 0.000000 | 0.000000 | 0.000000 |
| execution_components.delay_minutes | 0.000000 | 0.000000 | 0.000000 |
| execution_components.energy_cost | 2.635629 | 2.732933 | 0.097304 |
| execution_components.energy_units | 263.562939 | 273.293332 | 9.730393 |
| execution_components.risk_cost | 0.035794 | 0.035794 | 0.000000 |
| execution_components.risk_units | 0.143176 | 0.143176 | 0.000000 |
| execution_components.weighted_completion | 0.469740 | 0.482362 | 0.012622 |
| execution_components.weighted_remaining_fraction | 0.528809 | 0.515062 | -0.013746 |
| replan_components.call_count | 4.125000 | 5.125000 | 1.000000 |
| replan_components.communication_cost | 0.127500 | 0.147500 | 0.020000 |
| replan_components.communication_messages | 12.750000 | 14.750000 | 2.000000 |
| replan_components.continuation_planner_cpu_seconds | 14.412109 | 14.345703 | -0.066406 |
| replan_components.planner_cpu_cost | 0.144121 | 0.157428 | 0.013307 |
| replan_components.planner_cpu_seconds | 0.000000 | 1.397135 | 1.397135 |
| replan_components.reconstruction_cost | 0.206250 | 0.256250 | 0.050000 |
| replan_components.reconstruction_size | 3.791667 | 5.000000 | 1.208333 |
| replan_components.reconstruction_size_cost | 0.037917 | 0.050000 | 0.012083 |
| replan_components.switch_cost | 0.000000 | 0.004167 | 0.004167 |
| replan_components.switch_count | 0.000000 | 0.083333 | 0.083333 |
| safety_components.continuation_policy_cpu_cost | 0.053607 | 0.055169 | 0.001562 |
| safety_components.continuation_policy_cpu_seconds | 5.360677 | 5.516927 | 0.156250 |

| selected NOOP 全局成本分项均值 | keep | replan | replan − keep |
|---|---:|---:|---:|
| C_replan | 0.193026 | 0.250786 | 0.057760 |
| C_safety | 0.072547 | 0.072046 | -0.000501 |
| J_exec | 4.651191 | 4.651191 | 0.000000 |
| J_total | 4.916764 | 4.974023 | 0.057259 |
| execution_components.completion_cost | 0.638375 | 0.638375 | 0.000000 |
| execution_components.delay_cost | 2.819693 | 2.819693 | 0.000000 |
| execution_components.delay_minutes | 14.098465 | 14.098465 | 0.000000 |
| execution_components.energy_cost | 1.161069 | 1.161069 | 0.000000 |
| execution_components.energy_units | 116.106899 | 116.106899 | 0.000000 |
| execution_components.risk_cost | 0.032053 | 0.032053 | 0.000000 |
| execution_components.risk_units | 0.128213 | 0.128213 | 0.000000 |
| execution_components.weighted_completion | 0.360679 | 0.360679 | 0.000000 |
| execution_components.weighted_remaining_fraction | 0.638375 | 0.638375 | 0.000000 |
| replan_components.call_count | 1.585678 | 2.585678 | 1.000000 |
| replan_components.communication_cost | 0.042353 | 0.042353 | 0.000000 |
| replan_components.communication_messages | 4.235294 | 4.235294 | 0.000000 |
| replan_components.continuation_planner_cpu_seconds | 5.798773 | 5.825088 | 0.026315 |
| replan_components.planner_cpu_cost | 0.057988 | 0.065747 | 0.007760 |
| replan_components.planner_cpu_seconds | 0.000000 | 0.749640 | 0.749640 |
| replan_components.reconstruction_cost | 0.079284 | 0.129284 | 0.050000 |
| replan_components.reconstruction_size | 1.263427 | 1.263427 | 0.000000 |
| replan_components.reconstruction_size_cost | 0.012634 | 0.012634 | 0.000000 |
| replan_components.switch_cost | 0.000767 | 0.000767 | 0.000000 |
| replan_components.switch_count | 0.015345 | 0.015345 | 0.000000 |
| safety_components.continuation_policy_cpu_cost | 0.072547 | 0.072046 | -0.000501 |
| safety_components.continuation_policy_cpu_seconds | 7.254715 | 7.204644 | -0.050072 |

| selected non-NOOP 全局成本分项均值 | keep | replan | replan − keep |
|---|---:|---:|---:|
| C_replan | 0.549654 | 0.669156 | 0.119501 |
| C_safety | 0.057037 | 0.057491 | 0.000454 |
| J_exec | 3.569641 | 3.624501 | 0.054860 |
| J_total | 4.176333 | 4.351148 | 0.174815 |
| execution_components.completion_cost | 0.431838 | 0.412388 | -0.019450 |
| execution_components.delay_cost | 0.476923 | 0.461538 | -0.015385 |
| execution_components.delay_minutes | 2.384615 | 2.307692 | -0.076923 |
| execution_components.energy_cost | 2.605956 | 2.695651 | 0.089696 |
| execution_components.energy_units | 260.595588 | 269.565148 | 8.969560 |
| execution_components.risk_cost | 0.054924 | 0.054924 | 0.000000 |
| execution_components.risk_units | 0.219695 | 0.219695 | 0.000000 |
| execution_components.weighted_completion | 0.566932 | 0.585956 | 0.019024 |
| execution_components.weighted_remaining_fraction | 0.431838 | 0.412388 | -0.019450 |
| replan_components.call_count | 4.288462 | 5.288462 | 1.000000 |
| replan_components.communication_cost | 0.124615 | 0.143846 | 0.019231 |
| replan_components.communication_messages | 12.461538 | 14.384615 | 1.923077 |
| replan_components.continuation_planner_cpu_seconds | 17.273137 | 17.538161 | 0.265024 |
| replan_components.planner_cpu_cost | 0.172731 | 0.205694 | 0.032963 |
| replan_components.planner_cpu_seconds | 0.000000 | 3.031250 | 3.031250 |
| replan_components.reconstruction_cost | 0.214423 | 0.264423 | 0.050000 |
| replan_components.reconstruction_size | 3.788462 | 5.134615 | 1.346154 |
| replan_components.reconstruction_size_cost | 0.037885 | 0.051346 | 0.013462 |
| replan_components.switch_cost | 0.000000 | 0.003846 | 0.003846 |
| replan_components.switch_count | 0.000000 | 0.076923 | 0.076923 |
| safety_components.continuation_policy_cpu_cost | 0.057037 | 0.057491 | 0.000454 |
| safety_components.continuation_policy_cpu_seconds | 5.703726 | 5.749099 | 0.045373 |

| 全局成本分项（均值） | keep | replan | replan − keep |
|---|---:|---:|---:|
| J_total | 4.870598 | 4.935186 | 0.064588 |
| J_exec | 4.583756 | 4.587176 | 0.003421 |
| C_replan | 0.215262 | 0.276871 | 0.061609 |
| C_safety | 0.071580 | 0.071139 | -0.000441 |

execution / replan / safety 子项的全量数值与上述成本汇总一并保存在 `analysis.json` 的 `outcome_decomposition.global_cost_component_means`。这些值比较 paired keep 与 replan 的整体结果，不能单独解释为跨区的效果。

## 可识别性与限制

普通有效状态共 417 个，选派 NOOP 391 个、非 NOOP 26 个；NOOP 选择原因（仅 selected NOOP）：`{"NOOP_only_legal_candidate": 390, "NOOP_has_lower_public_cost_than_all_alternatives": 1}`；selected non-NOOP 原因：`{"NONNOOP_selected_by_public_cost": 26}`。候选与不可行数明细在 `state_detail.csv`。forced 的跨区选派没有可用 gain 标签，不能用来估计动作收益。

当前可从轨迹识别：候选是否合法且跨区、所选动作与公开排序数值、匹配 suffix 下 keep/replan 的整体成本和区域任务后果、全局 gain 正零负及 mask 原因。live safety receipt 只核对运行时处置回执及摘要，不代表所有硬约束或方法安全证明。不能从该采集识别：将同一时点同一局面随机分配跨区与本地动作的纯跨区因果效应；跨区具体机制相对能力/距离/供需排序的独立贡献；训练后模型门控的闭环收益；相对 WM 的性能/成本；真实救灾的外部有效性。

资源：选派 planner CPU 合计 6330.875 秒；已提交分片 runtime receipt 报告进程 CPU 82174.812 秒、墙钟 113233.955 秒，receipt 时 RSS 最大值 1279627264 bytes。峰值内存为 UNKNOWN，不能由 receipt 时 RSS 替代。
成本复核：独立复算覆盖 834 个普通有效 matched suffix；其中 keep/replan endpoint 共 67536 个完成任务记录，完成但 progress<1 的记录为 0。oracle 的 completed 任务分支直接跳过 residual 累加，因此完成任务残余惩罚为零；未完成任务按加权 clipped `(1-progress)` 计入。

## 下一步所需证据

168 个 train/validation 场景及 commit marker 均已校验；仍需检查 30/30 标签门槛及 train/validation 覆盖结果。
若要回答跨区纯因果问题，需预先定义并随机/成对构造具有共同可行性的 local 与 cross-region 动作对，并固定外生 suffix、安全继续策略和任务宇宙。若要回答门控与 WM，须在训练门槛通过后用闭环策略、相同 replanner/safety、独立测试场景比较，报告场景单位配对 CI。

原 45 条 smoke 记录属于工程 smoke，不能拼入正式类门槛或方法效应。
