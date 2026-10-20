# V33 结论分析（PARTIAL）

快照：2026-10-20T12:22:35.454784+00:00；66/168 场景、66 个已提交分片。marker 清单 SHA-256：`d91757825888ad6334cf9b7bdbab56f78955066c95da06787f85e2ed1b5329db`。未满 168 场景时，结论仅为 partial 诊断。

## 前页结论

- 跨区有效否：当前证据不足以作正式结论。发现跨区候选 action 3794 个、选派 action 475 个；候选/选派唯一状态数按 split 见下表。这些不是跨区相对本地策略的因果效应。
- 门控有效否：2 个正增益、0 个零增益、162 个负增益唯一普通状态。冻结门槛（总正≥30、总非正≥30，且 train 与 validation 各自两类均覆盖多个场景）未达。
- 推荐方法：仅能推荐继续完成预注册采集并按同一脚本重算；本 partial 不能选定模型或部署门控。
- WM 值不值得：当前数据没有 WM 与轻量方法/无模型的闭环对照，不能回答；不应据此训练或推荐 WM。
- 适用范围：合成 100×100、3×3 九区、36 平台、search-rescue 场景与已覆盖扰动；不能外推真实灾害或未覆盖任务。

## 状态与覆盖

| split | 唯一状态 | forced | empty | unknown | 可评估 | 正 | 非正 | 正类场景 | 非正场景 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 546 | 450 | 0 | 0 | 96 | 1 | 95 | 1 | 12 |
| validation | 461 | 389 | 4 | 0 | 68 | 1 | 67 | 1 | 9 |

类按每个唯一 `state_id` 的 oracle 重算均值划分；同一状态的匹配后缀只贡献一个状态。forced、empty obligation 和 unknown 均排除。30/30 为 train+validation 合计门槛，并要求两个 split 中正类与非正类分别覆盖多个场景；没有单独的 train≥30 要求。

区间按独立单位 `scenario_instance` 计算：先在场景内平均普通有效 state gain，再 bootstrap 场景 10,000 次。区间描述本批采集标签，不是策略或跨区处理效应 CI。

| split | 场景数 | 场景均值的等权平均 gain | 场景 bootstrap 95% CI |
|---|---:|---:|---:|
| train | 12 | -0.058712 | [-0.063167, -0.054716] |
| validation | 9 | -0.065645 | [-0.101578, -0.035281] |

## 跨区动作与后果

候选回执排序复核 1007 条、排序不匹配 0 条。跨区 action 以每个候选/选中动作计数，唯一状态按 `state_id` 去重：候选 3794 actions，选派 475 actions。按 queue operation 的原始语义分别计数，insert=3051、transfer=216、split=220、cooperate=216；split/cooperate 不计作 donor transfer。含跨区 direct dispatch 的候选 action=91、选派 action=82，即使没有 queue operation 也纳入 action 总数。
| action / state | 候选 action 数 | 候选唯一状态数 | 选派 action 数 | 选派唯一状态数 |
|---|---:|---:|---:|---:|
| any | 3794 | 700 | 475 | 475 |
| insert | 3051 | 595 | 393 | 393 |
| transfer | 216 | 71 | 0 | 0 |
| split | 220 | 71 | 0 | 0 |
| cooperate | 216 | 71 | 0 | 0 |
| direct_dispatch | 91 | 91 | 82 | 82 |

上述 action/state 统计另在 JSON 按 ordinary/forced 拆分；forced 动作不带有效 gain 标签。区域后果与成本分解只汇总普通、有效标签的匹配 suffix；每个 suffix 的明细见 `state_detail.csv`。区域进度变化是 endpoint 结果，不能单独识别跨区动作的因果效果。

| 区域 | paired suffix 数 | keep 完成任务数均值 | replan 完成任务数均值 | 完成数差均值 | keep 剩余比例和均值 | replan 剩余比例和均值 |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 328 | 4.5549 | 4.5549 | 0.0000 | 1.4360 | 1.4360 |
| 01 | 328 | 4.5091 | 4.5061 | -0.0030 | 1.4893 | 1.4924 |
| 02 | 328 | 4.3750 | 4.3780 | 0.0030 | 1.6217 | 1.6186 |
| 03 | 328 | 4.6189 | 4.6250 | 0.0061 | 1.3796 | 1.3735 |
| 04 | 328 | 4.3293 | 4.3262 | -0.0030 | 1.6651 | 1.6682 |
| 05 | 328 | 4.2348 | 4.2439 | 0.0091 | 1.7630 | 1.7538 |
| 06 | 328 | 4.1585 | 4.1616 | 0.0030 | 1.8384 | 1.8354 |
| 07 | 328 | 4.1372 | 4.1341 | -0.0030 | 1.8582 | 1.8613 |
| 08 | 328 | 4.0274 | 4.0366 | 0.0091 | 1.9665 | 1.9573 |

| 全局成本分项（均值） | keep | replan | replan − keep |
|---|---:|---:|---:|
| J_total | 5.204086 | 5.265897 | 0.061812 |
| J_exec | 4.923069 | 4.926039 | 0.002970 |
| C_replan | 0.214425 | 0.273822 | 0.059397 |
| C_safety | 0.066592 | 0.066036 | -0.000555 |

execution / replan / safety 子项的全量数值与上述成本汇总一并保存在 `analysis.json` 的 `global_cost_component_means`。这些值比较 paired keep 与 replan 的整体结果，不能单独解释为跨区的效果。

## 可识别性与限制

普通有效状态共 164 个，选派 NOOP 156 个、非 NOOP 8 个；NOOP 选择原因（仅 selected NOOP）：`{"NOOP_only_legal_candidate": 155, "NOOP_has_lower_public_cost_than_all_alternatives": 1}`；selected non-NOOP 原因：`{"NONNOOP_selected_by_public_cost": 8}`。候选与不可行数明细在 `state_detail.csv`。forced 的跨区选派没有可用 gain 标签，不能用来估计动作收益。

当前可从轨迹识别：候选是否合法且跨区、所选动作与公开排序数值、匹配 suffix 下 keep/replan 的整体成本和区域任务后果、全局 gain 正零负及 mask 原因。不能从该采集识别：将同一时点同一局面随机分配跨区与本地动作的纯跨区因果效应；跨区具体机制相对能力/距离/供需排序的独立贡献；训练后模型门控的闭环收益；相对 WM 的性能/成本；真实救灾的外部有效性。

资源：选派 planner CPU 合计 2486.875 秒；已提交分片 runtime receipt 报告进程 CPU 29163.594 秒、墙钟 32218.011 秒，receipt 时 RSS 最大值 1252184064 bytes。峰值内存为 UNKNOWN，不能由 receipt 时 RSS 替代。
成本复核：独立复算覆盖 328 个普通有效 matched suffix；其中 keep/replan endpoint 共 25555 个完成任务记录，完成但 progress<1 的记录为 0。oracle 的 completed 任务分支直接跳过 residual 累加，因此完成任务残余惩罚为零；未完成任务按加权 clipped `(1-progress)` 计入。

## 下一步所需证据

完成并校验全部 168 个 train/validation 场景及 commit marker；为完整采集保存绑定到 run ID 的墙钟、进程 CPU、峰值内存 receipt。若要回答跨区纯因果问题，需预先定义并随机/成对构造具有共同可行性的 local 与 cross-region 动作对，并固定外生 suffix、安全继续策略和任务宇宙。若要回答门控与 WM，须在训练门槛通过后用闭环策略、相同 replanner/safety、独立测试场景比较，报告场景单位配对 CI。

原 45 条 smoke 记录属于工程 smoke，不能拼入正式类门槛或方法效应。
