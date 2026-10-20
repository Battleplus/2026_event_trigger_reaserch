# V33 结论独立审核草稿

审核者：GPT-6-Astra；fresh reviewer；same-family；provisional；simulation_only。本审核不修改生产源码、原始分片、执行者报告或 focused 原证据。正式采集尚在进行，因此本文件不授予整体 PASS。

当前结论：focused 机制证据在限定范围内成立；已审 smoke/正式分片的标签与成本可重算；当前正式资料不足以支持门控收益、跨区处理效应、模型优劣或真实救灾结论。首版结论报告的跨区计数存在已实证的遗漏和误分类，须在修订后复核。

## 审核快照与复核方式

分片快照时间为 `reviews/V33_INDEPENDENT_EVIDENCE_CHECK.json` 的 `generated_at`（2026-10-20T08:17:34.374904+00:00）。逐分片校验 marker 自身 SHA-256、压缩数据 SHA-256、解压 JSONL SHA-256、行数、当前 source/config/baseline binding；重新调用独立实现的 gain_oracle，并另外用审核者编写的 `reviews/v33_independent_verify.py` 从原始事件和 endpoint 重算每个有效分支 J_total。没有仅接受原报告的 PASS 字样。

| 已审证据 | 场景 | 状态行 | ordinary | forced | 有效标签 | 正/非正 |
|---|---:|---:|---:|---:|---:|---:|
| smoke：六类扰动、每类3场景 | 18 | 278 | 48 | 230 | 45 | 0 / 45 |
| 正式主目录：none_or_light，12 train+4 validation | 16 | 222 | 40 | 182 | 40 | 0 / 40 |

smoke 的额外3个 mask 原因是 empty_obligation_window；其余 mask 均为 forced_safety_state。85个有效状态标签、各2个suffix的原始成本重算一致。500行标签的重算回执与持久化回执一致。主轨迹分别包含346和279个真实 planner receipt。全部34个分片均有90步轨迹、36平台、54个逻辑任务；审核快照内无哈希或成本重算错误。resume 目录不在这次分片快照的合并样本中，不能把同一 scenario 的复制/续跑文件直接累加。

## A. Ground truth 来源：PASS（仅模拟）

场景由固定seed的合成生成器定义：`package/civilian/v33_scenarios.py:64`、`:88`、`:251`，base显式标记 `synthetic=True`（`:264`）。truth来自实际环境状态与逻辑任务rollup（`event_environment.py:2359`），不是预测模型输出；paired标签来自同源状态的keep/replan实际分支（`counterfactual_pairs.py:1022`）。该事实支持 simulation_only，不构成真实数据集ground truth或现实外部有效性。

## B. 归一化、成本与残余惩罚：PASS（已审路径）

任务权重按共享obligation集合的任务priority总和归一化（`counterfactual_pairs.py:749`、`:835`；`gain_oracle.py:700`、`:784`），分母不取模型预测的最大/均值。两个分支使用相同权重和成本配置；原始delay/energy/risk/communication/switch/call/CPU分项保留。

完成任务在残余累计之前直接跳过（`cost_accounting.py:98`、`gain_oracle.py:514`），truth将completed progress固定为1（`event_environment.py:2416`、`:2444`）。物理事件计入J_exec；通信、切换、调用、重建规模计入C_replan；continuation的planner与safety CPU分开记账（`cost_accounting.py:199`、`:209`、`:213`；`counterfactual_pairs.py:525`）。初次replan调用由initial count计一次，replan_apply步不重复增加（`cost_accounting.py:140`、`:148`）。已审有效标签的独立加总一致。

这些是预设合成单位下的成本；不同单位与权重的实用合理性没有由本审核验证。不得把模拟成本数值直接称为真实救灾效益。

## C. Pair、未来信息与 focused exactly-once：限定支持

同一suffix由两个重新恢复的环境执行（`counterfactual_pairs.py:859`、`:1022`、`:1028`）；suffix只替换当前时刻之后的外生事件，并核验当前public observation/history不变（`:375`—`:430`）。生产replanner先于新suffix采样，按当前public状态候选与成本选动作（`v3_runner.py:319`—`:355`；`fixed_replanner.py:251`）。公开投影移除provenance（`counterfactual_pairs.py:149`），公开观察不含未来事件tape（`event_environment.py:396`）。

独立两时刻扰动检查另存 `V33_FUTURE_ISOLATION_INDEPENDENT_CHECK.json`。完整planner receipt包含 `candidates[*].repair_receipt.event_tape_id`（`event_environment.py:3321`），所以不能把完整receipt称作public-only输入；这是私有证据字段边界。当前报告没有训练模型，本审核不接受尚未执行的模型特征防泄漏结论。

focused脚本通过真实split/cooperate创建两个工作子任务，并故意把服务时间、位置、报告延迟设成可判别的测试夹具（`focused_check.py:56`—`:86`）。我独立重执行 `_run_mode` 两次，并另重算原始evidence内两个label：均在t=21最后子任务完成后只生成一次report，t=22收到一次，后续NOOP不重复；救援任务进入共享obligation；两个label均有效，gain=-0.050000000000000044。独立执行结果存 `V33_FOCUSED_INDEPENDENT_REEXECUTION.json`。

边界必须保留：脚本在救援已公开且又执行NOOP后才捕获paired snapshot（`:123`—`:143`），paired replan_action是NOOP（`:153`—`:157`）。因此该证据支持“split/cooperate生命周期exactly-once，以及已公开救援进入后续pair成本计算”；它不覆盖paired分支内部新发生split/cooperate和发现链，也不是收益性能实验。-0.05是NOOP对NOOP时replan arm的一次固定调用费用。

## D. 结果存在与调用：PASS（快照内）

原始分片、marker、完整trace、oracle回执存在且可重算。真实调用入口为 `v3_runner.py:254`、`:324`、`:344`、`:358`，不是仅有未调用的函数。forced标签结构有效与收益有效是不同含义；不能把500个 `oracle.valid=True` 都称作可用于学习的有效gain样本。

## E. 统计单位与scope：WARN / 结论未准入

冻结registry是7×(12 train+12 validation)=168场景（`v3_shards.py:137`）；当前主目录仅16个，全部none_or_light。smoke的45个有效状态不计入正式门槛。配置独立单位是scenario_instance；同一轨迹多个state、同一state两个suffix都不能当独立实验重复。

分析脚本 `tools/v33_conclusion_analysis.py:127` 先在场景内平均唯一有效ordinary state，再bootstrap场景，单位正确。当前有有效状态的train场景仅4个、validation仅1个；validation的退化bootstrap区间不提供可估计的抽样不确定性。无闭环门控、WM或本地/跨区随机对照，不能将标签均值CI称为策略收益或跨区处理效应CI。

冻结runner的30/30判定是train+validation合计并另检查两split各类及场景组覆盖（`v3_runner.py:737`—`:754`）；首版分析另算train-only30/30（`tools/v33_conclusion_analysis.py:325`）。当前两者均不通过，但未来不得混称同一冻结门槛。

## 尚需修正的结论问题

1. 首版 `conclusion.md` 报告771个跨区候选、98个选派。分析 `_candidate_transfer` 只遍历queue_operations（`:48`—`:66`），遗漏9个仅含跨区dispatch的候选及9个对应选派。按候选action/selected action计数应为780/107。例：`v33-train-none_or_light-search_rescue-n3-002-00008203-random_allow_suppress-t024-s008`，heli-r08-0派往rescue-r07-s01。
2. 首版称120个donor transfer；原始cross queue operations实际为insert651、transfer40、split41、cooperate39。split/cooperate共80个不能描述为整任务transfer。应按op语义分类，同时报告direct dispatch，写明是action数或operation数。
3. 正式采集尚未完成，不授予整体PASS；不能声称30/30通过、门控有效、跨区有效、WM值得训练或泛化稳健。当前40个ordinary有效状态均负，且分析记录只有NOOP合法候选，这更接近调用开销诊断。

以上是对结论准入的限制，不要求停掉已授权的采集。报告修订及新分片需要按新输入hash复核；本草稿不自动覆盖后续产生的证据。

## 可接受的当前表述

“在18个工程smoke场景和当前16个none_or_light正式场景的模拟证据中，持久化分片、paired标签与原始成本复算一致；两个定向split/cooperate夹具通过exactly-once检查。正式采集尚未完成，当前没有普通正增益状态，不能据此建立门控或跨区收益结论。”
