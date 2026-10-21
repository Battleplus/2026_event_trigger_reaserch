总体 **WARN**；`review_independence=same-family`，`acceptance_status=provisional`。没有发现伪造 GT、预测值自归一化或主要数字不实；不能授予方法有效性、全面安全或泛化方面的 PASS。

本次进行了只读源码审查、168 个 marker 的签名/绑定/注册表核对、报告内部重新计数、一个正式分片的完整哈希校验与正标签 oracle 复算、两个 focused 原始标签的 oracle 复算。未重跑模拟或全部历史测试，也未重新读取全部正式分片的所有原始成本。

- **A — PASS，来源是模拟轨迹。**
  `package/civilian/v3_runner.py:168` 实例化并运行环境；`counterfactual_pairs.py:235`、`:863`、`:687` 从环境取得起点/终点 truth；`:1022`、`:1028` 执行同 suffix 的两条恢复分支。标签不是收益模型预测。`v33_scenarios.py:264` 显式标记 synthetic。两臂共享外生 suffix 的机制见 `counterfactual_pairs.py:375`，公开前缀不变检查在 `:427`。这些证据支持 **simulation_only**，不支持真实救灾 ground truth。

- **B — PASS，附 CPU 计时限制。**
  权重分母是共享 obligation 集合的任务 priority 总和：`counterfactual_pairs.py:835`、`gain_oracle.py:784`；没有除以模型自身预测最大值/均值。完成任务在残余累计前跳过：`cost_accounting.py:98`、`gain_oracle.py:514`。物理、规划和 safety CPU 分项保存并独立复算。
  CPU 是实际 `process_time` 测量值，见 `counterfactual_pairs.py:536`、`:558`；不能把小幅 CPU 差异解释成物理调度改善。

- **C — PASS，限数字和本次核验范围。**
  实际找到 **168 个不同 scenario marker**，与冻结注册表完全一致，14 个 split×condition 单元各 12 个；所有 marker 自身哈希、当前源码/配置/baseline binding、对应数据文件存在性通过。重算清单 SHA-256 与报告一致：`9f6963b87c796714bc7c3001f88310c340f60d88e5b1bcf2c49f02986041f9b3`。注册表与绑定实现见 `v3_shards.py:112`、`:118`、`:137`；报告快照见 `analysis.json:101` 附近的 `snapshot` 字段。
  从 `analysis.json.states` 重新统计得到 **2,880 / 2,432 forced / 29 empty / 2 unknown / 417 evaluable / 3 positive / 414 negative**；summary 所有共享字段与 analysis 相同。逐状态平均 gain 为 **−0.06458849932681549**。
  正式原始样本 `shards/train/platform_failure_or_capability_loss/v33-train-platform_failure_or_capability_loss-search_rescue-n3-007-00010208.jsonl.gz` 第 6 行，经完整 marker/压缩/解压哈希校验及 oracle 复算，gain 为 **0.10518005041462719**，与报告一致。
  `COLLECTION_PROGRESS.json` 仍是旧进度，不能用其 118 判断未完成；`RETRY_PROGRESS.json:93` 为 remaining=0、`:98` 为 verified_committed=168，最终应以 marker 注册表为准。

- **D — PASS，oracle 确实被调用。**
  正式生成调用为 `v3_runner.py:344`、`:358`；结论分析对唯一状态调用 `validate_gain_label` 和 `recompute_gain_label`，见 `tools/v33_conclusion_analysis.py:420`。oracle 不导入生产 scorer，见 `gain_oracle.py:1`；从原始事件/endpoint 重算 costs，见 `:489`、`:834`、`:856`。
  “独立”应限定为独立实现的证据复算；它不是另一物理模拟器，也不是外部真实世界验证器。`oracle.valid=True` 表示结构有效，不能等同 `label_valid_mask=True`。

- **E — WARN，需要补充以下边界。**

  1. **普通标签实际上全部来自 Periodic-4min。**
     168 个场景的三种 trajectory source 各 56 个，但 **448 个普通状态、417 个有效标签、12 个有效跨区状态全部来自 Periodic-4min 的 56 个场景**。Existing-ET 与 Random-Allow-Suppress 各 56 场景没有普通标签。根据 `analysis.json:52201` 的 scenarios 与 `:94721` 的 states 关联重新统计所得；源选择/采样机制见 `v3_runner.py:230`、`:251`、`:284`。
     当前 FINAL 没有显式说明这一点。应补充，避免将 gain 分布理解为三种来源都获得了普通标签支持。场景 bootstrap 实际也是 train 28、validation 28，而不是每 split 84，现有 `conclusion.md:26`、`:27` 数字正确。

  2. **343 是安全处置核验未解决的回执数，全部原因都是 pre-public hash mismatch。**
     我对 `analysis.json.scenarios.*.live_safety_receipt_audit.actual_unresolved_unknown_reasons` 重计，343 个全部为 `safety_receipt_pre_public_hash_mismatch`；没有 unknown-resolution 布尔缺失。其代码含义是核验前态摘要不一致，见 `gain_safety.py:423`—`:426`，因此无法确认处置，不等于独立确认发生了 343 次物理安全违规。
     `FINAL_CONCLUSION.md:15` 已说明不是安全证明，但应直接补上 **“343 次均为前态哈希不一致导致的处置核验 unknown”**。也不能因 header 总数匹配就说安全模块无缺陷。两个 validation masked 标签与这些主轨迹计数属于不同统计对象。

  3. **3 个正状态标签没有 NOOP，也没有跨区动作。**
     三者均为本区 insert：gain 分别 **0.1051800504、0.2270037447、0.5586758100**，见 `analysis.json:403782`、`:774881`、`:825861`。第三个状态的两个 suffix 为 **−0.2015070436 / +1.3188586636**，均值为正不代表 suffix 稳定。
     另有 **3 个 NOOP 单 suffix** 为正，见 `analysis.json:144904`、`:257764`、`:582635`；其 `J_exec` 完全相同，正值来自 CPU 计时差抵消一次 **0.05** 调用费用。对应状态均值仍是负数，391 个 NOOP 中 **没有正状态标签**。建议报告明确区分，禁止把单 suffix 正值加进 30/30 正例。

  4. **跨区子集的区域/全局比较目前口径正确，但只支持描述。**
     `tools/v33_conclusion_analysis.py:258` 先选择普通有效且 selected-cross 的状态，再在同一子集计算区域与全局成本；12 状态=12 场景、24 suffix。全局 replan−keep **+0.18467739696**，CI 反号后 **[+0.1406057550,+0.2344692901]**，FINAL 数字正确。
     该子集含 **11 条 home-region insert、2 条 direct dispatch**，没有 selected donor transfer/split/cooperate。角色按平台 home 区分类，不证明平台空闲，也不证明 donor 已放弃本区任务。区域 endpoint 是全区域任务的未加权计数，而 completion cost 是 obligation 加权指标，不能将前者简单相加解释成本。现有 `conclusion.md:63` 已保留主要限定；FINAL 中“跨区动作可生成和执行”最好注明实际选中的是 insert/direct-dispatch。

  5. **30/30 的修订口径正确。**
     冻结 runner 是 **train+validation 合计正≥30、非正≥30，并且两个 split 各自两类分别覆盖至少两个场景**，不是 train 单独 30/30；见 `v3_runner.py:737`—`:754`，分析实现见 `tools/v33_conclusion_analysis.py:613`—`:622`。本批总正例 3，且 train 正例仅一个场景，两条均失败。无权将 smoke/forced/suffix 当额外独立正例。
     `conclusion.md:214` 仍说“仍需检查 30/30”，与已经得出的门槛失败矛盾，应改为“已检查，门槛未通过”；生成源为 `tools/v33_conclusion_analysis.py:807`。

  6. **冻结绑定的范围必须准确。**
     `v3_shards.py:112` 绑定的是 `package/civilian/*.py`，`:120` 绑定 configs JSON 及 baseline；不包括分析脚本、测试或 focused 脚本。当前所有 marker 与当前绑定相同。报告修订本身应另存审核输入 hash，不能声称 marker 已冻结所有后处理源码。oracle 检查 pair 携带的两臂 cost config 一致，见 `gain_oracle.py:817`，并不自行对照外部冻结配置；本次抽样参数与冻结默认值相同。

- **F — PASS：分类为 `simulation_only`。**
  这是合成环境内短时 paired label 收集与工程协议验证。没有训练后门控闭环比较、独立 test/OOD、WM/PPO 结果，也没有三种采集策略的成对性能比较。继续 Existing-ET 可以表述为保留当前基线的决策，不能写成它已被证明优于其他策略。

**Exactly-once：限定 PASS，整体覆盖 WARN。**
`evidence/v33-exact-once/focused_check.py:56`—`:86` 主动构造定向夹具；`:123`—`:136` 检查 split/cooperate 后只生成/接收一次发现报告及重复 NOOP 不增加报告。原证据各模式报告生成 t=21、接收 t=22、重复增加 0，见 `evidence.json:386049`—`:386053` 与 `:773778`—`:773782`。
我重新调用 oracle 验证两个 `raw_paired_label`，均有效、gain=−0.05。但 paired snapshot 在 **t=24、救援已公开之后**才捕获，且 replan action 为 NOOP，见 `focused_check.py:138`—`:157`。所以支持“定向生命周期 exactly-once；已公开救援进入后续 pair obligation”，**不覆盖 paired 分支内部新发生 split/cooperate→发现的整条链，也不是 split/cooperate 收益实验**。原独立重执行记录存在于 `reviews/V33_FOCUSED_INDEPENDENT_REEXECUTION.json:2`，本轮没有重跑它。
