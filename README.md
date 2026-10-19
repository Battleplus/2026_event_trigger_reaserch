# V3.2 性能与收益门控实验

执行：GPT-6 Luna；技术审核：主协调者只读审核，不冒称其他审核模型。

本分支执行 [V3.2协议](protocols/V32_PERFORMANCE_LIGHT.md)，与其他分支的 Isaac／完整世界模型研究分开。当前只完成性能诊断、两处配对接口修复及慢实例回归，完整 smoke 尚未通过；正式数据、模型、世界模型和 PPO 均未开始。

本次实际源码在 [独立发布目录](releases/gain-gate-v3-2-performance-light/)，包含准确的源码、测试、配置、来源和哈希。仓库根目录旧 `package/` 保留历史，不作为本次运行入口。固定场景为100×100、9区域、30平台、54任务、90分钟，候选上限16、标签窗口30分钟、每普通状态2个后缀。

诊断发现快照复制与序列化为主要热点；旧V3.1普通标签全部被两个接口错误屏蔽，不能用于训练。修复后的原seed慢实例完成90分钟，动作和物理轨迹哈希保持一致，8个普通状态获得有效标签。该诊断只保存了摘要，原始分支成本需从后续smoke完整分片再次独立复算；不把摘要当成原始数据。

运行入口（先进入发布目录）：

```powershell
python -B package/civilian/v3_runner.py smoke --output-dir evidence/smoke-runs --run-id <new-run-id>
```

批次上限1800 CPU秒、8GiB，必须18/18完成并经审核才进入168场景采集。再次超限则停止本版本。正式数据达到30正／30非正独立普通状态后，才允许CPU基线和GPU MLP训练。

服务器GPU硬件可达，但默认PyTorch环境存在CUDA动态库冲突；训练尚未启动。依赖文件沿用源版本，真实执行环境以运行收据为准，未宣称在全新环境完成安装验证。

查看 [状态](STATUS.md)、[性能诊断](reports/V32_PERFORMANCE_DIAGNOSIS.md) 和 [旧标签诊断](reports/V31_LABEL_DIAGNOSIS.md)。历史失败保留，不创建未通过阶段的成功标签。
