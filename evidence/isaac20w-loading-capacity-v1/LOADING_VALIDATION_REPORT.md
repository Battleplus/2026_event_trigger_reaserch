# Isaac 20w loading and capacity verification

2026-10-19. Result: **30_PLATFORM_KINEMATIC_LOAD_PASS**. This is a short loading/capacity result, not a 90-minute experiment or method-effectiveness result.

Isaac Sim 4.5.0 on an RTX2080Ti11GiB, driver535.161.08. Native scene:300km square,100m/USD-unit,1500m terrain,x10 vertical exaggeration,grid128,140 trees. Combat targets/demo disabled; weather effects disabled for this diagnostic. Source movement and geometry reused with a narrow USD transform compatibility adapter.

| Stage | Kit/app updates | Mean scene update | Mean app update | Complete stage wall time |
|---|---:|---:|---:|---:|
| 2_platforms | 300/300 | 6.03ms | 9.09ms | 5.31s |
| 30_platforms | 300/300 | 84.86ms | 9.03ms | 30.45s |

GPU1 sampled total memory peak:1839MiB(~1.80GiB). Process RSS peak:4277MiB(~4.18GiB). GPU0 remained0% sampled utilization but had a3MiB enumeration increase(6→9MiB); strictly zero GPU0 footprint is not demonstrated. Both platform stages moved and had zero maximum USD/motion position error recorded. Exit code0; source and config hashes independently matched the executed files.

The 30-platform scene uses30 quadrotor entities. This does not validate the eventual24UAV+6helicopter mission mix or54 tasks. Atdt=1/60s,30-platform scene+app mean update is93.89ms; complete300-frame stage takes30.45s for5s of nominal scene time. Current configuration is slower than real time. GPU memory capacity is not the observed short-test bottleneck; CPU scene update dominates measured frame time. GPU kernel timing was not directly measured.

Preserved nonfatal log issues: optional asset browser extension cannot import requests; repeated material displayColor warnings; CUDA enumeration warnings when using an explicit device mask. No packages were installed and no driver was changed. Earlier unmasked attempt was terminated whenGPU0 acquired188MiB of context; the next attempt lost final receipts because app.close preceded persistence. Both attempts remain failures/unaccepted; the accepted run persists receipts before shutdown.

Reproduction: provide the six hashed scene dependencies from the user-supplied archive, use the archived script/config and the installed Isaac Python. Expose physicalGPU1 through its UUID, render on the Vulkan GPU1 ordinal, use masked CUDA ordinal0, and verify the device table. Do not assume Vulkan and CUDA ordinals remap identically. Full source ZIP and raw private logs are not published in this archive.
