# Isaac formal collection: launched, not completed

Updated 2026-10-21 (Asia/Shanghai).

The user removed representative-scene, one-state two-suffix and 36-scene pilot
prerequisites. The new batch directly acquires 168 formal independent scenes:
seven conditions, 12 training and 12 validation scenes per condition. Each scene
uses a 90-minute Isaac trajectory to sample at most eight ordinary states, each
with two paired suffixes. Failed scenes are retained and other scenes continue;
no retries, replacement seeds or repeated preliminary audits are scheduled.

Luna implemented the collector and independent reducer; primary coordinator
reviewed the interfaces. Collector: six test functions directly passed; standard
pytest reported three passes and three setup permission errors, not assertion
failures. Reducer: ten synthetic unittest checks passed. These are not new Isaac
experiment results.

Batch launch confirmed at 2026-10-20 16:07:55 UTC (2026-10-21 00:07:55 local).
Isaac reported app ready. The first formal scene reported actual_isaac_run=true,
36 aircraft and 81 public search tasks; compressed raw trajectory was writing
(24,916,831 bytes at the first progress observation). GPU1 memory rose to about
2.3 GiB. No completed scene or gain conclusion is claimed at this checkpoint.

The transferred archive SHA256 is
`7ff189d4b0303ed5414b714aacede5c6e4d00bf6d49a857c289ba989df3a1f48`.
Registry SHA256 is
`d09eb7165511c34614219aa90c03cc9b0c14b13a766409d60bb7a82c76aabf70`.
The original archive checksum text used CRLF, causing Linux filename checks to
fail before launch. An LF-only checksum file verified the same source/config
bytes; the original checksum and failed verification receipt were retained.
No scene was retried for this packaging correction.

Thirty positive/thirty nonpositive unique ordinary states and train/validation
coverage are assessed once at collection end. Models, world model and PPO have
not run. The earlier non-Isaac analytical run is explicitly excluded.

Search with three targets per region is a frozen design value, not a pilot-selected
optimum. Source kinematic models, analytical energy/communication and 1-second
planning-time quantization remain scientific limitations.

Later observation: the first scene 90-minute acquisition receipt reached complete,
with 76,881,535 compressed trajectory bytes. Its first suffix worker was active
under the same batch process. No full-scene pair completion or label is claimed.
