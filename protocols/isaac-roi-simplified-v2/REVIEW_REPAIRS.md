# Protocol Repair Record

This revision corrects semantic and counting errors identified in read-only review. It is a record of edits, not an approval or PASS decision.

- Separated safety-critical `forced` decisions from ordinary decisions, event causes, unknown labels, and empty observation windows.
- Corrected A3 to at most eight decision states per trajectory, two future suffixes per state, and paired keep/replan branches per suffix. Positive/non-positive thresholds count deduplicated `state_id`s; the 168-trajectory cap is hard, and unmet thresholds stop the learned comparison.
- Corrected A4 to 42 sealed independent scenes (seven classes x six), with all six methods and three seeds paired on every scene: 756 method-seed runs. Post-freeze recollection applies only to reconstructor train/validation gain labels.
- Separated online command-effect latency from simulator/application stepping wall-clock cost.
- Defined `G` as graph PPO with ordinary critic and `GWM` as frozen WM plus auxiliary critic; auxiliary gradients update critic only, not actor directly or frozen WM, and do not change trigger rules. ET/light/WM-gate/4-minute remain independent conditions.
- Excluded identifiers from neural features and required physical separation of public observations and hidden state.

No experiment result, stage acceptance, or PASS status is asserted by this repair record.
