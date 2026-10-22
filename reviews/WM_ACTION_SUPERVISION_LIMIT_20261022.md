# World-model action supervision limitation

Coordinator read-only source finding; no new formal data processing or training.

`wm_ppo_v1/raw_public_windows.py::_tick_frame` currently emits `PublicHistoryFrame(action=None)`. `features.py` encodes a missing action as the all-zero five-dimensional vector, explicitly meaning unknown rather than NOOP. The five coordinates are known, applied, normalised magnitude, normalised duration and held.

`trainer.py` records overall action-known fractions and dynamic validity counts, but does not currently persist per-dynamic-dimension joint counts for known executed actions. Overall dynamic target coverage therefore does not establish action-conditioned supervision support. Merely changing a candidate action input during inference would not establish that the frozen model learnt its effects.

Consequences:

- State-history dynamic prediction and pre-candidate gain prediction remain separate, potentially usable paths after their existing data gates. This finding is not a claim that they fail.
- G-WM auxiliary loss must not invent executed-action information or assume that all dynamic dimensions are action-supervised.
- The PPO core source interface must receive an explicit joint supervision-support mask. Synthetic CPU gradient tests may supply known actions, but cannot establish real-data support.
- Real G-WM training needs a documented public-command-to-action binding and measured joint training support, or must report the action-conditioned auxiliary path unavailable. No new samples, hidden labels or automatic all-ones masks are authorised to conceal this limitation.

The live formal collector remains unchanged. This is an outstanding dependency for genuine G-WM execution, not a reason to restart the current batch or to claim a completed scientific result.
