# Isaac State Capture Pilot

This is an isolated capture runner based on frozen attempt008, with the separately reviewed service-safety fix from the mutable pilot applied only to this collection copy. `mission.py` and `isaac_backend.py` are local dependencies, so it does not import mutable sibling modules. Run it from this directory with Isaac Sim 4.5 Python, the archived Isaac source tree, and a local output directory:

```powershell
& $env:ISAAC_PYTHON run_capture.py --config configs\capture_v1_representative.json --source-root $env:SOURCE_ROOT --output $env:OUTPUT --phase representative
```

The runner preserves attempt008 candidate generation and mission execution. At an ordinary gate opening, before candidate generation, it captures eligible states that have at least one public ordinary event since the preceding consumed gate. A decision timestamp with a forced platform fault, communication hold, or energy hold is excluded. Duplicate timestamps are ignored. A SHA256 priority derived only from the separate sampling seed, ordinary decision sequence, and simulation time selects at most eight states. Snapshot copying happens only when a state can enter the bounded top eight.

Each selected private pickle contains the trusted Isaac backend snapshot and complete common mission state for later internal restoration. Treat these files as trusted local data; they are not model inputs. `*.public.json` files are built with an explicit whitelist containing public aircraft kind/zone/availability and safety flags, current positions, velocities, available energy, active tasks, activated queue tasks, and public observations. They exclude truth, private mission state, future activation schedules, candidate catalogs, control plans, and scene identifiers. `state_index.json` carries SHA256 hashes for every private and public file; `state_index.sha256` hashes the index itself. `public_states.json` is the separate public listing.

`sampler_offer_wall_ns` and `sampler_offer_cpu_ns` cover only the sampler offer method; they exclude public projection construction and output serialization. This timing is kept outside the planner's measured generation timing and is not a total offline resource receipt. Each index entry records `decision_time_s`, `backend_time_s`, and `continuation_phase`. The current gate runs after public updates at decision time `t`, while the Isaac backend is still at `t-1`, before the next backend step. A later restoration must continue from this pre-step boundary and account for already-applied public updates; it cannot pass these pickle files directly to the existing A1 pair runner as an end-of-step snapshot. No plugin, automatic restore, retry, or exception recovery is included. The paired two-suffix rollout and independent service trajectory recalculation are not connected yet. Isaac has not been run for this capture pilot, so this package contains no runtime capture results.

The collection copy includes the separate availability and pause gate for continuous `search_strike` service dwell. It remains unverified in Isaac, and trajectory service recalculation is not implemented here.

## Focused checks

Run only the bounded sampler tests:

```powershell
python -B -m unittest test_state_sampling -v
```

The focused tests cover outcome-independent selection and public projection, deduplication and the eight-state cap, forced-state exclusion, and deep-copy isolation.

## Local Verification Receipt

Captured locally on 2026-10-20. Isaac execution: `NOT_RUN`. Paired two-suffix rollout: `NOT_IMPLEMENTED`. The collection config copies every attempt008 representative parameter except `run_id`, which is set to `isaac-roi-capture-v1-20261020`; the attempt008 source config SHA256 is `D57264A06B1012CCE5343CCCDC7760668413A722BB0A6DC8697A3A6A71FD683F`.

`python -B -m unittest test_state_sampling -v` exited 0 with stdout:

```text
test_duplicate_time_and_limit (test_state_sampling.StateSamplingTests.test_duplicate_time_and_limit) ... ok
test_projection_and_selection_ignore_gain_and_hidden_truth (test_state_sampling.StateSamplingTests.test_projection_and_selection_ignore_gain_and_hidden_truth) ... ok
test_selected_state_is_deep_copied (test_state_sampling.StateSamplingTests.test_selected_state_is_deep_copied) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.002s

OK
```

`python -B -m unittest test_collection_service_safety -v` exited 0 with stdout:

```text
test_paused_step_resets_dwell (test_collection_service_safety.CollectionServiceSafetyTests.test_paused_step_resets_dwell) ... ok
test_unavailable_step_resets_dwell (test_collection_service_safety.CollectionServiceSafetyTests.test_unavailable_step_resets_dwell) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.004s

OK
```

`python -B -m py_compile run_capture.py state_sampling.py mission.py isaac_backend.py` exited 0 with empty stdout.

SHA256 (collection sources and focused tests):

| File | SHA256 |
| --- | --- |
| `run_capture.py` | `6FEDA3E131F62874000CB39563F167CC2AAC021E13A9E82C02E5514193115328` |
| `state_sampling.py` | `907B49C6F89D4B280333C6ECEF47A1A7E56F32272037EBF658DC6572559042C0` |
| `mission.py` | `0BCE912C53822B90002D5166A45C09459F35CB3FCE263716EDA7D2C76C48E9B1` |
| `isaac_backend.py` | `4B4C67BEC549F1A34694DFF2B9C3767A7B9213728A6F607EF36A92DDB01AF504` |
| `test_state_sampling.py` | `2F430A118E2FDC4D91078F6237568FDADA09F5483DDBB614C3F274D4F5975255` |
| `test_collection_service_safety.py` | `998D905C202EA2621EADC09C5C71F0325C5ED237F14B01653E3E90888BE384A0` |
| `configs/capture_v1_representative.json` | `55890A9C544BBC6046AA3CFF12DAFDE450100155E639F05A659CE57D903AE46B` |

Config identity follow-up checks: JSON parse exited 0 (`JSON_OK`); structured comparison after substituting only `run_id` exited 0 (`PARAMETERS_MATCH_EXCEPT_RUN_ID`); `python -B -m py_compile run_capture.py` exited 0 with empty stdout. No unit suite rerun and no Isaac execution was performed for this config-only identity correction.
