# V3.3 Civilian Search, Rescue, and Cross-Region Support

**Status: implementation in progress.** The 18-instance smoke, configuration pilot, formal collection, and model training have not run. No method-performance result is claimed. The V3.2 stop and its partial evidence remain historical and are not V3.3 results.

## Scope

This version studies civilian waypoint search and water-delivery rescue tasks. It does not model unstructured grid detection, human evacuation, or strike missions. Initial assignments come from the public-information C1 adapter; that adapter is not represented as the user's complete Chapter 1 solver.

The simulated map is 100 x 100, divided into nine responsibility regions. Each region has two UAVs and two helicopters, for 18 UAVs and 18 helicopters total. Platforms may support other regions; the intended support search proceeds from home region to neighboring regions and then to global fallback, subject to task, energy, communication, and safety constraints.

The task mode is `search_only` or `search_rescue`. The per-region target count `n` is one of 1, 3, or 6 and includes one latent search target in every region. All nine latent searches arrive in every disturbance condition, one addressed to each region, at independently sampled times from minute 22 through minute 26. The `task_update_or_arrival` condition adds one deadline update to an initially public task when one exists; at `n=1`, the update is omitted because there is no initially public task. In `search_rescue`, every target also has a dependent water-delivery task that becomes available after its search. Thus `search_only` has `9*n` logical tasks and `search_rescue` has `18*n` logical tasks. At `n=1`, there are intentionally no initially public search tasks; all nine searches are latent arrivals.

Episodes run for 90 simulated minutes. Local disturbances occupy 2 x 2 map cells and are independent of the responsibility-region boundaries. Replanning has at most 16 candidates, including NOOP. Paired keep/replan labels use a 30-minute horizon, truncated at episode end, with two matched future suffixes.

## Commands

Run from the project directory. These are the runner's current command forms; they have not yet been executed for V3.3.

```powershell
python -B -m package.civilian.v3_runner audit --output-dir runs-v33
python -B -m package.civilian.v3_runner validate --output-dir runs-v33 --task-mode search_rescue --targets-per-region 3
python -B -m package.civilian.v3_runner smoke --output-dir runs-v33 --task-mode search_rescue --targets-per-region 3
python -B -m package.civilian.v3_runner pilot --output-dir runs-v33
python -B -m package.civilian.v3_runner collect --output-dir runs-v33 --task-mode search_rescue --targets-per-region 3
python -B -m package.civilian.v3_runner verify --output-dir runs-v33 --run-id run-id-from-prior-command
```

`smoke` is six disturbance families by three instances. `pilot` compares two task modes, three target densities, and three shared seeds. The formal target is seven conditions by 12 training and 12 validation instances per condition. The formal registry is restricted to those 168 training/validation instances; pilot scenarios are separate.

There is no custom time- or memory-guardian termination in this version. Runs record observed resource usage and preserve actual failures. Raw runs are local evidence and should not be added to a public source archive.

## Provenance and review

V3.2 source hashes and file-level reuse decisions are in [REUSE_INVENTORY.json](REUSE_INVENTORY.json). The source archive is pinned in [BASELINE_PROVENANCE.json](BASELINE_PROVENANCE.json). V3.3 changes require a later source/config/evidence freeze and coordinator review before smoke or collection is treated as accepted evidence.
