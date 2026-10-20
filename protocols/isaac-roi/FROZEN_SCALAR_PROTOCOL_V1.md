# Frozen scalar label protocol v1

Status: FROZEN FOR PILOT LABEL REDUCTION, NOT YET APPLIED TO ANY RUNTIME RESULT. Frozen 2026-10-20. This document fixes the reduction before any paired suffix labels are computed. It may not be changed after seeing pilot gains; any change creates v2 and invalidates affected labels.

## Evaluation window

For each ordinary state and each suffix, evaluate exactly `H = min(1800, 5400 - (decision_time_s - 1))` one-second mission steps. The paired branch starts from the same pre-physical-step state and follows the same suffix tape. A suffix is a repeated future draw, not an independent state. Unknown or failed evaluator fields remain `valid_mask=false`.

## Execution score

Let `N` be the evaluator-only count of unique hidden targets in the scenario. It is never controller input. Let `d` be unique discovered targets and `s` unique abstract-service completions, each target counted at most once. For search mode, `Q_terminal = 1 - d/N`; for search+abstract-service, `Q_terminal = 0.5*(1-d/N) + 0.5*(1-s/N)`. `L` is the one-second integral of the same unresolved-work quantity over the valid window divided by its valid duration. No deadline delay is invented for tasks without configured deadlines.

`J_exec = Q_terminal + 0.2*L + 0.01*(energy_kWh/1000) + 0.25*controllable_violation_count`.

The dimensions and raw components are always reported. Forced decision states are excluded from ordinary-state sampling. Safety interventions that occur later inside either suffix remain part of its execution consequences; they must not be discarded to improve the score. Missing or unmeasurable consequences make the label unknown.

## Incremental resource cost

`C_increment = 0.01*(messages_increment/1000) + 0.05*(switches_increment/100) + 0.05*(planner_calls_increment/100) + 0.01*(changed_actors/36) + 0.01*planning_cpu_seconds_increment`.

For messages, switches, planner calls and planning CPU, incremental means the signed difference (replan minus keep) over the paired window, not a second subtraction of the full replan total. Changed actors means unique platforms whose assignments are changed by the initial replan; subsequent common safety changes are reported separately. Only these incremental resource quantities are subtracted. A communication, switch, waiting, or execution consequence already represented in `J_exec` is not subtracted again. Simulator wall time, GPU training time, and offline snapshot/label generation time are reported separately; they are not silently inserted into `C_increment`.

## Gain and validity

`G = J_exec_keep - J_exec_replan - C_increment`. Positive means replan has lower unresolved-work/physical score after the incremental cost. `G <= 0` is non-positive; no positive-class resampling or rounding threshold is applied. `forced`, empty-window, unknown, failed, or duplicate `state_id` rows are excluded from 30/30 and retained with a reason code. Pairwise suffix mean is the state label; suffix variance is reported and is not an extra sample.

This protocol does not claim deadline optimization, monetary cost, real radio cost, or physical energy measurement. It supplies the fixed scalar needed to test learnability and must be independently recomputed from raw suffix traces rather than imported from controller output.
