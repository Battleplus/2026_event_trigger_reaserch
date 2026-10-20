# Chapter 2 experiment status — 2026-10-20

## Current conclusion

The simplified Isaac representative run and its old pair audit provide engineering evidence that the 36-platform scene, six disturbance records, safety holds, and state restoration can execute. They do not provide a valid scientific gain label because the old pair had an unequal replan timing path and other audit defects.

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| Isaac representative engineering run | `ENGINEERING_PARTIAL` | attempt007 representative audit; retained as historical evidence |
| Fair keep/replan suffixes | `NOT_RUN` for the repaired package | repaired pair runner is archived; no new remote suffix trace |
| Independent gain reduction | `READY` | local focused tests pass; raw trace required |
| 30/30 gain gate | `NOT_ASSESSED` | no valid two-suffix state labels |
| Light models | `NOT_RUN` | 30/30 gate not met/assessed |
| World model and PPO | `NOT_RUN` | conditional on light-model evidence |
| Scientific method conclusion | `INCONCLUSIVE` | required data and independent test evidence are absent |

## Required next operation

After explicit authorization for a new remote directory, run one representative capture, audit it, run two suffixes from one ordinary state, reduce both suffixes, and aggregate by `state_id`. Do not start the 36-scene pilot or model training before that sequence passes.

## Archive state

Local archive commits through `17b59ae` contain the repaired reducer, split-coverage gate, pair-source update, tests, hashes, and run instructions. GitHub push is pending because the configured network proxy currently rejects the connection.
