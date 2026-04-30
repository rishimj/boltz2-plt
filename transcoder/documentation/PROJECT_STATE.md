# Project State

**As of:** April 30, 2026

## Summary

The `transcoder/` subtree is the active research area in this repository. The current work is focused on collecting Boltz internal activations, training sparse universal transcoders over selected Pairformer layers, and validating whether those transcoders can be used for analysis and controlled intervention.

The project has moved past the earliest single-layer pilot stage. There is now working code for:

- activation collection from Boltz internals
- universal transcoder training
- multi-layer training variants
- validation and PLT-style verification experiments
- documentation of the math, architecture, and prior experimental results

At the same time, the directory had accumulated a mix of durable source code, useful documentation, stale runtime files, and old experiment outputs. This cleanup establishes a clearer distinction between those categories.

## What Is Active

These areas should be treated as the main working set:

- `transcoder/collection_scripts/`
  - scripts for collecting activations from Boltz runs
- `transcoder/universal_transcoder/`
  - current universal transcoder model and training entrypoints
- `transcoder/training_scripts/`
  - training helpers and analysis scripts
- `transcoder/validation_scripts/`
  - reproducibility and intervention validation
- `transcoder/scripts/`
  - PLT insertion and structure verification utilities
- `transcoder/documentation/`
  - durable project documentation
- `transcoder/analysis_output/`
  - small durable analysis artifact currently tracked in git

## Current Technical State

Based on the checked-in code and docs, the project currently has:

1. A working activation collection path that hooks Boltz Pairformer transition modules.
2. A universal transcoder implementation that maps single representations into sparse latent features and decodes pair representations.
3. Multi-layer and online-training variants for scaling beyond the earliest pilot experiments.
4. Validation scripts aimed at checking reproducibility and PLT-style intervention behavior.
5. Multiple prior experiment outputs and checkpoints, some of which were retained in git and some of which were only local generated state.

## What Is Legacy Or Generated

The following categories are not core source-of-truth and should be treated as generated or historical material:

- transient logs and `.status` files
- PID files from prior background runs
- Python cache directories
- local virtual environments and cache directories
- one-off inference/test outputs
- older archived experiment outputs under explicitly legacy locations

Some historical experiment artifacts remain in the repository because they are already tracked and may still be useful as reference data. The goal of this cleanup is not to erase all experimental history, but to remove the most obviously stale operational clutter and make the active paths easier to navigate.

## Recommended Entry Points

For someone resuming work, start here:

- `transcoder/documentation/TRANSCODER_PROJECT_SUMMARY.md`
- `transcoder/documentation/BOLTZ_TRANSCODER_ARCHITECTURE_MATH.md`
- `transcoder/documentation/DIRECTORY_STRUCTURE.md`
- `transcoder/universal_transcoder/train_universal.py`
- `transcoder/collection_scripts/collect_batch.py`
- `transcoder/validation_scripts/README.md`

## Repository Hygiene After This Cleanup

After this cleanup, the intended organization is:

- keep durable docs in `transcoder/documentation/`
- keep active source in the script/model directories
- treat runtime logs, PIDs, caches, checkpoints, and local outputs as generated state unless there is a specific reason to version them
- avoid committing new stale status files that encode old process IDs or one-off run state

## Next Practical Work

If development continues from here, the next high-value steps are:

1. Consolidate the preferred activation collection path and retire duplicated early scripts as they become clearly obsolete.
2. Decide which checkpoint and validation artifacts are truly canonical and which should live outside git.
3. Keep `DIRECTORY_STRUCTURE.md` and this file updated whenever the active workflow changes materially.
