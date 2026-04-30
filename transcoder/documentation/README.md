# Transcoder Documentation

This directory documents the active transcoder research work for Boltz internal representations.

Start with `PROJECT_STATE.md` for the current repository state and `DIRECTORY_STRUCTURE.md` for the current layout.

## Overview

The current codebase supports:

- activation collection from Boltz internal layers
- universal transcoder training over single and pair representations
- multi-layer training variants
- PLT-style validation and intervention experiments

## Main Entry Points

- `PROJECT_STATE.md`: current status and orientation
- `TRANSCODER_PROJECT_SUMMARY.md`: high-level technical summary
- `BOLTZ_TRANSCODER_ARCHITECTURE_MATH.md`: detailed architecture and math
- `DIRECTORY_STRUCTURE.md`: current directory map
- `QUICKSTART.md`: workflow-oriented getting-started guide

## Active Code Locations

- `../collection_scripts/`: activation collection
- `../universal_transcoder/`: main model and training code
- `../training_scripts/`: analysis and older training helpers
- `../validation_scripts/`: reproducibility and intervention checks
- `../scripts/`: PLT utilities

## Notes

- Earlier pilot-stage paths described in older docs are historical and may no longer match the active file layout.
- Generated logs, PID files, and local outputs should not be treated as durable project documentation.
