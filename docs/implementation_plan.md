# Implementation Plan

## Phase 0 — Audit & research contract ✅

- Repository inspected (was empty)
- Reference docs archived
- `docs/source_of_truth.md`, ethics, data card, experiment protocol created

## Phase 1A — Data pipeline ✅ (code) / pending dataset file

- Schema, load, validate, stratified 70/15/15 split, quality report
- Command: `python -m src.data.prepare`

## Phase 1B — Five-model experiment framework ✅ (code)

- 10-model registry; Phase 1 config-driven five
- Shared training command: `python -m src.training.train`

## Phase 1C — Evaluation & reporting ✅ (code)

- Metrics module, comparison CSV/JSON, phase1 report generator
- Figures via `python -m src.evaluation.evaluate`

## Phase 1D — Paper & presentation drafts ✅ (stubs with TBD metrics)

- `research/paper/phase1_draft.md`
- `presentation/phase1/outline.md`

## Phase 2 — Explainability

- SHAP/LIME generators + concordance metrics implemented
- Run after at least one trained artifact exists

## Phase 3 — Calibration & threshold study

- Isotonic calibration on validation; threshold discussion in report

## Phase 4 — FastAPI

- Health / model-info / predict scaffolded; Dockerized

## Phase 5 — Flutter mobile

- Placeholder under `mobile/flutter_app/` (to be generated after model freeze)

## Sequence for a developer this week

1. Confirm five models (or accept proposed list formally)
2. `python scripts/download_dataset.py` (or manual place CSV)
3. `pip install -r requirements/base.txt`
4. `python -m src.data.prepare`
5. `pytest`
6. Dev smoke: set `dev_train_subsample: 5000` in phase1 config, run train with `--skip-test`
7. Full Phase-1 after confirmation: unlock official run flags, train, evaluate, report, fill paper TBDs programmatically
