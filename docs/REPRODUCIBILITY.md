# Decision2Care reproducibility scope

## What this release establishes

- The core diagnostic router and ICD-10-CM scorer are available under a unified package.
- Medication and procedure selection accept an explicitly restored diagnostic chart, without depending on a historical local run directory.
- The forecasting router accepts a time-gated record snapshot and a read-only retrieval catalog, applies the supplied disease prompt, and makes one final four-window forecast.
- A fresh installation can run all three synthetic workflows without model credentials or patient data.

The model backend for the research implementation is Jev 1.13.0. Both inference clients check the returned model version. A model version being named here does not guarantee current provider access. Source code is provided; model weights are not distributed.

## What is not a full-paper reproduction

The public package does not contain the fixed 546-admission diagnostic/comparator dataset, all mortality-cohort selection lists and their original source tables, MIMIC-derived treatment menus, physician responses, original response caches, repeated-run archives, comparator orchestration, or figure scripts. The raw-data cohort builders and ICU-hourly / ICU24-hour runners remain outside this initial release. Consequently, a passing synthetic test is not a reproduction of the manuscript's reported numerical results.

Original task prompts and decision methods are traced in [SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json). The portable forecast wrapper supports the eight ED8-hour disease premises. It standardizes input loading and prompt selection and does not claim byte-for-byte equivalence to every historical disease/dataset runner or optional context-compaction mode. The AKI primary request is explicitly checked against the native source implementation during release validation.

The code retains legacy optional diagnostic settings in the underlying classes for source traceability. The supported public runner is the primary record-history workflow; optional patient-dialogue and LLM-helper variants are rejected. The synthetic ICD tree and menus are intentionally small and cannot substitute for the observed code universe and clinical menus of a research run.

## Run evaluation after inference

Reference labels are loaded only by the offline evaluator:

```bash
python scripts/evaluate.py --task prognosis \
  --input private/evaluation_rows.json --output private/metrics.json
```

Evaluation rows depend on the task:

| Task | Fields per row | Measure |
| --- | --- | --- |
| Diagnosis | `predicted_title`, `reference_pathology` | Existing strict MIMIC-CDM title-matching rule |
| Treatment | `reference_positive`, `selected` (booleans) | Precision, recall and F1 for the explicitly supplied reference |
| Prognosis | `label_dead` (0/1/null), `mortality_score` | AUROC, excluding unknown outcomes |

For treatment, select `--treatment-reference indication_rubric` or `documented_care` to identify what was scored. Recorded-care agreement, rubric adherence, clinical correctness, and safety are different constructs. These metrics do not establish treatment safety or prospective clinical benefit.

For prognosis, evaluate the intended horizon separately. The returned scores are `p(death ≤7 days)`, `p(death ≤30 days)`, and `1 − p(alive at 90 days)` derived from the four-option distribution. They are uncalibrated ranking scores. The horizons are measured after `prediction_time`.

## Local source audit

The local diagnosis/treatment source snapshot contains the primary agent, record tools, ICD hierarchy, medication selection, procedure selection, raw-data preparation utilities, and diagnostic/treatment evaluation scripts. The separate prognosis source contains the eight disease drivers, ED8-hour record preparation, ICU-hourly and ICU24-hour workflows, and historical frozen versions. The manuscript repository contains figure/statistical verification scripts. This first release extracts the three core inference workflows rather than uploading these complete workspaces.

The local prognostic archive itself notes that full cohort source tables are separate. Therefore, the presence of all inference code does not imply that an arbitrary new user can reconstruct the original cohorts from this public repository alone.
