# Decision2Care input formats

Run commands from the repository root. All bundled examples are invented schema fixtures. They contain no MIMIC records or patient identifiers.

## Diagnosis

`--record` follows [examples/diagnosis_record.json](../examples/diagnosis_record.json). Required fields for the primary configuration include `Patient History`, `Laboratory Tests`, and the explicitly prepared completeness flags. Optional record fields hold examination findings, vital signs, radiology reports, home medications, and the allergy-bearing source note. Integer laboratory item identifiers become JSON string keys and are restored on loading.

The primary configuration serves masked, impression-preserving radiology through `Report MIRA`; `MIRA Reports Loaded` records that this preprocessing has been done. `Microbiology Complete` indicates that the full microbiology results were included, including negative cultures. These flags are assertions about preparation, not a substitute for performing it. Reference pathology names and final diagnoses must not appear in the model-visible history or reports.

`--menus` follows [examples/menus.json](../examples/menus.json). It contains laboratory dictionaries and groups, panel membership, microbiology options, and imaging studies. Use a fixed action universe across cases. The small example menu is for execution testing only.

`--tree` follows [examples/icd_tree.json](../examples/icd_tree.json), with native `Node` fields, parent/child links, and chapter roots. The exporter builds this from an authorized resource directory containing:

- `d_labitems_min_1.csv`, `micro_tests_universe.csv`, and `lab_test_mapping.pkl`;
- `imaging_universe.csv` (plus extra imaging resource directories when needed);
- `icd_diagnosis_universe.csv` and `d_icd_diagnoses.csv`.

These are the research implementation's prepared resource files, not a raw MIMIC export. Supply the original authorized `d_icd_diagnoses.csv.gz` through `--icd-dictionary` if the prepared directory does not contain an uncompressed copy. `lab_test_mapping.pkl` must be a trusted locally prepared pickle. This release provides resource conversion, not the full cohort extractor. For raw cohort preparation, consult the original [MIMIC-CDM implementation](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset) and [MIRA dataset workflow](https://github.com/Dyke-F/MIRA/tree/main/src/dataset).

The diagnostic output preserves the original visible chart and a fingerprint of the supplied record. Treatment verifies both. Do not edit diagnostic outputs to mix records or admissions.

## Treatment

`--treatment-menus` follows [examples/treatment_menus.json](../examples/treatment_menus.json):

- `purposes`: purpose name → ordered regimen options. A regimen contains `items`, `drugs`, and `shown`.
- `doses`: `default` and the two separate anticoagulation purpose names → drug name → dose options, each with `shown`.
- `procedure_categories`: two-digit ICD-9-CM group → ordered `[code, title]` pairs.

The original research workflow used a purpose-to-regimen table, a purpose-specific dose table, and the frequency-ordered MIMIC-IV procedure catalog. Combine those locally prepared resources under the three keys above without sorting or changing their option order. Preserve therapeutic and prophylactic anticoagulation as separate purposes. Replace all synthetic options before research inference.

The initial public release accepts these prepared catalogs; their raw-MIMIC generation pipeline is outside this release. Missing catalog resources are a prerequisite to resolve locally, not a reason to run with the example catalog on real records.

## Prognosis

`--case` follows [examples/prognosis_case.json](../examples/prognosis_case.json): a versioned snapshot with `id`, `title`, `provenance`, `prediction_time`, and `records`. Each record has exactly `id`, `kind`, `title`, `source`, `event_time`, `available_time`, and `text`. Kinds are `history`, `vitals`, `labs`, `radiology`, and `treatment`.

The loader excludes records whose availability time is later than `prediction_time`. It rejects duplicate IDs, availability before event time, and unexpected outcome fields. You must also enforce the lower ED-arrival bound during preparation and verify the disease-specific eligibility criteria. The generic loader cannot determine whether a date is a true release timestamp or a proxy; the preset's timing policy discloses the intended convention.

`--catalog` follows [examples/prognosis_catalog.json](../examples/prognosis_catalog.json). Each `t<number>` key maps to `tool`, `arguments`, and nonempty `record_ids`. The catalog must partition all eligible records exactly once. Valid tools are `PatientHistory`, `VitalSignsRequest`, `LabRequest`, `UrineRequest`, and `RadiologyRequest`. A selected laboratory tool should return the complete time series for that test/specimen in the permitted window.

Presets: `aki`, `ards_candidates`, `ich`, `tbi`, `sepsis`, `pneumonia`, `pulmonary_embolism`, and `polytrauma`. The disease premise is supplied context established at the cutoff; it is not a diagnosis inferred by the forecasting agent. The `iii` / `iv` switch selects MIMIC-III / MIMIC-IV availability-policy wording. This release does not include ICU-hourly or ICU24-hour runners.

For MIMIC remote inference, the original record loader requires a local account-confirmation object scoped to the endpoint and exact snapshot:

```json
{
  "basis": "user_account_confirmation",
  "zero_data_retention": true,
  "no_training": true,
  "no_human_review": true,
  "endpoint_sha256": "SHA256 of the configured full endpoint URL",
  "case_snapshots": {"local-case-id": "Case.fingerprint"}
}
```

Only supply these confirmations when they describe your actual provider/account conditions. The synthetic examples require no such object. Outcome labels must remain in a separate file and must never be added to a forecast input.
