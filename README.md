# Decision2Care

**Auditable clinical support with decision agents.**

Decision2Care connects model scores over explicit alternatives to deterministic clinical actions. This first release contains the core inference workflows for **diagnosis, treatment selection, and mortality forecasting**, with synthetic examples that can run without a model account or patient data.

| Workflow | Core implementation | Runnable entrypoint |
| --- | --- | --- |
| Diagnosis | [Evidence retrieval](src/decision2care/agent.py), [record tools](src/decision2care/tools.py), [hierarchical ICD-10-CM scoring](src/decision2care/diagnosis_scorer.py) | `decision2care diagnosis` |
| Treatment | [Medication E v3 and procedure walk v2](src/decision2care/treatment.py) | `decision2care treatment` |
| Prognosis | [Adaptive retrieval and four-window forecast](src/decision2care/prognosis.py) | `decision2care prognosis` |

The clinical decision kernels come from the local research implementation. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records source hashes and packaging changes. Project paths, historical run replay, and input loading have been replaced by portable, explicit inputs. Jev is the external decision-model backend; its actual API identifier remains `jev-1.13.0`.

## Install

Use Python 3.10 or newer (Python 3.12 is the validated environment). From the repository root:

```bash
git clone https://github.com/peterhan91/Decision2Care.git
cd Decision2Care
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

For live diagnosis and treatment inference, also install the pinned model SDK:

```bash
python -m pip install -e '.[live]'
```

## Run the three synthetic examples

The bundled records and small action menus are entirely synthetic. The deterministic `--dry-run` backend verifies execution, logging, and input/output contracts. Its outputs are **protocol tests**, not clinical predictions or reproduced benchmark scores.

```bash
bash scripts/run_synthetic.sh
python -m pytest -q
```

The script writes three separate outputs under `private/synthetic/`. Choose a new output directory when repeating it:

```bash
bash scripts/run_synthetic.sh private/synthetic-second-run
```

## Reproduction workflow

Follow **prepare authorized inputs → configure the model → diagnose → select treatment / forecast prognosis → evaluate separately**. Details and exact input schemas are in [docs/INPUTS.md](docs/INPUTS.md) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

### 1. Prepare authorized inputs

The diagnostic runner accepts a normalized record, an action menu, and an observed ICD-10-CM tree. An exporter is included for locally prepared MIMIC-CDM resource files:

```bash
decision2care prepare-resources \
  --resources /path/to/authorized/resources \
  --icd-dictionary /path/to/mimic-iv/hosp/d_icd_diagnoses.csv.gz \
  --output private/resources
```

Treatment uses the completed diagnostic chart plus drug-purpose, dose, and procedure catalogs. Prognosis uses a separately prepared, time-gated record snapshot and a complete read-only retrieval catalog. Formats are illustrated in [examples/](examples/); replace the synthetic menus and records with authorized inputs for research runs.

This initial release does **not** reconstruct the complete paper cohorts or reproduce all reported figures. The original fixed cohort lists, clinical menus derived from MIMIC, clinician reviews, and patient-level experiment outputs are not distributed here. Synthetic examples establish that the three workflows execute; they do not establish diagnostic performance, treatment safety, calibration, or clinical benefit.

### 2. Configure live inference

Use [.env.example](.env.example) as a template and export the variables in your shell. The application does not automatically load a `.env` file. Live diagnosis and treatment require `TYPESAFE_API_KEY`; prognosis also requires the full `TYPESAFE_API_URL` endpoint and `TYPESAFE_ENABLED=true`. Model access and provider availability must be arranged separately. Changing the model is a new experimental condition.

### 3. Diagnosis

```bash
decision2care diagnosis \
  --record private/record.json \
  --menus private/resources/menus.json \
  --tree private/resources/icd_tree.json \
  --config configs/diagnosis.json \
  --case-id 1 --output private/diagnosis.json
```

The output saves each retrieval step, the ranked diagnosis codes, and the exact visible chart needed for the subsequent treatment stage. The same original record must be supplied to treatment; its fingerprint and restored chart are checked.

### 4. Treatment

```bash
decision2care treatment \
  --record private/record.json \
  --menus private/resources/menus.json \
  --diagnosis private/diagnosis.json \
  --treatment-menus private/treatment_menus.json \
  --output private/treatment.json
```

Medication selection retains purpose → regimen → dose → home-medication decisions and the original model-based allergy, kidney, bleeding, and duplication checks. Procedure selection retains the ICD-9-CM group walk and redundancy rules. These checks are not clinician-adjudicated safety or evidence of safe prescribing.

### 5. Prognosis

```bash
decision2care prognosis \
  --case private/prognosis_case.json \
  --catalog private/prognosis_catalog.json \
  --disease aki --dataset iv \
  --authorization private/provider_confirmation.json \
  --output private/prognosis.json
```

The first forecasting entrypoint is the ED-arrival +8-hour protocol, with eight disease presets listed in the [input guide](docs/INPUTS.md). Records must satisfy the stated availability rules and the disease premise must be established by that cutoff. The model chooses retrieval tools, then selects one of four survival windows. The returned distribution supports 7-, 30-, and 90-day mortality ranking scores; it is not a calibrated clinical probability. Outcome labels are stored and evaluated separately from model inputs.

## Data access and scope

Obtain MIMIC access from [PhysioNet](https://physionet.org/), complete the applicable credentialing and training, and follow the relevant data-use and model-service conditions. This repository contains code and synthetic examples, not MIMIC patient records, derived patient lists, provider keys, or model weights. Keep real records, prompts, traces, caches, and generated clinical outputs in your authorized environment.

## Attribution and license

Decision2Care is released under the [MIT license](LICENSE). Selected MIMIC-CDM helpers retain their [original license](src/decision2care/vendor/mimic_cdm/LICENSE.txt). Data and external models retain their own terms. The release structure follows the clear separation of implementation, preparation, runs, and evaluation in [MIRA](https://github.com/Dyke-F/MIRA); Decision2Care does not include MIRA's orchestration or FHIR server.

Relevant precedents: [MIMIC-CDM](https://doi.org/10.1038/s41591-024-03097-1), [MIRA](https://doi.org/10.1038/s41586-026-10675-5), and [CLEAR](https://doi.org/10.1038/s41551-026-01741-4). The associated manuscript is titled *Auditable Clinical Support with Decision Agents*.
