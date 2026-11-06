#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
destination="${1:-private/synthetic}"
python -m decision2care diagnosis --dry-run \
  --record examples/diagnosis_record.json --menus examples/menus.json \
  --tree examples/icd_tree.json --config configs/diagnosis.json \
  --case-id 1 --output "$destination/diagnosis.json"
python -m decision2care treatment --dry-run \
  --record examples/diagnosis_record.json --menus examples/menus.json \
  --diagnosis "$destination/diagnosis.json" \
  --treatment-menus examples/treatment_menus.json --output "$destination/treatment.json"
python -m decision2care prognosis --dry-run \
  --case examples/prognosis_case.json --catalog examples/prognosis_catalog.json \
  --disease aki --dataset iv --output "$destination/prognosis.json"
