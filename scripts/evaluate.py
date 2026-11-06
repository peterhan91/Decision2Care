"""Evaluate locally prepared predictions against separate reference labels."""
import argparse
from decision2care.io import read_json,write_json
from decision2care.evaluation import evaluate

p=argparse.ArgumentParser()
p.add_argument('--task',required=True,choices=['diagnosis','treatment','prognosis'])
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--treatment-reference',default='indication_rubric',choices=['indication_rubric','documented_care'])
a=p.parse_args()
write_json(a.output,evaluate(a.task,read_json(a.input),a.treatment_reference))
