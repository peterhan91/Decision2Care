"""Command-line entrypoints for the three Decision2Care workflows."""
import argparse
from pathlib import Path
import json

def main(argv=None):
    parser = argparse.ArgumentParser(prog='decision2care')
    sub = parser.add_subparsers(dest='task', required=True)
    for task in ['diagnosis','treatment','prognosis']:
        p = sub.add_parser(task)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--dry-run', action='store_true', help='Synthetic protocol test; no model or network calls.')
        p.add_argument('--authorization', type=Path, help='Local provider/account confirmation for MIMIC prognosis inputs.')
        if task != 'prognosis':
            p.add_argument('--record', type=Path, required=True)
            p.add_argument('--menus', type=Path, required=True)
        if task == 'diagnosis':
            p.add_argument('--tree', type=Path, required=True)
            p.add_argument('--config', type=Path, default=Path('configs/diagnosis.json'))
            p.add_argument('--case-id', type=int, default=1)
        elif task == 'treatment':
            p.add_argument('--diagnosis', type=Path, required=True)
            p.add_argument('--treatment-menus', type=Path, required=True)
        else:
            p.add_argument('--case', type=Path, required=True)
            p.add_argument('--catalog', type=Path, required=True)
            p.add_argument('--disease', default='aki', choices=['aki','sepsis','tbi','pneumonia','ich','polytrauma','ards_candidates','pulmonary_embolism'])
            p.add_argument('--dataset', default='iv', choices=['iii','iv'])
            p.add_argument('--max-calls', type=int, default=20)
    p = sub.add_parser('prepare-resources')
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--extra-imaging-dir', type=Path, action='append')
    p.add_argument('--icd-dictionary', type=Path, help='Authorized d_icd_diagnoses.csv or .csv.gz if stored outside the prepared resource directory.')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from .io import read_json, write_json
    if args.task == 'prepare-resources':
        from .io import export_resources
        export_resources(args.resources,args.output,args.extra_imaging_dir,args.icd_dictionary)
        return
    if args.output.exists():
        parser.error('Output already exists; choose a new path to preserve previous results.')
    from .offline import SyntheticClient
    client = None
    if args.dry_run:
        client = SyntheticClient()
        if args.task == 'prognosis':
            if read_json(args.case).get('provenance') != 'synthetic':
                parser.error('--dry-run requires a synthetic case.')
        elif read_json(args.record).get('Decision2Care provenance') != 'synthetic':
            parser.error('--dry-run requires a synthetic record.')
    try:
        if args.task == 'prognosis':
            from .prognosis import PrognosisNavigator, mortality_scores
            from .prognosis_data import Case
            if client is None:
                from .prognosis_client import PrognosisConfig, PrognosisClient
                from dataclasses import replace
                client = PrognosisClient(replace(PrognosisConfig.from_env(), model='jev-1.13.0'))
            nav = PrognosisNavigator(Case(read_json(args.case)),read_json(args.catalog),client,
                disease=args.disease,dataset=args.dataset,max_calls=args.max_calls,
                authorization=read_json(args.authorization) if args.authorization else None)
            while not nav.stop_reason:
                nav.step()
            nav.assess()
            result = nav.export()
            result['mortality_scores'] = mortality_scores(result['outcome']['raw_choice_distribution'])
        else:
            from .io import load_record, load_menus_json
            if client is None:
                from .client import DecisionClient
                client = DecisionClient(model='jev-1.13.0',request_model='jev-1.13.0')
            menus, record = load_menus_json(args.menus), load_record(args.record)
            if args.task == 'diagnosis':
                from .diagnosis import diagnose
                from .io import load_tree_json
                _,result = diagnose(record,args.case_id,menus,load_tree_json(args.tree),client,read_json(args.config))
            else:
                from .diagnosis import restore_chart
                from .treatment import MedicationSelector, ProcedureSelector
                saved, resources = read_json(args.diagnosis), read_json(args.treatment_menus)
                case = restore_chart(record,saved,menus)
                procedure_case = restore_chart(record,saved,menus)
                diagnosis = saved['diagnosis']['top1']
                if not diagnosis:
                    raise ValueError('Treatment needs a completed principal diagnosis.')
                c = dict(hadm_id=saved['case_id'],code=diagnosis['code'],title=diagnosis['title'])
                result = dict(case_id=saved['case_id'],
                    medications=MedicationSelector(case,client,resources['purposes'],resources['doses']).ask(c),
                    procedures=ProcedureSelector(procedure_case,client,resources['procedure_categories']).ask(c))
        result.update(project='Decision2Care',synthetic_protocol_test=args.dry_run)
        if args.dry_run:
            result['notice'] = 'Deterministic synthetic protocol output; not a model prediction or benchmark result.'
        write_json(args.output,result)
        print(json.dumps(dict(task=args.task,output=str(args.output),synthetic_protocol_test=args.dry_run)))
    finally:
        if client is not None and hasattr(client,'close'):
            client.close()
