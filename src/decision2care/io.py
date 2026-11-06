"""JSON interchange for prepared records, action menus, and ICD trees."""
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
from .menus import Menus, LabItem
from .icd_hierarchy import ICD10Tree, Node

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    path.chmod(0o600)

def load_record(path):
    value = read_json(path)
    for field in ['Laboratory Tests', 'Microbiology', 'Laboratory Tests As Built', 'Microbiology As Built']:
        if isinstance(value.get(field), dict):
            value[field] = {int(k): v for k, v in value[field].items()}
    return value

def load_menus_json(path):
    value = read_json(path)
    return Menus(
        lab_test_mapping=pd.DataFrame(value['lab_test_mapping']),
        lab_items={int(k): LabItem(**v) for k, v in value['lab_items'].items()},
        lab_groups=value['lab_groups'], panels=value['panels'],
        imaging={k: tuple(v) for k,v in value['imaging'].items()},
        micro_items={int(k): LabItem(**v) for k,v in value.get('micro_items',{}).items()},
        item_class={int(k): tuple(v) for k,v in value.get('item_class',{}).items()})

def load_tree_json(path):
    value = read_json(path)
    tree = ICD10Tree.__new__(ICD10Tree)
    tree.nodes = {k: Node(**v) for k,v in value['nodes'].items()}
    tree.chapters = value['chapters']
    tree._block_ranges = [tuple(v) for v in value.get('block_ranges', [])]
    tree.icd9_to_10 = value.get('icd9_to_10', {})
    tree.icd9_titles = value.get('icd9_titles', {})
    tree.n_unplaced = 0
    if not tree.chapters or any(k not in tree.nodes for k in tree.chapters):
        raise ValueError('Invalid ICD tree root.')
    for key,node in tree.nodes.items():
        if key != node.code or any(c not in tree.nodes for c in node.children):
            raise ValueError('Invalid ICD tree node or child.')
        if node.parent is not None and node.parent not in tree.nodes:
            raise ValueError('Invalid ICD tree parent.')
    return tree

def export_resources(resources_dir, destination, extra_imaging_dirs=None, icd_dictionary=None):
    """Export an authorized, locally built MIMIC-CDM resource directory to JSON.

    The original lab_test_mapping.pkl is a trusted local Python pickle. Never
    use a pickle downloaded from an untrusted source.
    """
    from .menus import load_menus
    resources_dir, destination = Path(resources_dir), Path(destination)
    dictionary = Path(icd_dictionary) if icd_dictionary else resources_dir/'d_icd_diagnoses.csv'
    if not dictionary.is_file():
        raise FileNotFoundError('Supply the authorized d_icd_diagnoses.csv or .csv.gz using --icd-dictionary; it is not included in this code release.')
    menus = load_menus(resources_dir, extra_imaging_dirs=extra_imaging_dirs)
    tree = ICD10Tree(resources_dir/'icd_diagnosis_universe.csv', dictionary)
    write_json(destination/'menus.json', dict(
        lab_test_mapping=json.loads(menus.lab_test_mapping.to_json(orient='records')),
        lab_items={k:asdict(v) for k,v in menus.lab_items.items()},
        lab_groups=menus.lab_groups, panels=menus.panels, imaging=menus.imaging,
        micro_items={k:asdict(v) for k,v in menus.micro_items.items()}, item_class=menus.item_class))
    write_json(destination/'icd_tree.json', dict(
        nodes={k:asdict(v) for k,v in tree.nodes.items()}, chapters=tree.chapters,
        block_ranges=tree._block_ranges, icd9_to_10=tree.icd9_to_10, icd9_titles=tree.icd9_titles))
