from pathlib import Path
import copy
import json
import math
import sys
from types import SimpleNamespace
import pytest

from decision2care.io import read_json,load_record,load_menus_json,load_tree_json
from decision2care.diagnosis import diagnose,restore_chart
from decision2care.offline import SyntheticClient
from decision2care.prognosis import PrognosisNavigator,mortality_scores,decode,FORECAST_WINDOWS
from decision2care.prognosis_data import Case
from decision2care.treatment import MedicationSelector,ProcedureSelector,THERAPEUTIC,PROPHYLAXIS
from decision2care.evaluation import auroc,evaluate

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def diagnostic_run():
    record=load_record(ROOT/'examples/diagnosis_record.json')
    menus=load_menus_json(ROOT/'examples/menus.json')
    case,saved=diagnose(record,1,menus,load_tree_json(ROOT/'examples/icd_tree.json'),SyntheticClient(),read_json(ROOT/'configs/diagnosis.json'))
    return record,menus,case,json.loads(json.dumps(saved))

def test_diagnosis_and_restored_chart(diagnostic_run):
    record,menus,case,saved=diagnostic_run
    assert saved['diagnosis']['top1']['code'] in {'K3580','K810'}
    assert len(saved['trajectory']) > 2
    assert {'Laboratory Tests','Imaging'} <= {t['tool'] for t in saved['trajectory']}
    restored=restore_chart(record,saved,menus)
    assert restored.state()==case.state()
    assert restored.do_home_meds()==case.do_home_meds()
    assert restored.state()==case.state()

def test_treatment_rejects_changed_record(diagnostic_run):
    record,menus,_,saved=diagnostic_run
    changed=copy.deepcopy(record);changed['Patient History']+=' Changed.'
    with pytest.raises(ValueError,match='differs'):
        restore_chart(changed,saved,menus)

def test_medication_and_procedure_kernels(diagnostic_run):
    record,menus,_,saved=diagnostic_run
    resources=read_json(ROOT/'examples/treatment_menus.json')
    c=dict(hadm_id=1,code='K3580',title='Unspecified acute appendicitis')
    med=MedicationSelector(restore_chart(record,saved,menus),SyntheticClient(),resources['purposes'],resources['doses']).ask(c)
    assert med['regimens']
    assert all(r['purpose'] != PROPHYLAXIS for r in med['regimens'])
    assert any(r['purpose']==PROPHYLAXIS for r in med['dropped'])
    assert med['safety']
    categories={'47':[['4709','Other appendectomy']],'54':[['5421','Laparoscopy']]}
    proc=ProcedureSelector(restore_chart(record,saved,menus),SyntheticClient(),categories).ask(c)
    assert any(r['icd_code']=='4709' for r in proc['procedures'])
    assert not any(r['icd_code']=='5421' for r in proc['procedures'])
    assert any('approach' in r['reason'] for r in proc['removed'])

def test_unsafe_medication_is_not_silently_accepted(diagnostic_run):
    class AllergyClient(SyntheticClient):
        def ask(self,state,questions,tag='misc',key_extra=None):
            a=super().ask(state,questions,tag,key_extra)
            for q in a:
                if q.startswith('allergy'):
                    a[q]['noul']=0.99
            return a
    record,menus,_,saved=diagnostic_run
    r=read_json(ROOT/'examples/treatment_menus.json')
    out=MedicationSelector(restore_chart(record,saved,menus),AllergyClient(),r['purposes'],r['doses']).ask(dict(hadm_id=1,code='K3580',title='Appendicitis'))
    assert any(x['unsafe'] for x in out['safety'])
    assert max(x['round'] for x in out['safety'])<=3
    assert not out['regimens']

def test_future_records_and_outcome_fields():
    data=read_json(ROOT/'examples/prognosis_case.json')
    later=copy.deepcopy(data['records'][0]);later.update(id='future',available_time='2020-01-01T09:00:00')
    data['records'].append(later)
    case=Case(data)
    assert case.excluded_count==1 and 'future' not in case.by_id
    data['outcome']='died'
    with pytest.raises(ValueError,match='Unexpected'):
        Case(data)

@pytest.mark.parametrize('disease',['aki','ards_candidates','ich','tbi','sepsis','pneumonia','pulmonary_embolism','polytrauma'])
@pytest.mark.parametrize('dataset',['iii','iv'])
def test_all_forecasting_presets(disease,dataset):
    case=Case(read_json(ROOT/'examples/prognosis_case.json'))
    nav=PrognosisNavigator(case,read_json(ROOT/'examples/prognosis_catalog.json'),SyntheticClient(),disease=disease,dataset=dataset)
    request=nav.selection_request()
    assert not {'outcome','outcomes','labels','alive_at','reference_diagnosis'} & set(request['state'])
    while not nav.stop_reason:
        nav.step()
    result=nav.assess()
    assert result['window'] in FORECAST_WINDOWS
    state=nav.final_request()['state']
    assert (decode(state) if 'record_groups' in state else state['records'])==nav.records()
    assert len(nav.called)>=1 and nav.decisions<=20

def test_catalog_requires_complete_disjoint_partition():
    case=Case(read_json(ROOT/'examples/prognosis_case.json'))
    catalog=read_json(ROOT/'examples/prognosis_catalog.json')
    catalog['t1']['record_ids'].append('r1')
    with pytest.raises(AssertionError):
        PrognosisNavigator(case,catalog,SyntheticClient())

def test_forecast_horizon_scores_and_auc_ties():
    dist=dict(zip(FORECAST_WINDOWS,[0.1,0.2,0.3,0.4]))
    scores=mortality_scores(dist)
    assert scores==pytest.approx({'7':0.1,'30':0.3,'90':0.6})
    assert auroc([0,1],[0.5,0.5])==0.5
    assert auroc([0,1,1,None],[0.1,0.8,0.9,0.0])==1
    assert auroc([1,1],[0.1,0.9]) is None
    with pytest.raises(ValueError):
        auroc([0,1],[0.1,float('nan')])

def test_pinned_backend_version(monkeypatch):
    class API:
        def __init__(self,**kwargs):pass
        def system_one(self,**kwargs):return SimpleNamespace(model='a-different-model')
        def close(self):pass
    monkeypatch.setitem(sys.modules,'typesafe_sdk',SimpleNamespace(TypeSafeClient=API,RetryPolicy=lambda **k:None))
    from decision2care.client import DecisionClient
    client=DecisionClient(model='jev-1.13.0',request_model='jev-1.13.0',api_key='synthetic-test-key',use_cache=False)
    with pytest.raises(RuntimeError,match='model version drift'):
        client.ask({}, {'q':dict(type='noul',instructions='Synthetic check')})
    client.close()

def test_evaluation_constructs_are_named():
    rows=[dict(reference_positive=True,selected=True),dict(reference_positive=False,selected=True)]
    a=evaluate('treatment',rows,'documented_care')
    assert a['reference']=='documented_care' and a['precision']==0.5 and a['recall']==1

def test_prepared_resource_export_and_icd_dictionary(tmp_path):
    import pandas as pd
    from decision2care.io import export_resources
    from decision2care.menus import PANELS
    resources=tmp_path/'resources';resources.mkdir()
    example=read_json(ROOT/'examples/menus.json')
    mapping=pd.DataFrame(example['lab_test_mapping'])
    mapping['fluid']='Blood'
    existing=set(mapping['label'])
    additional=[dict(label=name,corresponding_ids=[],fluid='Blood')
                for name in sorted({name for _,members in PANELS.values() for name in members}-existing)]
    mapping=pd.concat([mapping,pd.DataFrame(additional)],ignore_index=True)
    mapping['itemid']=[ids[0] if len(ids)==1 and label!='Complete Blood Count (CBC)' else None
                       for ids,label in zip(mapping['corresponding_ids'],mapping['label'])]
    mapping.to_pickle(resources/'lab_test_mapping.pkl')
    pd.DataFrame(example['lab_items'].values()).to_csv(resources/'d_labitems_min_1.csv',index=False)
    pd.DataFrame(columns=['itemid','label','count']).to_csv(resources/'micro_tests_universe.csv',index=False)
    pd.DataFrame([dict(modality='CT',region='Abdomen',n_reports=20)]).to_csv(resources/'imaging_universe.csv',index=False)
    rows=[dict(icd_code='K3580',icd_version=10,long_title='Unspecified acute appendicitis',n_admissions=20,n_primary=20)]
    pd.DataFrame(rows).to_csv(resources/'icd_diagnosis_universe.csv',index=False)
    dictionary=tmp_path/'d_icd_diagnoses.csv.gz'
    pd.DataFrame(rows).to_csv(dictionary,index=False,compression='gzip')
    with pytest.raises(FileNotFoundError,match='icd-dictionary'):
        export_resources(resources,tmp_path/'export')
    export_resources(resources,tmp_path/'export',icd_dictionary=dictionary)
    menus=load_menus_json(tmp_path/'export/menus.json')
    tree=load_tree_json(tmp_path/'export/icd_tree.json')
    assert menus.lab_items[50912].label=='Creatinine'
    assert tree.title('K3580')=='Unspecified acute appendicitis'
