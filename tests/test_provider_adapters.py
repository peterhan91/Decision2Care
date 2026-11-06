"""Exercise production response handlers without contacting a model provider."""
from pathlib import Path
import json
import httpx
import pytest

from decision2care.diagnosis import diagnose, restore_chart
from decision2care.io import read_json,load_record,load_menus_json,load_tree_json
from decision2care.offline import SyntheticClient
from decision2care.prognosis import PrognosisNavigator
from decision2care.prognosis_data import Case
from decision2care.prognosis_client import PrognosisClient,PrognosisConfig,PrognosisResponseError
from decision2care.treatment import MedicationSelector,ProcedureSelector

ROOT=Path(__file__).resolve().parents[1]

def provider_payload(request, scripted):
    return dict(model='jev-1.13.0',usage=dict(input_tokens=10,output_tokens=0),
                answers=scripted.ask(request['state'],request['questions']))

def test_prognosis_real_http_adapter_end_to_end():
    requests=[]
    scripted=SyntheticClient()
    def respond(request):
        body=json.loads(request.content)
        requests.append(body)
        return httpx.Response(200,json=provider_payload(body,scripted))
    cfg=PrognosisConfig(api_url='https://example.invalid/v1/systemone',
        api_key='synthetic-test-key',enabled=True,model='jev-1.13.0')
    client=PrognosisClient(cfg,transport=httpx.MockTransport(respond))
    nav=PrognosisNavigator(Case(read_json(ROOT/'examples/prognosis_case.json')),
        read_json(ROOT/'examples/prognosis_catalog.json'),client)
    while not nav.stop_reason:
        nav.step()
    result=nav.assess()
    assert result['model']=='jev-1.13.0'
    assert len(requests)==nav.decisions+1
    assert nav.audit[-1]['response']['choice_acceptance']['raw_response']['answers']

def test_prognosis_real_adapter_rejects_malformed_response():
    request=dict(model='jev-1.13.0',state={'task':'Synthetic check'},questions={
        'next_action':dict(type='choice',instructions='Select an option.',criteria={'read':'Read','finish':'Finish'})})
    def respond(_):
        return httpx.Response(200,json=dict(model='jev-1.13.0',answers={
            'next_action':dict(type='choice',choice='read',confidence=0.9,probabilities={'read':0.9})}))
    client=PrognosisClient(PrognosisConfig(api_url='https://example.invalid/v1/systemone',
        api_key='synthetic-test-key',enabled=True),transport=httpx.MockTransport(respond))
    with pytest.raises(PrognosisResponseError):
        client.evaluate(request)

def test_diagnosis_and_treatment_real_sdk_adapters():
    sdk=pytest.importorskip('typesafe_sdk')
    httpx2=pytest.importorskip('httpx2')
    from decision2care.client import DecisionClient
    requests=[]
    scripted=SyntheticClient()
    def respond(request):
        body=json.loads(request.content)
        requests.append(body)
        return httpx2.Response(200,json=provider_payload(body,scripted))
    client=DecisionClient(api_key='synthetic-test-key',model='jev-1.13.0',
                          request_model='jev-1.13.0',use_cache=False)
    client._client.close()
    client._client=sdk.TypeSafeClient(api_key='synthetic-test-key',model='jev-1.13.0',
                                     transport=httpx2.MockTransport(respond))
    try:
        record=load_record(ROOT/'examples/diagnosis_record.json')
        menus=load_menus_json(ROOT/'examples/menus.json')
        _,saved=diagnose(record,1,menus,load_tree_json(ROOT/'examples/icd_tree.json'),
                         client,read_json(ROOT/'configs/diagnosis.json'))
        saved=json.loads(json.dumps(saved))
        assert saved['diagnosis']['top1']
        resources=read_json(ROOT/'examples/treatment_menus.json')
        dx=saved['diagnosis']['top1'];case=dict(hadm_id=1,code=dx['code'],title=dx['title'])
        med=MedicationSelector(restore_chart(record,saved,menus),client,
            resources['purposes'],resources['doses']).ask(case)
        proc=ProcedureSelector(restore_chart(record,saved,menus),client,
            resources['procedure_categories']).ask(case)
        assert med['regimens'] and proc['procedures']
        assert client.resolved_model=='jev-1.13.0' and len(requests)>3
    finally:
        client.close()
