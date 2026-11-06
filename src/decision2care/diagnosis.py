"""Decision2Care diagnostic work-up and hierarchical ICD-10-CM scoring."""
from dataclasses import asdict
import copy
import hashlib
import json
from .agent import RoutingAgent, AgentConfig
from .diagnosis_scorer import HierarchicalDiagnosisScorer, ScorerConfig
from .tools import PatientCase, ToolConfig

def diagnose(record, case_id, menus, tree, client, config):
    tools = config.get('tools', {})
    if tools.get('history', 'record') != 'record' or tools.get('ask_patient') or tools.get('llm_suggest'):
        raise ValueError('This release supports the primary record-history workflow without optional LLM helpers.')
    case = PatientCase(case_id, copy.deepcopy(record), menus, ToolConfig(**tools))
    workup = RoutingAgent(client, menus, AgentConfig(**config.get('agent', {}))).run(case)
    scorer = HierarchicalDiagnosisScorer(client, tree, ScorerConfig(**config.get('scorer', {})))
    # The primary diagnosis score is independent of the labelled reference disease.
    result = scorer.score(case.state())
    chart_fields = ['pe_seen','vitals_seen','labs_seen','micro_seen','imaging_seen',
                    'home_meds_seen','ed_meds_seen','unavailable','ecg_seen','echo_seen',
                    'log','notes','events']
    return case, dict(case_id=case_id, workup=workup, diagnosis=result,
                      record_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
                      chart={k:copy.deepcopy(getattr(case,k)) for k in chart_fields},
                      final_state=case.state(), tools_config=asdict(case.cfg),
                      trajectory=case.trajectory)

def restore_chart(record, saved, menus):
    """Restore the exact visible chart for treatment without another diagnostic call."""
    case = PatientCase(saved['case_id'], copy.deepcopy(record), menus, ToolConfig(**saved['tools_config']))
    fingerprint = hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    if fingerprint != saved['record_sha256']:
        raise ValueError('The supplied record differs from the diagnostic run.')
    for key,value in saved['chart'].items():
        if key in ('labs_seen','micro_seen'):
            value = {int(k):v for k,v in value.items()}
        setattr(case,key,copy.deepcopy(value))
    if case.state() != saved['final_state']:
        raise ValueError('Restored diagnostic chart differs from the saved state.')
    return case
