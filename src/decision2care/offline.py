"""Deterministic protocol checks on synthetic inputs; never a predictive model."""
from types import SimpleNamespace
from .client import OfflineClient, _canonical
import hashlib

class SyntheticClient(OfflineClient):
    def __init__(self):
        super().__init__(noul_default=0.8)
        self.key_salt = None
        self.resolved_model = 'jev-1.13.0'
        self.config = SimpleNamespace(api_url='https://example.invalid/synthetic')

    def key(self, state, questions, key_extra=None):
        return hashlib.sha256(_canonical([state,questions,self.key_salt,key_extra]).encode()).hexdigest()

    def ask(self, state, questions, tag='misc', key_extra=None):
        answers = super().ask(state, questions, tag, key_extra)
        for key,question in questions.items():
            if question['type'] == 'noul' and key.startswith(('allergy','kidney','bleeding','duplicate')):
                answers[key]['noul'] = 0.1
            if question['type'] == 'choice':
                options = question['criteria']
                if key == 'next_action' and 'finish' in options:
                    selected = 'finish'
                else:
                    selected = next(iter(options))
                answers[key].update(choice=selected, confidence=1.0,
                                    probabilities={k:float(k==selected) for k in options})
        return answers

    def evaluate(self, request):
        return dict(model='jev-1.13.0', answers=self.ask(request['state'],request['questions']),
                    source='synthetic_protocol_test')
