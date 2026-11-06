"""TypeSafe /v1/systemone adapter. No network access until explicitly configured.

Contract checked against https://docs.typesafe.ai/api on 2026-09-24 and
https://docs.typesafe.ai/images on 2026-10-15. This is not an OpenAI-compatible
chat API. Image bytes are sent only when a caller puts them in `assets`; those
requests carry the experimental images header the API requires.
"""
from __future__ import annotations

import json
import math
import os
import time
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
from .input_language import require_english_input


class PrognosisError(RuntimeError):
    pass


class PrognosisNotConfigured(PrognosisError):
    pass


class PrognosisResponseError(PrognosisError):
    pass


@dataclass(frozen=True)
class PrognosisConfig:
    api_url: str = ''
    api_key: str = field(default='', repr=False)
    model: str = 'jev-latest'
    enabled: bool = False
    timeout_seconds: float = 20.
    min_confidence: float = .5

    def __post_init__(self):
        if not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 120:
            raise ValueError('TYPESAFE_TIMEOUT_SECONDS must be between 1 and 120')
        if not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError('TYPESAFE_MIN_CONFIDENCE must be between 0 and 1')
        if self.api_url:
            url = urlsplit(self.api_url)
            if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query or url.fragment:
                raise ValueError('TYPESAFE_API_URL must be a full HTTPS endpoint without credentials or query')

    @classmethod
    def from_env(cls):
        return cls(api_url=os.environ.get('TYPESAFE_API_URL', '').strip(),
                   api_key=os.environ.get('TYPESAFE_API_KEY', '').strip(),
                   model=os.environ.get('TYPESAFE_MODEL', 'jev-latest').strip() or 'jev-latest',
                   enabled=os.environ.get('TYPESAFE_ENABLED', 'false').lower() == 'true',
                   timeout_seconds=float(os.environ.get('TYPESAFE_TIMEOUT_SECONDS', '20')),
                   min_confidence=float(os.environ.get('TYPESAFE_MIN_CONFIDENCE', '.5')))

    @property
    def missing(self):
        return [name for name, value in [('TYPESAFE_API_URL', self.api_url),
                ('TYPESAFE_API_KEY', self.api_key), ('TYPESAFE_ENABLED=true', self.enabled)] if not value]

    def public_status(self):
        # Never return the key or endpoint (which may identify a private deployment).
        return dict(backend='jev', ready=not self.missing, model=self.model,
                    enabled=self.enabled, endpoint_configured=bool(self.api_url),
                    key_configured=bool(self.api_key), missing=self.missing,
                    min_confidence=self.min_confidence, live_verified=False)


def validate_choice(payload, criteria, question='next_action'):
    """Validate the wire contract before any viewport state can change."""
    try:
        if not isinstance(payload['model'], str) or not payload['model']:
            raise ValueError('missing model')
        answer = payload['answers'][question]
        choice = answer['choice']
        probabilities = answer['probabilities']
        if answer['type'] != 'choice' or choice not in criteria:
            raise ValueError('unknown action')
        if set(probabilities) != set(criteria):
            raise ValueError('distribution options do not match the submitted options')
        for v in [answer['confidence'], *probabilities.values()]:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError('invalid probability or confidence')
        # The live API rounds probabilities to two decimals.
        if not math.isclose(sum(probabilities.values()), 1., abs_tol=max(.001, len(criteria)*.0051)):
            raise ValueError('probabilities do not sum to one')
        if probabilities[choice] + 1e-6 < max(probabilities.values()):
            raise ValueError('choice is not a maximum-probability action')
        return dict(choice=choice, confidence=answer['confidence'],
                    probabilities=probabilities, model=payload['model'],
                    usage=payload.get('usage', {}))
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        error=PrognosisResponseError('Jev response failed action/probability validation; no action executed.')
        error.response={k:payload[k] for k in ('model','answers','usage') if isinstance(payload,dict) and k in payload}
        raise error from e


IMAGES_HEADER = {'TypeSafe-Danger-Zone-Images': 'true'}




def validate_named_choice(payload,criteria,question):
    try:
        assert isinstance(payload['model'],str) and payload['model']
        a=payload['answers'][question];p=a['probabilities']
        assert a['type']=='choice' and a['choice'] in criteria
        assert set(p)==set(criteria)
        assert all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1
                   for v in [a['confidence'],*p.values()])
        assert math.isclose(sum(p.values()),1.,abs_tol=max(.001,len(criteria)*.0051))
        return dict(choice=a['choice'],confidence=a['confidence'],probabilities=copy.deepcopy(p),
                    model=payload['model'],usage=payload.get('usage',{}))
    except (AssertionError,KeyError,TypeError,ValueError,AttributeError) as e:
        error=PrognosisResponseError('JEV named choice or response schema is invalid; no action executed.')
        error.response=copy.deepcopy(payload)
        raise error from e

def accept_named_choices(request,payload):
    return dict(model=payload.get('model'),source='typesafe_api',usage=payload.get('usage',{}),
        answers={q:validate_named_choice(payload,spec['criteria'],q) for q,spec in request['questions'].items()},
        choice_acceptance=dict(rule='Execute JEV choice exactly; probability rank is not an acceptance condition.',
                               raw_response=copy.deepcopy(payload)))

class PrognosisClient:
    """Preserve scientific micro-units; never alter the transmitted clinical text."""
    def __init__(self,cfg,transport=None):self.config,self.transport=cfg,transport
    def evaluate(self,request):
        import httpx
        assert not self.config.missing
        try:validate_language(request)
        except ValueError as e:raise PrognosisError(str(e)) from None
        start=time.perf_counter()
        try:
            with httpx.Client(timeout=self.config.timeout_seconds,transport=self.transport,follow_redirects=False) as client:
                response=client.post(self.config.api_url,json=request,headers={'Authorization':'Bearer '+self.config.api_key})
        except httpx.HTTPError:raise PrognosisError('Jev connection failed or timed out; no action executed.') from None
        if response.status_code!=200:
            error=PrognosisError(f'Jev returned HTTP {response.status_code}; no action executed.')
            # Keep provider diagnostics in the private attempt log, without request headers.
            try:error.response=response.json()
            except ValueError:error.response={'status':response.status_code,'body':response.text[:2000]}
            raise error
        try:payload=response.json()
        except ValueError:raise PrognosisError('Jev returned invalid JSON; no action executed.') from None
        result=accept_named_choices(request,payload);result['api_ms']=(time.perf_counter()-start)*1000;return result

def validate_language(request):
    from .input_language import require_english_input
    def validation_copy(x):
        if isinstance(x,str):return x.replace('µg','microgram').replace('μg','microgram').replace('µL','microlitre').replace('μL','microlitre').replace('µIU','microinternationalunit').replace('μIU','microinternationalunit')
        if isinstance(x,list):return [validation_copy(v) for v in x]
        if isinstance(x,dict):return {k:validation_copy(v) for k,v in x.items()}
        return x
    require_english_input(validation_copy(request))
