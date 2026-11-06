"""Text-only, time-gated PE evidence. Outcome data has a separate file/schema."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

KINDS = {
    'radiology': 'Radiology reports', 'vitals': 'Respiratory and circulatory observations',
    'labs': 'Laboratory results', 'history': 'Previously documented history',
    'treatment': 'Treatments already administered',
}
ID = re.compile(r'^[A-Za-z0-9_-]{1,100}$')


def timestamp(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('A nonempty ISO timestamp is required.')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def strict_keys(data, allowed, required):
    if not isinstance(data, dict) or set(data) - set(allowed) or set(required) - set(data):
        raise ValueError('Unexpected or missing case fields; outcomes must be stored separately.')


@dataclass(frozen=True)
class Record:
    id: str
    kind: str
    title: str
    source: str
    event_time: str
    available_time: str
    text: str

    @classmethod
    def parse(cls, data):
        names = tuple(cls.__dataclass_fields__)
        strict_keys(data, names, names)
        if not all(isinstance(data[k], str) and data[k].strip() for k in names):
            raise ValueError('Record fields must be nonempty strings.')
        if not ID.fullmatch(data['id']) or data['kind'] not in KINDS:
            raise ValueError('Invalid record identifier or kind.')
        if len(data['title']) > 240 or len(data['source']) > 240 or len(data['text']) > 200000:
            raise ValueError('Record exceeds supported size.')
        if timestamp(data['event_time']) > timestamp(data['available_time']):
            raise ValueError('A record cannot be available before its event time.')
        return cls(**data)

    def metadata(self):
        return {k: getattr(self, k) for k in ('id', 'kind', 'title', 'source', 'event_time', 'available_time')}


class Case:
    def __init__(self, data):
        strict_keys(data, ('schema_version', 'id', 'title', 'provenance', 'prediction_time', 'records', 'data_status'),
                    ('schema_version', 'id', 'title', 'provenance', 'prediction_time', 'records'))
        if data['schema_version'] != 1 or not isinstance(data['id'], str) or not ID.fullmatch(data['id']):
            raise ValueError('Invalid case schema or identifier.')
        if data['provenance'] not in ('synthetic', 'mimic'):
            raise ValueError('Unknown data provenance.')
        if not isinstance(data['title'], str) or not 1 <= len(data['title']) <= 240:
            raise ValueError('Invalid case title.')
        if not isinstance(data['records'], list) or len(data['records']) > 100000:
            raise ValueError('Invalid records.')
        self.id, self.title = data['id'], data['title']
        self.provenance = data['provenance']
        self.data_status = data.get('data_status', 'synthetic' if self.provenance == 'synthetic' else 'linked')
        if self.data_status not in ('synthetic', 'reports_only', 'linked'):
            raise ValueError('Unknown case data status.')
        self.prediction_time = data['prediction_time']
        cutoff = timestamp(self.prediction_time)
        parsed = [Record.parse(r) for r in data['records']]
        if len({r.id for r in parsed}) != len(parsed):
            raise ValueError('Duplicate record IDs.')
        self.records = sorted((r for r in parsed if timestamp(r.available_time) <= cutoff),
                              key=lambda r: (timestamp(r.available_time), r.id))
        self.excluded_count = len(parsed) - len(self.records)
        self.by_id = {r.id: r for r in self.records}
        # Bind saved sessions to the permitted snapshot, never to outcome data.
        visible = [dict(**r.metadata(), text=r.text) for r in self.records]
        self.fingerprint = hashlib.sha256(json.dumps([self.prediction_time, visible], sort_keys=True).encode()).hexdigest()

    def public(self):
        return dict(id=self.id, title=self.title, provenance=self.provenance, data_status=self.data_status,
                    prediction_time=self.prediction_time, record_count=len(self.records),
                    excluded_future_records=self.excluded_count,
                    categories={k: sum(r.kind == k for r in self.records) for k in KINDS})

    def require_remote_access(self, authorization=None, api_url=None):
        if self.provenance == 'synthetic':
            return
        # An account confirmation is scoped to exact case snapshots and the
        # configured endpoint. No global bypass or provenance rewriting.
        if isinstance(authorization, dict) and isinstance(api_url, str) and api_url:
            confirmed = all(authorization.get(k) is True for k in
                            ('zero_data_retention', 'no_training', 'no_human_review'))
            endpoint = hashlib.sha256(api_url.encode()).hexdigest()
            if (confirmed and authorization.get('basis') == 'user_account_confirmation'
                    and authorization.get('endpoint_sha256') == endpoint
                    and authorization.get('case_snapshots', {}).get(self.id) == self.fingerprint):
                return
        raise PermissionError('MIMIC remote inference requires an account confirmation scoped to this endpoint and case snapshot. Local browsing and request preview remain available.')


def load_case(path):
    return Case(json.loads(Path(path).read_text()))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.chmod(0o600)
    temp.replace(path)


def load_env(path):
    """Use existing local configuration without logging its values or running a shell."""
    import os
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('\"').strip("'"))
