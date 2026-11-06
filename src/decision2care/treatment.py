"""Decision2Care medication and procedure selection kernels."""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from .client import choice, noul

DOSE_Q = "Given the principal diagnosis and the findings documented in the state, which regimen of {drug} is indicated for this patient now?"

OTHER_DOSE = {"other": "Another dose or regimen."}

PURPOSE_Q = "Given the principal diagnosis and the findings documented in the state, is {purpose} indicated for this patient now, at the start of treatment?"

REGIMEN_Q = "Given the principal diagnosis and the findings documented in the state, which regimen of {purpose} is indicated for this patient now?"

NO_REGIMEN = {"none": "None of these regimens."}

HOME_Q = "Given the principal diagnosis and the findings documented in the state, what should be done now with the home medication {med}?"

HOME_OPTIONS = {"continue": "Continue it.", "hold": "Hold it for now.", "stop": "Stop it."}

THERAPEUTIC = ("therapeutic anticoagulation (treatment dose: a heparin infusion, treatment-dose low-molecular-weight heparin, a direct oral "
               "anticoagulant or warfarin)")

PROPHYLAXIS = "pharmacological prophylaxis of venous thromboembolism (prophylactic dose: subcutaneous heparin or low-molecular-weight heparin)"

SALT_V3 = "treatment_v3e_2026-10-18"  # 'e': this design (the runs before it kept their own salts: none, b, c, d; d also ran once on the 546)

SAFETY_QS = {
    "allergy": "Does the allergy list in the state name the drug of the order '{order}', a brand name of it, or a drug of the same class that cross-reacts with it?",
    "kidney": "The kidney function in the state is {kidney}. Is the order '{order}' a dose too high for this kidney function, or a drug to avoid at it?",
    "bleeding": ("Is the order '{order}' contraindicated for this patient now by active bleeding, an intracranial hemorrhage, a bleeding disorder "
                 "(such as von Willebrand disease) or a coagulation test far above range (such as an INR above 3 or a PTT above 100 seconds) "
                 "documented in the state?"),
    "duplicate": "Does the home medication of the order '{order}' duplicate a new order of the treatment plan (the same drug, or another drug of its class)?",
}

MAX_ROUNDS = 3  # safety rounds: the first plan, then up to two new choices of the purposes found unsafe

BLEEDING_PURPOSES = (THERAPEUTIC, PROPHYLAXIS, "antiplatelet therapy", "thrombolysis")

BLEEDING_DRUG = re.compile(r"NSAID|(?<!Non-)Salicylate|Platelet Aggregation|Anticoagul|Heparin|Coumarin|Factor Xa|Thrombin Inhibitor|Thrombolytic", re.I)

RECHOOSE_Q = ("Given the principal diagnosis and the findings documented in the state, which regimen of {purpose} is indicated for this patient now, "
              "if any? The orders the state lists as unsafe for this patient, with the reason, are not among the options.")

UNSAFE = 0.5  # an order is unsafe at this probability or more

HOME_ANTICOAGULANT = re.compile(r"apixaban|rivaroxaban|dabigatran|edoxaban|eliquis|xarelto|pradaxa|savaysa|warfarin|coumadin|jantoven", re.I)

PROC_Q = ("Given the principal diagnosis and the findings documented in the state, which of these procedures is indicated for this patient "
          "during this admission, if any?")

NONE = {"none": "None of these procedures is indicated for this patient during this admission."}

CATEGORY_Q = ("Given the principal diagnosis and the findings documented in the state, is a procedure of this group indicated for this "
              "patient during this admission? The group holds procedures such as: {ex}.")

WALK_EXAMPLES, WALK_CANDIDATES, MIN_CANDIDATE = 8, 3, 0.05

SALT_WALK2 = "walk_v2_2026-10-18"

INDICATED_Q = "Given the principal diagnosis and the findings documented in the state, is the procedure '{title}' indicated for this patient during this admission?"

ADDITION_Q = ("Given the principal diagnosis and the findings documented in the state, the treatment plan already includes: {done}. Is the procedure "
              "'{title}' needed in addition, as a separate procedure?")

def approach(code: str) -> bool:
    """ICD-9-CM codes of an approach or an adjunct, coded within the operation they serve: laparoscopy (54.21), exploratory and other
    laparotomy (54.11, 54.19), robotic assistance (17.4x)."""
    return code in ("5421", "5411", "5419") or code.startswith("174")

def in_abdomen(code: str) -> bool:
    """An operation in the abdomen or pelvis (ICD-9-CM 42-59, 65-71, laparoscopic hernia repair and large-bowel resection 17.1-17.3)."""
    return (42 <= int(code[:2]) <= 59 or 65 <= int(code[:2]) <= 71 or code[:3] in ("171", "172", "173")) and not approach(code)

NO_KNOWN = re.compile(r"no known|no allergies|\bnkda\b|none known", re.I)

def allergy_list(rec: Dict[str, Any]) -> Optional[List[str]]:
    """The allergy list of the record's discharge note: [] when it records none, None when the line is missing or masked."""
    m = re.search(r"Allergies:\s*(.*?)(?:\n\s*\n|Attending:)", str(rec.get("Discharge") or ""), re.S)
    if not m:
        return None
    text = " ".join(m.group(1).split())
    if NO_KNOWN.search(text):
        return []
    items = [x.strip() for x in text.split(" / ")]
    return [x for x in items if x and not re.fullmatch(r"_+", x)] or None

def creatinine(rec: Dict[str, Any]) -> Optional[float]:
    """The record's first creatinine (MIMIC-IV item 50912, mg/dL)."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str((rec.get("Laboratory Tests") or {}).get(50912, "")))
    return float(m.group(1)) if m else None

def egfr(rec: Dict[str, Any]) -> Optional[float]:
    """eGFR by CKD-EPI 2021 (mL/min/1.73 m2) from the first creatinine, age and sex."""
    scr = creatinine(rec)
    if scr is None or not rec.get("Age"):
        return None
    female = str(rec.get("Sex", "")).lower().startswith("f")
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    return 142 * min(scr / k, 1) ** a * max(scr / k, 1) ** -1.200 * 0.9938 ** float(rec["Age"]) * (1.012 if female else 1)

def safety_facts(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Allergy and kidney-function fields added to the treatment state."""
    al, scr, e = allergy_list(rec), creatinine(rec), egfr(rec)
    return {"allergies": al if al else ("no known drug allergies" if al == [] else "not recorded"),
            "kidney_function": ({"creatinine": f"{scr:g} mg/dL (first value of the admission)", "eGFR": f"{e:.0f} mL/min/1.73 m2 (CKD-EPI 2021)"}
                                if e is not None else "not measured")}

def dotted(code: str) -> str:
    return code[:3] + "." + code[3:] if len(code) > 3 else code


class PreparedCaseAdapter:
    """Inject the completed diagnostic chart instead of replaying a local run archive."""
    def __init__(self, case, client):
        self.case, self.jev = case, client
        self.cohort = {case.hadm_id: case.rec}
    def rebuild(self, hadm_id, same='actions'):
        if hadm_id != self.case.hadm_id:
            raise ValueError('Case identifier does not match the supplied diagnostic chart.')
        return self.case, None


class MedicationSelector(PreparedCaseAdapter):
    """Medication selection on the completed diagnostic chart. Each request's stage and cache key are recorded in `trail`."""

    def __init__(self, case, client, purposes, doses):
        super().__init__(case, client)
        self.purposes = {k: v for k, v in purposes.items() if not k.startswith('diagnostic')}
        self.doses = doses
        if not all((k in self.purposes for k in (THERAPEUTIC, PROPHYLAXIS))):
            raise ValueError('The treatment menu must distinguish therapeutic anticoagulation and prophylaxis.')
        self.jev.key_salt = '|'.join((x for x in (self.jev.key_salt, SALT_V3) if x))

    def _ask(self, trail: List[dict], stage: str, state: Dict[str, Any], qs: Dict[str, Any]) -> Dict[str, Any]:
        if not qs:
            return {}
        trail.append({'stage': stage, 'key': self.jev.key(state, qs), 'questions': len(qs)})
        return self.jev.ask(state, qs, tag='treatment_e3')

    def _regimens(self, trail, stage, state, purposes: List[str], unsafe: set=frozenset(), q: str=REGIMEN_Q) -> List[Dict[str, Any]]:
        """One regimen for each purpose (a Choice); in a new choice (`unsafe` drugs left out), 'none' is an option."""
        menus = {p: [(i, r) for i, r in enumerate(self.purposes[p]) if not set(r['drugs']) & unsafe] for p in purposes}
        qs = {f'g{j:02d}': choice(q.format(purpose=p), {**{f'r{i}': r['shown'] for i, r in menus[p]}, **(NO_REGIMEN if unsafe else {})}) for j, p in enumerate(purposes) if menus[p]}
        a = self._ask(trail, stage, state, qs)
        picks = []
        for j, p in enumerate(purposes):
            ans = a.get(f'g{j:02d}')
            if ans and ans['choice'] != 'none':
                r = self.purposes[p][int(ans['choice'][1:])]
                picks.append({'purpose': p, 'items': r['items'], 'drugs': r['drugs'], 'shown': r['shown'], 'p': round(ans['probabilities'][ans['choice']], 4), **({'p_none': round(ans['probabilities'].get('none', 0.0), 4)} if unsafe else {})})
        return picks

    def _doses(self, trail, stage, state, picks: List[Dict[str, Any]]) -> None:
        drugs = [(j, d) for j, pk in enumerate(picks) for d in pk['drugs'] if self.options(pk['purpose'], d) and d not in (pk.get('doses') or {})]
        qs = {f'x{n:03d}': choice(DOSE_Q.format(drug=d), {**{f'g{i}': o['shown'] for i, o in enumerate(self.options(picks[j]['purpose'], d))}, **OTHER_DOSE}) for n, (j, d) in enumerate(drugs)}
        a = self._ask(trail, stage, state, qs)
        for n, (j, d) in enumerate(drugs):
            x = a[f'x{n:03d}']
            picks[j].setdefault('doses', {})[d] = 'another' if x['choice'] == 'other' else self.options(picks[j]['purpose'], d)[int(x['choice'][1:])]

    @staticmethod
    def order_line(pk: Dict[str, Any], d: str) -> str:
        names = str(pk['shown']).split(' + ')
        name = names[pk['drugs'].index(d)] if len(names) == len(pk['drugs']) else d
        dose = (pk.get('doses') or {}).get(d)
        text = dose['shown'] if isinstance(dose, dict) else 'dose to be chosen by the treating team'
        return f"{name}, {text}, for {pk['purpose']}"

    @staticmethod
    def home_line(m: Dict[str, Any]) -> str:
        return f"{m['med']} ({m['class']}), a home medication to be continued"

    def _check(self, trail, stage, state, picks, home, which) -> Dict[tuple, Dict[str, float]]:
        """The four safety questions for the orders `which` ((pick index, drug) or ('home', index)), with the whole plan in the state."""
        plan = [self.order_line(pk, d) for pk in picks for d in pk['drugs']] + [self.home_line(m) for m in home if m['choice'] == 'continue']
        lines = {w: self.order_line(picks[w[0]], w[1]) if w[0] != 'home' else self.home_line(home[w[1]]) for w in which}
        kf = state.get('kidney_function')
        kidney = f"an eGFR of {kf['eGFR']} (creatinine {kf['creatinine']})" if isinstance(kf, dict) else 'not measured'

        def bleeds(w) -> bool:
            if w[0] == 'home':
                return bool(BLEEDING_DRUG.search(home[w[1]]['class']))
            pk = picks[w[0]]
            return pk['purpose'] in BLEEDING_PURPOSES or bool(BLEEDING_DRUG.search(pk['items'][pk['drugs'].index(w[1])].split('|')[0]))
        asked = {w: [k for k in SAFETY_QS if not (k == 'duplicate' and w[0] != 'home') and (not (k == 'bleeding' and (not bleeds(w))))] for w in which}
        qs = {f'{k}{n:03d}': noul(SAFETY_QS[k].format(order=lines[w], kidney=kidney)) for n, w in enumerate(which) for k in asked[w]}
        a = self._ask(trail, stage, {**state, 'treatment_plan': plan}, qs)
        return {w: {k: round(a[f'{k}{n:03d}']['noul'], 4) for k in asked[w]} for n, w in enumerate(which)}

    @staticmethod
    def verdict(line: str, rnd: int, p: Dict[str, float]) -> Dict[str, Any]:
        reasons = [k for k, v in p.items() if v >= UNSAFE]
        return {'round': rnd, 'order': line, 'p_unsafe': max(p.values()), 'p': p, 'reasons': reasons, 'unsafe': bool(reasons)}

    def ask(self, c: Dict[str, Any]) -> Dict[str, Any]:
        h = c['hadm_id']
        case, _ = self.rebuild(h, same='actions')
        dx = f"{dotted(c['code'])} {c['title']}"
        facts = safety_facts(self.cohort[h])
        state = {**case.state(), 'principal_diagnosis': dx, **facts}
        trail: List[dict] = []
        names = sorted(self.purposes)
        a1 = self._ask(trail, 'purposes', state, {f'u{i:02d}': noul(PURPOSE_Q.format(purpose=p)) for i, p in enumerate(names)})
        yes = [p for i, p in enumerate(names) if a1[f'u{i:02d}']['noul'] >= 0.5]
        picks = self._regimens(trail, 'regimens', state, yes)
        self._doses(trail, 'doses', state, picks)
        listing = case.do_home_meds()
        meds = [(m.group(1).strip(), m.group(2).strip()) for m in (re.match('^(.*?) \\[(.*)\\]$', line) for line in str(listing).splitlines()) if m]
        hstate = {**case.state(), 'principal_diagnosis': dx, **facts}
        home = []
        if meds:
            a4 = self._ask(trail, 'home', hstate, {f'h{i:02d}': choice(HOME_Q.format(med=f'{n} ({k})'), HOME_OPTIONS) for i, (n, k) in enumerate(meds)})
            home = [{'med': n, 'class': k, 'choice': a4[f'h{i:02d}']['choice'], 'p': round(a4[f'h{i:02d}']['probabilities'][a4[f'h{i:02d}']['choice']], 4)} for i, (n, k) in enumerate(meds)]

        def treated() -> bool:
            return any((pk['purpose'] == THERAPEUTIC for pk in picks)) or any((m['choice'] == 'continue' and HOME_ANTICOAGULANT.search(m['med']) for m in home))
        dropped = [pk for pk in picks if pk['purpose'] == PROPHYLAXIS] if treated() else []
        picks = [pk for pk in picks if pk not in dropped]
        safety, rechosen, removed, restored, unsafe_drugs, unsafe_lines = ([], [], [], [], set(), [])
        which = [(j, d) for j, pk in enumerate(picks) for d in pk['drugs']] + [('home', i) for i, m in enumerate(home) if m['choice'] == 'continue']
        rnd = 1
        while which:
            p = self._check(trail, 'safety' if rnd == 1 else f'safety{rnd}', hstate, picks, home, which)
            verdicts = {w: self.verdict(self.order_line(picks[w[0]], w[1]) if w[0] != 'home' else self.home_line(home[w[1]]), rnd, p[w]) for w in which}
            safety += list(verdicts.values())
            for w, v in verdicts.items():
                if w[0] == 'home' and v['unsafe']:
                    home[w[1]] = {**home[w[1]], 'choice': 'hold', 'held_by_check': True}
            bad = [w for w, v in verdicts.items() if w[0] != 'home' and v['unsafe']]
            if bad and rnd == MAX_ROUNDS:
                gone = {w[0] for w in bad}
                removed = [picks[j]['shown'] for j in sorted(gone)]
                picks = [pk for j, pk in enumerate(picks) if j not in gone]
                break
            new = []
            if bad:
                unsafe_drugs |= {w[1] for w in bad}
                unsafe_lines += [f"{verdicts[w]['order']} (the reason: {', '.join(verdicts[w]['reasons'])})" for w in bad]
                redo = sorted({picks[w[0]]['purpose'] for w in bad})
                old = {pk['purpose']: pk for pk in picks if pk['purpose'] in redo}
                picks = [pk for pk in picks if pk['purpose'] not in redo]
                ustate = {**state, 'unsafe_for_this_patient': unsafe_lines}
                suffix = '' if rnd == 1 else str(rnd)
                new = self._regimens(trail, f'rechoose{suffix}', ustate, redo, unsafe=unsafe_drugs, q=RECHOOSE_Q)
                self._doses(trail, f'rechoose_doses{suffix}', ustate, new)
                rechosen += [{'round': rnd, 'purpose': q, 'from': old[q]['shown'], 'to': next((x['shown'] for x in new if x['purpose'] == q), 'none')} for q in redo]
                picks += new
            back = dropped if dropped and (not treated()) else []
            picks += back
            restored += back
            dropped = [pk for pk in dropped if not any((pk is x for x in back))]
            which = [(j, d) for j, pk in enumerate(picks) if any((pk is x for x in new + back)) for d in pk['drugs']]
            rnd += 1
        return {'hadm_id': h, 'purposes_yes': {p: round(a1[f'u{i:02d}']['noul'], 4) for i, p in enumerate(names) if a1[f'u{i:02d}']['noul'] >= 0.5}, 'regimens': picks, 'dropped': dropped, 'home': home, 'facts': facts, 'safety': safety, 'rechosen': rechosen, 'removed': removed, 'restored': [pk['shown'] for pk in restored], 'anticoagulation': {'therapeutic': any((pk['purpose'] == THERAPEUTIC for pk in picks)), 'prophylaxis': any((pk['purpose'] == PROPHYLAXIS for pk in picks))}, 'model': self.jev.resolved_model, 'trail': trail}

    def options(self, purpose: str, drug: str) -> List[Dict[str, Any]]:
        """A drug's dose options: an anticoagulant's of the purpose's dose kind, any other drug's default ones."""
        return (self.doses[purpose] if purpose in (THERAPEUTIC, PROPHYLAXIS) else self.doses['default']).get(drug) or []

class ProcedureSelector(PreparedCaseAdapter):
    """Procedure selection on the completed diagnostic chart, including allergy and kidney-function information.
    Each request's stage and cache key are recorded in `trail`."""

    def __init__(self, case, client, categories):
        super().__init__(case, client)
        self.cats = categories
        self.jev.key_salt = '|'.join((x for x in (self.jev.key_salt, SALT_WALK2) if x))

    def _ask(self, trail: List[dict], stage: str, state: Dict[str, Any], qs: Dict[str, Any]) -> Dict[str, Any]:
        if not qs:
            return {}
        trail.append({'stage': stage, 'key': self.jev.key(state, qs), 'questions': len(qs)})
        return self.jev.ask(state, qs, tag='treatment_walk2')

    def ask(self, c: Dict[str, Any]) -> Dict[str, Any]:
        h = c['hadm_id']
        case, _ = self.rebuild(h, same='actions')
        state = {**case.state(), 'principal_diagnosis': f"{dotted(c['code'])} {c['title']}", **safety_facts(self.cohort[h])}
        trail: List[dict] = []
        a1 = self._ask(trail, 'groups', state, {f'cat{k}': noul(CATEGORY_Q.format(ex='; '.join((t for _, t in codes[:WALK_EXAMPLES])))) for k, codes in self.cats.items()})
        yes = [k for k in self.cats if a1[f'cat{k}']['noul'] >= 0.5]
        a2 = self._ask(trail, 'rank', state, {f'pick{k}': choice(PROC_Q, {**{f'o{i}': t for i, (_, t) in enumerate(self.cats[k])}, **NONE}) for k in yes})
        ranked, cands = ({}, [])
        for k in yes:
            pr, pick = (a2[f'pick{k}']['probabilities'], a2[f'pick{k}']['choice'])
            top = sorted(((v, o) for o, v in pr.items() if o != 'none'), key=lambda x: -x[0])[:WALK_CANDIDATES]
            ranked[k] = {'choice': pick, 'p_none': round(pr.get('none', 0.0), 4), 'top': [[self.cats[k][int(o[1:])][0], round(v, 4)] for v, o in top]}
            if pick == 'none':
                continue
            cands += [{'category': k, 'icd_code': self.cats[k][int(o[1:])][0], 'icd_version': 9, 'long_title': self.cats[k][int(o[1:])][1], 'p_choice': round(v, 4)} for v, o in top if v >= MIN_CANDIDATE or o == pick]
        a3 = self._ask(trail, 'indicated', state, {f'c{n:02d}': noul(INDICATED_Q.format(title=x['long_title'])) for n, x in enumerate(cands)})
        for n, x in enumerate(cands):
            x['p_indicated'] = round(a3[f'c{n:02d}']['noul'], 4)
        kept, removed = ([], [])
        for x in sorted((x for x in cands if x['p_indicated'] >= 0.5), key=lambda x: (-x['p_indicated'], -x['p_choice'])):
            twin = next((y for y in kept if y['icd_code'][:3] == x['icd_code'][:3]), None)
            if twin:
                removed.append({**x, 'reason': f"same subcategory as {twin['icd_code']}"})
            else:
                kept.append(x)
        ops = [x for x in kept if int(x['icd_code'][:2]) < 87 and (not approach(x['icd_code']))]
        a4 = self._ask(trail, 'addition', state, {f'a{n:02d}': noul(ADDITION_Q.format(done='; '.join((y['long_title'] for y in ops[:n])), title=x['long_title'])) for n, x in enumerate(ops) if n})
        keep = []
        for n, x in enumerate(ops):
            if n:
                x['p_addition'] = round(a4[f'a{n:02d}']['noul'], 4)
                if x['p_addition'] < 0.5:
                    removed.append({**x, 'reason': 'not needed in addition'})
                    continue
            keep.append(x)
        host = next((x for x in keep if in_abdomen(x['icd_code'])), None)
        for x in kept:
            if approach(x['icd_code']):
                if host:
                    removed.append({**x, 'reason': f"the approach of {host['icd_code']}"})
                else:
                    keep.append(x)
            elif int(x['icd_code'][:2]) >= 87:
                keep.append(x)
        procedures = [{**x, 'p': 1.0} for x in keep]
        return {'hadm_id': h, 'categories_yes': {k: round(a1[f'cat{k}']['noul'], 4) for k in yes}, 'ranked': ranked, 'candidates': cands, 'procedures': procedures, 'removed': removed, 'model': self.jev.resolved_model, 'trail': trail}
