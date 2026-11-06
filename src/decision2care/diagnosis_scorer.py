"""Score the full ICD-10-CM label space with Jev: beam search over chapter -> block -> category -> code.

Every node of the beam is one Choice question; all nodes at the same depth are asked in a single
request (they share the state), so the whole scoring costs 4 sequential requests regardless of beam
width. Path probability = product of edge probabilities, which gives a (beam-pruned) probability
distribution over billable codes. Categories with more than `leaf_chunk` codes are scored in chunks
followed by a run-off Choice over the best candidates of each chunk (TypeSafe's 2-stage recipe).
Treatment is asked as typed Noul questions in the first request (independent of the diagnosis).
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from decision2care.icd_hierarchy import ICD10Tree, _norm
from decision2care.client import choice, noul
from decision2care.prompts import DEFAULT as _DEFAULT_PROMPTS
from decision2care.prompts import get_prompts


@dataclass
class ScorerConfig:
    beam_chapters: int = 4
    beam_blocks: int = 4
    beam_categories: int = 5
    leaf_chunk: int = 200
    top_n_codes: int = 50
    ask_treatment: bool = True
    # final run-off (2026-09-28): when the best code by path probability is not in the category with the largest mass, or a code
    # from another category is close behind it, ask one cross-category Choice among the best code of each beam category. This
    # removes the bias against diseases whose probability is split over many sibling codes (heart failure, bowel obstruction).
    final_runoff: bool = False
    runoff_k: int = 6
    runoff_close: float = 0.5
    # leaf questions (2026-09-28): "flat" asks one Choice over every observed billable code of a category (up to 200 options);
    # "hierarchical" descends the ICD-10-CM sub-group tree (4th to 7th character) with one small Choice per level, keeping the
    # best `leaf_beam` nodes per level; single-child chains auto-descend and the encounter suffix (7th character) is not asked.
    leaf_mode: str = "flat"
    leaf_beam: int = 3
    # robustness evaluation (2026-10-01): instruction wording, and the order in which options are offered. With
    # option_order_seed set, every Choice offers its options in a seeded random order; with order_replicates > 1, every Choice
    # is asked under that many seeded orders in the same request and the probabilities are averaged. Requests that differ from
    # the default only in option order are salted in the cache key (the key sorts dict keys, so they would collide otherwise).
    prompt_set: str = "default"
    option_order_seed: Optional[int] = None
    order_replicates: int = 1


TREATMENT_QUESTIONS: Dict[str, Tuple[str, str]] = {
    # key: (question, phrase used to rebuild a treatment sentence for the MIMIC-CDM treatment evaluator)
    "appendectomy": ("Is surgical removal of the appendix (appendectomy) indicated for this patient?", "appendectomy"),
    "cholecystectomy": ("Is surgical removal of the gallbladder (cholecystectomy), now or after recovery, indicated for this patient?", "cholecystectomy"),
    "ercp": ("Is endoscopic retrograde cholangiopancreatography (ERCP) indicated for this patient?", "ERCP"),
    "drainage": ("Is percutaneous or surgical drainage of an abscess or fluid collection indicated for this patient?", "percutaneous drainage of the abscess"),
    "bowel_resection": ("Is resection of colon or small bowel (colectomy, sigmoidectomy) indicated for this patient?", "colectomy"),
    "antibiotics": ("Are antibiotics indicated for this patient?", "antibiotics"),
    "iv_fluids": ("Is intravenous fluid therapy indicated for this patient?", "IV fluids"),
    "analgesia": ("Is analgesia for pain control indicated for this patient?", "analgesia for pain"),
    "monitoring": ("Is close clinical monitoring (serial examinations, laboratory tests and vital signs) indicated for this patient?", "monitoring"),
    "bowel_rest": ("Is bowel rest (nothing by mouth) with nutritional support indicated for this patient?", "bowel rest and nutrition support"),
    "colonoscopy_after_recovery": ("Is a colonoscopy after the acute episode has resolved indicated for this patient?", "colonoscopy after recovery"),
}

# Disease-specific treatment sets for the MIRA extension diseases (guideline items scored by decision2care.evaluators_ext).
# The abdominal set above is left untouched so cached answers for the MIMIC-CDM diseases stay valid.
_OXYGEN = ("Is supplemental oxygen or respiratory support (oxygen therapy, non-invasive or mechanical ventilation) indicated for this patient?", "supplemental oxygen")
_ANTICOAG = ("Is therapeutic anticoagulation (heparin infusion, low-molecular-weight heparin or a direct oral anticoagulant) indicated for this patient?", "therapeutic anticoagulation")
TREATMENT_SETS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "abdominal": TREATMENT_QUESTIONS,
    "pneumonia": {
        "antibiotics": TREATMENT_QUESTIONS["antibiotics"],
        "oxygen": _OXYGEN,
        "iv_fluids": TREATMENT_QUESTIONS["iv_fluids"],
        "analgesia": TREATMENT_QUESTIONS["analgesia"],
        "monitoring": TREATMENT_QUESTIONS["monitoring"],
        "pleural_drainage": ("Is drainage of a pleural effusion or empyema (thoracentesis or chest tube) indicated for this patient?", "chest tube drainage"),
        "anticoagulation": _ANTICOAG,
    },
    "uti": {
        "antibiotics": TREATMENT_QUESTIONS["antibiotics"],
        "iv_fluids": TREATMENT_QUESTIONS["iv_fluids"],
        "analgesia": TREATMENT_QUESTIONS["analgesia"],
        "monitoring": TREATMENT_QUESTIONS["monitoring"],
        "urinary_drainage": ("Is relief of urinary tract obstruction (ureteral stent or nephrostomy) indicated for this patient?", "ureteral stent or nephrostomy for urinary obstruction"),
    },
    "pulmonary_embolism": {
        "anticoagulation": _ANTICOAG,
        "thrombolysis": ("Is reperfusion therapy (systemic or catheter-directed thrombolysis, or thrombectomy) indicated for this patient?", "thrombolysis"),
        "ivc_filter": ("Is an inferior vena cava (IVC) filter indicated for this patient?", "IVC filter"),
        "oxygen": _OXYGEN,
        "iv_fluids": TREATMENT_QUESTIONS["iv_fluids"],
        "analgesia": TREATMENT_QUESTIONS["analgesia"],
        "monitoring": TREATMENT_QUESTIONS["monitoring"],
        "antibiotics": TREATMENT_QUESTIONS["antibiotics"],
    },
}

# Disease-specific sets for the MIMIC_rewoo diseases and the cardiac sets (asked as Nouls with the chapter Choice; reported
# descriptively, no MIMIC-CDM-style rubric evaluator exists for them yet).
_DIURETICS = ("Is intravenous loop diuretic therapy indicated for this patient?", "IV diuretics")
_VASOPRESSORS = ("Is vasopressor support (e.g. norepinephrine) indicated for this patient?", "vasopressors")
_NG_TUBE = ("Is nasogastric tube decompression indicated for this patient?", "nasogastric tube decompression")
_LAPAROTOMY = ("Is operative management (laparotomy or laparoscopy with adhesiolysis, reduction or bowel resection) indicated for this patient, now or if non-operative management fails?", "surgery")
_PTBD = ("Is percutaneous transhepatic biliary drainage indicated for this patient if endoscopic drainage fails or is not possible?", "percutaneous biliary drainage")
_ACYCLOVIR = ("Is empiric antiviral therapy (intravenous acyclovir) indicated for this patient?", "acyclovir")
_DEXAMETHASONE = ("Is adjunctive dexamethasone indicated for this patient?", "dexamethasone")
_LP = ("Is a lumbar puncture for cerebrospinal fluid analysis indicated for this patient?", "lumbar puncture")
_SOURCE_CONTROL = ("Is source control (drainage of an abscess, removal of an infected device or debridement of infected tissue) indicated for this patient?", "source control")
_ANEURYSM_REPAIR = ("Is securing of a ruptured intracranial aneurysm (endovascular coiling or surgical clipping) indicated for this patient?", "aneurysm coiling or clipping")
_NIMODIPINE = ("Is oral nimodipine indicated for this patient?", "nimodipine")
_BP_CONTROL = ("Is intravenous blood pressure control indicated for this patient?", "blood pressure control")
_EVD = ("Is an external ventricular drain for hydrocephalus indicated for this patient?", "external ventricular drain")
_REVERSAL = ("Is reversal of anticoagulant or antiplatelet effect indicated for this patient?", "reversal of anticoagulation")
_SEIZURE_PPX = ("Is antiseizure medication indicated for this patient?", "antiseizure medication")
_ANTIPLATELET = ("Is dual antiplatelet therapy (aspirin plus a P2Y12 inhibitor) indicated for this patient?", "dual antiplatelet therapy")
_PCI = ("Is urgent or early invasive coronary angiography with percutaneous coronary intervention indicated for this patient?", "coronary angiography with PCI")
_BETA_BLOCKER = ("Is beta-blocker therapy indicated for this patient?", "beta-blocker")
_STATIN = ("Is high-intensity statin therapy indicated for this patient?", "statin")
_NITRATES = ("Is intravenous vasodilator therapy (nitrates) indicated for this patient?", "IV nitrates")
_GDMT = ("Is initiation or optimization of guideline-directed heart failure medication (beta-blocker, ACE inhibitor/ARB/ARNI, mineralocorticoid receptor antagonist, SGLT2 inhibitor) indicated for this patient?", "guideline-directed heart failure therapy")
_INOTROPES = ("Is inotropic support (e.g. dobutamine, milrinone) indicated for this patient?", "inotropes")
_FLUID_RESTRICTION = ("Is fluid and sodium restriction indicated for this patient?", "fluid restriction")
_AVR = ("Is aortic valve replacement (surgical or transcatheter) indicated for this patient?", "aortic valve replacement")
_BAV = ("Is balloon aortic valvuloplasty indicated for this patient?", "balloon valvuloplasty")
_MV_REPAIR = ("Is mitral valve repair or replacement (surgical or transcatheter) indicated for this patient?", "mitral valve repair")
_COMMON = {"iv_fluids": TREATMENT_QUESTIONS["iv_fluids"], "analgesia": TREATMENT_QUESTIONS["analgesia"], "monitoring": TREATMENT_QUESTIONS["monitoring"]}
TREATMENT_SETS.update({
    "cholangitis": {"antibiotics": TREATMENT_QUESTIONS["antibiotics"], "ercp": TREATMENT_QUESTIONS["ercp"], "ptbd": _PTBD, "cholecystectomy": TREATMENT_QUESTIONS["cholecystectomy"], **_COMMON},
    "bowel_obstruction": {"ng_decompression": _NG_TUBE, "bowel_rest": TREATMENT_QUESTIONS["bowel_rest"], "surgery": _LAPAROTOMY, "antibiotics": TREATMENT_QUESTIONS["antibiotics"], **_COMMON},
    "meningitis": {"antibiotics": TREATMENT_QUESTIONS["antibiotics"], "acyclovir": _ACYCLOVIR, "dexamethasone": _DEXAMETHASONE, "lumbar_puncture": _LP, **_COMMON},
    "pyelonephritis": {"antibiotics": TREATMENT_QUESTIONS["antibiotics"], "urinary_drainage": TREATMENT_SETS["uti"]["urinary_drainage"], **_COMMON},
    "sepsis": {"antibiotics": TREATMENT_QUESTIONS["antibiotics"], "iv_fluids": TREATMENT_QUESTIONS["iv_fluids"], "vasopressors": _VASOPRESSORS, "source_control": _SOURCE_CONTROL, "oxygen": _OXYGEN, "monitoring": TREATMENT_QUESTIONS["monitoring"]},
    "subarachnoid_hemorrhage": {"aneurysm_repair": _ANEURYSM_REPAIR, "nimodipine": _NIMODIPINE, "bp_control": _BP_CONTROL, "evd": _EVD, "reversal": _REVERSAL, "seizure_prophylaxis": _SEIZURE_PPX, "analgesia": TREATMENT_QUESTIONS["analgesia"], "monitoring": TREATMENT_QUESTIONS["monitoring"]},
    "myocardial_infarction": {"antiplatelet": _ANTIPLATELET, "anticoagulation": _ANTICOAG, "coronary_angiography_pci": _PCI, "beta_blocker": _BETA_BLOCKER, "statin": _STATIN, "analgesia": TREATMENT_QUESTIONS["analgesia"], "oxygen": _OXYGEN, "monitoring": TREATMENT_QUESTIONS["monitoring"]},
    "congestive_heart_failure": {"diuretics": _DIURETICS, "oxygen": _OXYGEN, "nitrates": _NITRATES, "gdmt": _GDMT, "fluid_restriction": _FLUID_RESTRICTION, "inotropes": _INOTROPES, "monitoring": TREATMENT_QUESTIONS["monitoring"]},
    "aortic_stenosis": {"valve_replacement": _AVR, "balloon_valvuloplasty": _BAV, "diuretics": _DIURETICS, "oxygen": _OXYGEN, "monitoring": TREATMENT_QUESTIONS["monitoring"]},
    "mitral_regurgitation": {"mitral_valve_repair": _MV_REPAIR, "diuretics": _DIURETICS, "nitrates": _NITRATES, "anticoagulation": _ANTICOAG, "oxygen": _OXYGEN, "monitoring": TREATMENT_QUESTIONS["monitoring"]},
})
TREATMENT_SET_BY_PATHOLOGY = {
    "appendicitis": "abdominal", "cholecystitis": "abdominal", "pancreatitis": "abdominal", "diverticulitis": "abdominal",
    "pneumonia": "pneumonia", "uti": "uti", "pulmonary_embolism": "pulmonary_embolism",
    **{k: k for k in ("cholangitis", "bowel_obstruction", "meningitis", "pyelonephritis", "sepsis", "subarachnoid_hemorrhage",
                      "myocardial_infarction", "congestive_heart_failure", "aortic_stenosis", "mitral_regurgitation")},
}
TREATMENT_PHRASES: Dict[str, str] = {k: v[1] for qs in TREATMENT_SETS.values() for k, v in qs.items()}

PRINCIPAL = _DEFAULT_PROMPTS["principal"]  # default wording; the scorer reads its prompt set


def _norm_probs(p: Dict[str, float]) -> Dict[str, float]:
    s = sum(p.values())
    return {k: v / s for k, v in p.items()} if s > 0 else p


class HierarchicalDiagnosisScorer:
    def __init__(self, client, tree: ICD10Tree, cfg: ScorerConfig = ScorerConfig()):
        self.client = client
        self.tree = tree
        self.cfg = cfg
        self.P = get_prompts(cfg.prompt_set)

    # ------------------------------------------------------------------ helpers
    def _permuted(self) -> bool:
        return self.cfg.option_order_seed is not None or int(self.cfg.order_replicates or 1) > 1

    def _orders(self, key: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The option orders under which one Choice is asked: [opts] by default, seeded shuffles otherwise."""
        if not self._permuted():
            return [opts]
        seed = 0 if self.cfg.option_order_seed is None else self.cfg.option_order_seed
        out = []
        for r in range(max(1, int(self.cfg.order_replicates or 1))):
            keys = list(opts)
            random.Random(f"{seed}|{key}|{r}").shuffle(keys)
            out.append({k: opts[k] for k in keys})
        return out

    def _put(self, questions: Dict[str, dict], qid: str, instr: str, opts: Dict[str, Any], key: str) -> None:
        vs = self._orders(key, opts)
        if len(vs) == 1:
            questions[qid] = choice(instr, vs[0])
        else:
            for r, v in enumerate(vs):
                questions[f"{qid}##{r}"] = choice(instr, v)

    @staticmethod
    def _get(ans: Dict[str, Any], qid: str) -> Tuple[Dict[str, float], Optional[float]]:
        """(normalised probabilities, confidence) of a Choice; replicates (qid##r) are averaged."""
        if qid in ans:
            return _norm_probs(ans[qid]["probabilities"]), ans[qid].get("confidence")
        reps = [a for q, a in ans.items() if q.startswith(qid + "##")]
        if not reps:
            return {}, None
        ps = [_norm_probs(a["probabilities"]) for a in reps]
        keys = list(dict.fromkeys(k for p in ps for k in p))
        confs = [a.get("confidence") for a in reps if a.get("confidence") is not None]
        return {k: sum(p.get(k, 0.0) for p in ps) / len(ps) for k in keys}, (sum(confs) / len(confs) if confs else None)

    def _ask(self, state, questions: Dict[str, dict], tag: str) -> Dict[str, Any]:
        if not self._permuted():
            return self.client.ask(state, questions, tag=tag)
        return self.client.ask(state, questions, tag=tag, key_extra=f"option_order:{self.cfg.option_order_seed}:{self.cfg.order_replicates}")

    def _ask_level(self, state, parents: List[Optional[str]], instr_fn, tag: str) -> Dict[Optional[str], Dict[str, float]]:
        """One request with one Choice per parent node; returns {parent: {child: prob}}."""
        out: Dict[Optional[str], Dict[str, float]] = {}
        questions: Dict[str, dict] = {}
        asked: List[Optional[str]] = []
        for p in parents:
            opts = self.tree.options(p)
            if len(opts) == 0:
                out[p] = {}
            elif len(opts) == 1:
                out[p] = {next(iter(opts)): 1.0}
            else:
                self._put(questions, f"n::{p}", instr_fn(p), opts, key=f"n::{p}")
                asked.append(p)
        if questions:
            ans = self._ask(state, questions, tag)
            for p in asked:
                out[p] = self._get(ans, f"n::{p}")[0]
        return out

    def _leaf_probs(self, state, cat: str) -> Tuple[Dict[str, float], int]:
        """p(code | category) for every observed code under `cat` (chunked when large)."""
        opts = self.tree.options(cat)
        if not opts:
            return {cat: 1.0}, 0
        if len(opts) == 1:
            return {next(iter(opts)): 1.0}, 0
        instr = self.P["dx_leaf"].format(code=cat, title=self.tree.title(cat), principal=self.P["principal"])
        keys = list(opts)
        if len(keys) <= self.cfg.leaf_chunk:
            q: Dict[str, dict] = {}
            self._put(q, "leaf", instr, opts, key=f"leaf::{cat}")
            return self._get(self._ask(state, q, "dx.L4"), "leaf")[0], 1
        n_chunks = math.ceil(len(keys) / self.cfg.leaf_chunk)
        size = math.ceil(len(keys) / n_chunks)
        qs: Dict[str, dict] = {}
        for i in range(n_chunks):
            self._put(qs, f"chunk{i}", instr, {k: opts[k] for k in keys[i * size : (i + 1) * size]}, key=f"chunk::{cat}::{i}")
        ans = self._ask(state, qs, "dx.L4chunk")
        per_chunk = max(2, self.cfg.leaf_chunk // n_chunks)
        finalists: Dict[str, Any] = {}
        if not self._permuted():  # answer order, as in every run before the robustness evaluation (cache-compatible)
            chunk_probs = [a["probabilities"] for a in ans.values()]
        else:
            chunk_probs = [self._get(ans, f"chunk{i}")[0] for i in range(n_chunks)]
        for probs in chunk_probs:
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:per_chunk]
            finalists.update({k: opts[k] for k, _ in top})
        q2: Dict[str, dict] = {}
        self._put(q2, "leaf", instr, finalists, key=f"leaf::{cat}::final")
        return self._get(self._ask(state, q2, "dx.L4"), "leaf")[0], 1 + n_chunks

    # ------------------------------------------------------------------ hierarchical leaf descent
    @staticmethod
    def _dotted(code: str) -> str:
        return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"

    def _subgroup_children(self, node: str, leaves: Dict[str, str]) -> List[str]:
        """Sub-groups of `node` in ICD-10-CM (4th to 7th character) that are, or contain, an observed billable code."""
        import simple_icd_10_cm as cm

        try:
            kids = cm.get_children(self._dotted(node))
        except Exception:  # noqa: BLE001
            kids = []
        out = []
        for k in kids:
            kn = k.replace(".", "").upper()
            if kn in leaves or any(l.startswith(kn) for l in leaves):
                out.append(kn)
        return out

    def _node_title(self, node: str) -> str:
        import simple_icd_10_cm as cm

        if node in self.tree.nodes:
            return self.tree.title(node)
        try:
            return cm.get_description(self._dotted(node))
        except Exception:  # noqa: BLE001
            return node

    def _leaf_probs_hier(self, state, cat: str) -> Tuple[Dict[str, float], int]:
        """p(code | category) by descending the sub-group tree: one request per level with a Choice per kept node over its
        sub-groups (a node that is itself a valid code is offered among its own sub-groups), beam of `leaf_beam` nodes per
        level, single-child chains auto-descend, and the encounter suffix (A/D/S) resolves to the initial encounter."""
        t = self.tree
        leaves = t.options(cat)
        if not leaves:
            return {cat: 1.0}, 0
        if len(leaves) == 1:
            return {next(iter(leaves)): 1.0}, 0
        frontier: List[Tuple[str, float]] = [(cat, 1.0)]
        out: Dict[str, float] = {}
        n_calls = 0
        for _depth in range(6):
            if not frontier:
                break
            questions: Dict[str, dict] = {}
            weights: Dict[str, float] = {}
            nxt: List[Tuple[str, float]] = []
            for node, p in frontier:
                kids = self._subgroup_children(node, leaves)
                if not kids:
                    out[node] = out.get(node, 0.0) + p
                    continue
                if len(kids) == 1 and node not in leaves:
                    nxt.append((kids[0], p))
                    continue
                if all(len(k) == 7 for k in kids) and len({k[:6] for k in kids}) == 1 and node not in leaves:
                    nxt.append((next((k for k in kids if k.endswith("A")), kids[0]), p))
                    continue
                opts = {k: self._node_title(k) for k in kids}
                if node in leaves:  # the node itself is a valid code (e.g. an 'unspecified' 4-character code)
                    opts[node] = self._node_title(node) + " (this code itself)"
                questions[f"sub::{node}"] = choice(
                    self.P["dx_subgroup"].format(code=node, title=self._node_title(node), principal=self.P["principal"]), opts)
                weights[node] = p
            if questions:
                ans = self.client.ask(state, questions, tag="dx.L4")
                n_calls += 1
                for qid, a in ans.items():
                    node = qid.split("::", 1)[1]
                    for k, pk in _norm_probs(a["probabilities"]).items():
                        if k == node:
                            out[node] = out.get(node, 0.0) + weights[node] * pk
                        else:
                            nxt.append((k, weights[node] * pk))
            frontier = sorted(nxt, key=lambda kv: -kv[1])[: self.cfg.leaf_beam]
        for node, p in frontier:  # depth limit reached: count what is left as leaves
            out[node] = out.get(node, 0.0) + p
        return out, n_calls

    # ------------------------------------------------------------------ main
    def _candidate_codes(self, state, codes: List[str], code_paths: Dict[str, float]) -> Tuple[List[str], int]:
        """Outside candidates (ICD-10-CM codes, dotted or not) as codes of the tree: a code as it stands; else the best code of its
        three-character category, by path probability when the beam reached the category, by one leaf Choice otherwise. Codes
        whose category is not in the tree are dropped. Returns the codes and the number of requests asked."""
        t, out, n = self.tree, [], 0
        for c in codes:
            c = _norm(c)
            if t.level(c) == "code":
                out.append(c)
            elif t.level(c[:3]) == "category":
                inside = [k for k in code_paths if t.level(k) == "code" and t.category_of(k) == c[:3]]
                if inside:
                    out.append(max(inside, key=code_paths.get))
                else:
                    leaf, k = self._leaf_probs(state, c[:3])
                    n += k
                    if leaf:
                        out.append(max(leaf, key=leaf.get))
        return list(dict.fromkeys(out)), n

    def score(self, state: Dict[str, Any], treatment_set: str = "abdominal", extra_candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        """`extra_candidates`: diagnoses proposed from outside (llm_suggest's ICD-10-CM codes). After the usual decision a second
        run-off offers them next to Jev's own candidates (the run-off's, or the top code); Jev's choice there is the diagnosis, and
        `candidate` records the choice without them (`top1_without`)."""
        t = self.tree
        tx_questions = TREATMENT_SETS[treatment_set]
        # ---- L1: chapters (+ treatment nouls in the same request) ----
        principal = self.P["principal"]
        questions: Dict[str, dict] = {}
        self._put(questions, "n::", self.P["dx_chapter"].format(principal=principal), t.options(None), key="chapter")
        if self.cfg.ask_treatment:
            for k, (q, _) in tx_questions.items():
                questions[f"tx::{k}"] = noul(q)
        ans = self._ask(state, questions, "dx.L1")
        chapter_probs, chapter_conf = self._get(ans, "n::")
        treatment = {k: ans[f"tx::{k}"]["noul"] for k in tx_questions if f"tx::{k}" in ans}

        # ---- L2: blocks of the top chapters ----
        beam_ch = sorted(chapter_probs.items(), key=lambda kv: -kv[1])[: self.cfg.beam_chapters]
        blk = self._ask_level(
            state,
            [c for c, _ in beam_ch],
            lambda c: self.P["dx_block"].format(title=t.title(c), code=c, principal=principal),
            tag="dx.L2",
        )
        block_paths: Dict[str, float] = {}
        for c, pc in beam_ch:
            for b, pb in blk.get(c, {}).items():
                block_paths[b] = pc * pb
        beam_blk = sorted(block_paths.items(), key=lambda kv: -kv[1])[: self.cfg.beam_blocks]

        # ---- L3: categories of the top blocks ----
        cat = self._ask_level(
            state,
            [b for b, _ in beam_blk],
            lambda b: self.P["dx_category"].format(title=t.title(b), code=b, principal=principal),
            tag="dx.L3",
        )
        cat_paths: Dict[str, float] = {}
        for b, pb in beam_blk:
            for c3, pc3 in cat.get(b, {}).items():
                cat_paths[c3] = pb * pc3
        beam_cat = sorted(cat_paths.items(), key=lambda kv: -kv[1])[: self.cfg.beam_categories]

        # ---- L4: codes of the top categories ----
        code_paths: Dict[str, float] = {}
        n_leaf_calls = 0
        for c3, p3 in beam_cat:
            leaf_p, n_calls = (self._leaf_probs_hier if self.cfg.leaf_mode == "hierarchical" else self._leaf_probs)(state, c3)
            n_leaf_calls += n_calls
            for code, pl in leaf_p.items():
                code_paths[code] = p3 * pl
        ranked = sorted(code_paths.items(), key=lambda kv: -kv[1])
        top_codes = [
            {"code": c, "title": t.title(c), "prob": round(p, 6), "category": t.category_of(c) if t.level(c) == "code" else c}
            for c, p in ranked[: self.cfg.top_n_codes]
        ]
        runoff: Optional[Dict[str, Any]] = None
        if self.cfg.final_runoff and len(ranked) > 1:
            cat_of = lambda c: t.category_of(c) if t.level(c) == "code" else c  # noqa: E731
            best_by_cat: Dict[str, Tuple[str, float]] = {}
            for c, p in ranked:  # descending, so the first code of a category is its best
                best_by_cat.setdefault(cat_of(c), (c, p))
            cat_order = [c3 for c3, _ in sorted(cat_paths.items(), key=lambda kv: -kv[1]) if c3 in best_by_cat]
            top1_cat = cat_of(ranked[0][0])
            runner = next(((c, p) for c, p in ranked[1:] if cat_of(c) != top1_cat), None)
            ambiguous = (cat_order and cat_order[0] != top1_cat) or (runner is not None and runner[1] >= self.cfg.runoff_close * ranked[0][1])
            if ambiguous:
                cands = [best_by_cat[c3][0] for c3 in cat_order[: self.cfg.runoff_k]]
                if ranked[0][0] not in cands:
                    cands = [ranked[0][0]] + cands[: self.cfg.runoff_k - 1]
                opts = {c: f"{c} {t.title(c)}" for c in cands}
                qr: Dict[str, dict] = {}
                self._put(qr, "runoff", self.P["dx_runoff"].format(principal=principal), opts, key="runoff")
                rp = self._get(self._ask(state, qr, "dx.runoff"), "runoff")[0]
                winner = max(rp.items(), key=lambda kv: kv[1])[0] if rp else ranked[0][0]
                runoff = {"triggered": True, "candidates": cands, "probs": {k: round(v, 5) for k, v in rp.items()}, "winner": winner,
                          "path_top1": ranked[0][0], "changed": winner != ranked[0][0]}
                n_leaf_calls += 1
                if winner != ranked[0][0]:  # the winner leads the ranking; the others keep their path order
                    w = [x for x in top_codes if x["code"] == winner] or [{"code": winner, "title": t.title(winner), "prob": round(code_paths.get(winner, 0.0), 6), "category": cat_of(winner)}]
                    top_codes = w + [x for x in top_codes if x["code"] != winner]
            else:
                runoff = {"triggered": False}
        top1 = top_codes[0] if top_codes else None
        candidate: Optional[Dict[str, Any]] = None
        if extra_candidates and ranked:
            extra, n_x = self._candidate_codes(state, extra_candidates, code_paths)
            n_leaf_calls += n_x
            base = list(runoff["candidates"]) if runoff and runoff.get("triggered") else [ranked[0][0]]
            cands = base + [c for c in extra if c not in base]
            candidate = {"proposed": list(extra_candidates), "codes": extra, "top1_without": top1["code"] if top1 else None, "chosen": False}
            if len(cands) > len(base):  # an option Jev's own decision did not hold
                opts = {c: f"{c} {t.title(c)}" for c in cands}
                qc: Dict[str, dict] = {}
                self._put(qc, "runoff_candidate", self.P["dx_runoff"].format(principal=principal), opts, key="runoff_candidate")
                rp = self._get(self._ask(state, qc, "dx.runoff_candidate"), "runoff_candidate")[0]
                n_leaf_calls += 1
                winner = max(rp.items(), key=lambda kv: kv[1])[0] if rp else (top1["code"] if top1 else cands[0])
                candidate.update(candidates=cands, probs={k: round(v, 5) for k, v in rp.items()}, winner=winner, chosen=winner not in base)
                if top1 is None or winner != top1["code"]:
                    cat = t.category_of(winner) if t.level(winner) == "code" else winner
                    w = [x for x in top_codes if x["code"] == winner] or [{"code": winner, "title": t.title(winner), "prob": round(code_paths.get(winner, 0.0), 6), "category": cat}]
                    top_codes = w + [x for x in top_codes if x["code"] != winner]
                    top1 = top_codes[0]
        # separation between the best and second-best path (cookbook metric)
        sep = (ranked[0][1] / ranked[1][1]) if len(ranked) > 1 and ranked[1][1] > 0 else float("inf")
        return {
            "top_codes": top_codes,
            "top1": top1,
            "chapter_probs": {k: round(v, 5) for k, v in chapter_probs.items()},
            "chapter_confidence": chapter_conf,
            "block_probs": {k: round(v, 5) for k, v in block_paths.items()},
            "category_probs": {k: round(v, 5) for k, v in sorted(cat_paths.items(), key=lambda kv: -kv[1])},
            "beam": {"chapters": [c for c, _ in beam_ch], "blocks": [b for b, _ in beam_blk], "categories": [c for c, _ in beam_cat]},
            "separation": round(sep, 3) if sep != float("inf") else None,
            "runoff": runoff,
            **({"candidate": candidate} if candidate is not None else {}),
            "treatment_nouls": treatment,
            "treatment_set": treatment_set,
            "n_requests": 3 + n_leaf_calls,
            "config": asdict(self.cfg),
        }


def treatment_text(treatment_nouls: Dict[str, float], threshold: float = 0.5) -> str:
    """Rebuild a free-text treatment plan from the typed answers for MIMIC-CDM's treatment evaluator."""
    phrases = [TREATMENT_PHRASES[k] for k, p in treatment_nouls.items() if p >= threshold and k in TREATMENT_PHRASES]
    return ", ".join(phrases) if phrases else "supportive care"
