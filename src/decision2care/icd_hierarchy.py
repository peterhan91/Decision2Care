"""ICD-10-CM label space for full-list diagnosis scoring.

Four fixed levels: chapter -> block -> category (3 characters) -> code (billable leaf). The
chapter/block/category skeleton comes from the ICD-10-CM tabular list shipped with
`simple_icd_10_cm`; category titles are taken from MIMIC-IV's `d_icd_diagnoses` where
available (same coding era as the data); leaves are every ICD-10 code that appears as a
hospital diagnosis in MIMIC-IV (`icd_diagnosis_universe.csv`, written by build_dataset).

Also provides the ICD-9 -> ICD-10 crosswalk (inverted MIMIC-CDM `ICD10_to_9.csv`, most
frequent target wins) used to map ground truth of ICD-9-coded admissions, and the
MIMIC-CDM pathology keyword rules applied to ICD titles.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from decision2care.paths import CDM_DIR

VENDOR_ICD = Path(__file__).resolve().parent / "vendor" / "mimic_cdm" / "icd"

_RANGE_RE = re.compile(r"\(([A-Z]\d\d)-([A-Z]\d\d)\)\s*$")


def _norm(code: str) -> str:
    return code.replace(".", "").strip().upper()


@dataclass
class Node:
    code: str
    title: str
    level: str  # chapter | block | category | code
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    n_admissions: int = 0
    n_primary: int = 0


class ICD10Tree:
    def __init__(
        self,
        universe_csv: Path = CDM_DIR / "icd_diagnosis_universe.csv",
        d_icd_csv: Path = CDM_DIR / "d_icd_diagnoses.csv",
        exclude_chapters: Tuple[str, ...] = ("V00-Y99",),
        min_leaf_admissions: int = 1,
    ):
        import simple_icd_10_cm as cm

        self.nodes: Dict[str, Node] = {}
        self.chapters: List[str] = []
        self._block_ranges: List[Tuple[str, str, str]] = []  # (start, end, block key)

        d_icd = pd.read_csv(d_icd_csv, dtype=str)
        d10 = d_icd[d_icd["icd_version"] == "10"]
        mimic_titles = dict(zip(d10["icd_code"].map(_norm), d10["long_title"]))

        # ---- skeleton: chapters / blocks / categories from the ICD-10-CM tabular list ----
        for ch in [c for c in cm.get_all_codes(False) if cm.is_chapter(c)]:
            desc = cm.get_description(ch)
            m = _RANGE_RE.search(desc)
            key = f"{m.group(1)}-{m.group(2)}" if m else f"chapter-{ch}"
            title = _RANGE_RE.sub("", desc).strip()
            if key in exclude_chapters:
                continue
            self.nodes[key] = Node(key, title, "chapter")
            self.chapters.append(key)
            for blk in cm.get_children(ch):
                raw_key = blk.strip().upper()
                s, e = raw_key.split("-") if "-" in raw_key else (raw_key, raw_key)
                bkey = f"{s}-{e}"  # e.g. K35-K38; single-category blocks become N99-N99
                self.nodes[bkey] = Node(bkey, _RANGE_RE.sub("", cm.get_description(blk)).strip(), "block", parent=key)
                self.nodes[key].children.append(bkey)
                self._block_ranges.append((s, e, bkey))
                for cat in cm.get_children(blk):
                    ckey = _norm(cat)
                    self.nodes[ckey] = Node(ckey, mimic_titles.get(ckey, cm.get_description(cat)), "category", parent=bkey)
                    self.nodes[bkey].children.append(ckey)

        # ---- leaves: MIMIC-observed ICD-10 codes ----
        uni = pd.read_csv(universe_csv, dtype={"icd_code": str, "icd_version": str})
        uni10 = uni[(uni["icd_version"] == "10") & (uni["n_admissions"] >= min_leaf_admissions)]
        self.n_unplaced = 0
        for r in uni10.itertuples():
            code = _norm(r.icd_code)
            if len(code) < 3:
                continue
            cat = code[:3]
            if cat not in self.nodes:  # category unknown to the tabular list (e.g. later additions): place by block range
                bkey = self._block_for(cat)
                if bkey is None:
                    self.n_unplaced += 1
                    continue
                self.nodes[cat] = Node(cat, mimic_titles.get(cat, str(r.long_title)), "category", parent=bkey)
                self.nodes[bkey].children.append(cat)
            if self.nodes[cat].parent not in self.nodes:  # excluded chapter
                continue
            title = mimic_titles.get(code, str(r.long_title) if isinstance(r.long_title, str) else code)
            if code == cat:  # a 3-char code used directly as a diagnosis: keep as category-level leaf
                self.nodes[cat].n_admissions += int(r.n_admissions)
                self.nodes[cat].n_primary += int(r.n_primary)
                continue
            self.nodes[code] = Node(code, title, "code", parent=cat, n_admissions=int(r.n_admissions), n_primary=int(r.n_primary))
            self.nodes[cat].children.append(code)
        # propagate counts upwards
        for n in list(self.nodes.values()):
            if n.level == "code":
                p = n
                while p.parent is not None:
                    p = self.nodes[p.parent]
                    p.n_admissions += n.n_admissions
                    p.n_primary += n.n_primary
        # prune categories/blocks without any observed leaf (keeps menus to what MIMIC actually codes)
        for cat in [k for k, n in self.nodes.items() if n.level == "category"]:
            n = self.nodes[cat]
            if not n.children and n.n_admissions == 0:
                self.nodes[n.parent].children.remove(cat)
                del self.nodes[cat]
        for blk in [k for k, n in self.nodes.items() if n.level == "block"]:
            n = self.nodes[blk]
            if not n.children:
                self.nodes[n.parent].children.remove(blk)
                del self.nodes[blk]
        for ch in list(self.chapters):
            if not self.nodes[ch].children:
                del self.nodes[ch]
                self.chapters.remove(ch)

        # ---- ICD-9 -> ICD-10 crosswalk (MIMIC-CDM file, inverted; most frequent target wins) ----
        prevalence = {_norm(r.icd_code): int(r.n_admissions) for r in uni10.itertuples()}
        best: Dict[str, Tuple[str, Tuple[int, int]]] = {}
        with (open(VENDOR_ICD / "ICD10_to_9.csv") if (VENDOR_ICD / "ICD10_to_9.csv").exists() else __import__("io").StringIO()) as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                c10, c9, cnt = _norm(row[0]), _norm(row[1]), int(row[2])
                score = (prevalence.get(c10, 0), cnt)  # MIMIC prevalence first, crosswalk count second
                if c9 not in best or score > best[c9][1]:
                    best[c9] = (c10, score)
        self.icd9_to_10 = {k: v[0] for k, v in best.items()}
        self.icd9_titles = dict(zip(d_icd[d_icd["icd_version"] == "9"]["icd_code"].map(_norm), d_icd[d_icd["icd_version"] == "9"]["long_title"]))

    # ------------------------------------------------------------------ structure helpers
    def _block_for(self, cat: str) -> Optional[str]:
        for s, e, bkey in self._block_ranges:
            if s <= cat <= e and bkey in self.nodes:
                return bkey
        return None

    def options(self, key: Optional[str]) -> Dict[str, str]:
        """Children of `key` (None -> chapters) as {code: title} for a Choice question."""
        keys = self.chapters if key is None else self.nodes[key].children
        return {k: self.nodes[k].title for k in keys}

    def title(self, key: str) -> str:
        return self.nodes[key].title if key in self.nodes else key

    def level(self, key: str) -> Optional[str]:
        return self.nodes[key].level if key in self.nodes else None

    def path(self, code: str) -> Optional[Tuple[str, str, str, str]]:
        """(chapter, block, category, code) for an ICD-10 code; unknown codes resolve by category/range."""
        code = _norm(code)
        cat = code[:3]
        if code not in self.nodes and cat not in self.nodes:
            bkey = self._block_for(cat)
            if bkey is None:
                return None
            return (self.nodes[bkey].parent, bkey, cat, code)
        c = self.nodes.get(cat) or self.nodes.get(code)
        blk = c.parent
        return (self.nodes[blk].parent, blk, cat, code)

    def category_of(self, code: str) -> str:
        return _norm(code)[:3]

    def map_icd9(self, code9: str) -> Optional[str]:
        code9 = _norm(code9)
        if code9 in self.icd9_to_10:
            return self.icd9_to_10[code9]
        # fall back to shorter prefixes (e.g. 5409 -> 540)
        for n in (4, 3):
            c = code9[:n]
            if c in self.icd9_to_10:
                return self.icd9_to_10[c]
        return None

    def to_icd10(self, code: str, version: int) -> Optional[str]:
        return _norm(code) if int(version) == 10 else self.map_icd9(code)

    def summary(self) -> dict:
        lv = defaultdict(int)
        for n in self.nodes.values():
            lv[n.level] += 1
        return {"chapters": len(self.chapters), **dict(lv), "max_children": max(len(n.children) for n in self.nodes.values()), "unplaced_codes": self.n_unplaced}


# ---------------------------------------------------------------------------------------------
# MIMIC-CDM pathology keyword rules applied to ICD titles (mirrors the evaluators' name lists)
# ---------------------------------------------------------------------------------------------
PATHOLOGY_RULES = {
    "appendicitis": {
        "word": "appendicitis",
        "alternatives": [("appendi", ["gangren", "infect", "inflam", "abscess", "rupture", "necros", "perf"])],
        "gracious": [],
        "categories": {"K35", "K36", "K37"},
        "gracious_categories": {"K35", "K36", "K37", "K38"},
    },
    "cholecystitis": {
        "word": "cholecystitis",
        "alternatives": [("gallbladder", ["gangren", "infect", "inflam", "abscess", "necros", "perf"]), ("cholangitis", ["cholangitis"])],
        "gracious": [("acute gallbladder", ["disease", "attack"]), ("acute biliary", ["colic"])],
        "categories": {"K80", "K81"},  # ICD-9 574.0x / ICD-10 K80.0x-K80.1x code calculus *with* cholecystitis under K80
        "gracious_categories": {"K80", "K81", "K82", "K83"},
    },
    "pancreatitis": {
        "word": "pancreatitis",
        "alternatives": [("pancrea", ["gangren", "infect", "inflam", "abscess", "necros"])],
        "gracious": [],
        "categories": {"K85"},
        "gracious_categories": {"K85", "K86"},
    },
    "diverticulitis": {
        "word": "diverticulitis",
        "alternatives": [("diverticul", ["inflam", "infect", "abscess", "perf", "rupture"])],
        "gracious": [("acute colonic", ["perfor"]), ("sigmoid", ["perfor"]), ("sigmoid", ["colitis"])],
        "categories": {"K57"},
        "gracious_categories": {"K57", "K63"},
    },
    # ---- MIRA extension diseases (Ferber et al. 2026); rules mirror the MIMIC-CDM style above ----
    "pneumonia": {
        "word": "pneumonia",
        "alternatives": [("lung", ["abscess", "infect"]), ("legionnaires", ["legionnaires"])],
        "gracious": [("pneumonit", ["pneumonit"]), ("empyema", ["empyema"]), ("lower respiratory", ["infection"])],
        "categories": {"J12", "J13", "J14", "J15", "J16", "J17", "J18"},
        "gracious_categories": {"J12", "J13", "J14", "J15", "J16", "J17", "J18", "J69", "J85", "J86", "J22", "A48"},
    },
    "uti": {
        "word": r"(?:urinary tract infection|\bcystitis)",  # "cystitis" as a word: not inside "cholecystitis" or "dacryocystitis"
        "alternatives": [("pyelonephritis", ["pyelonephritis"]), ("urosepsis", ["urosepsis"]), ("urinary", ["infect"])],
        "gracious": [("prostatitis", ["prostatitis"]), ("urethritis", ["urethritis"]), ("kidney", ["infect", "abscess"])],
        "categories": {"N39", "N30", "N10"},
        "gracious_categories": {"N39", "N30", "N10", "N11", "N12", "N13", "N15", "N34", "N41"},
    },
    "pulmonary_embolism": {
        "word": "pulmonary embolism",
        "alternatives": [("pulmonary", ["embol", "thromboembol", "infarction"]), ("embolus of pulmonary artery", ["embolus"])],
        "gracious": [("pulmonary", ["thrombosis"])],
        "categories": {"I26"},
        "gracious_categories": {"I26", "I27"},
    },
    "pancreatic_cancer": {
        "word": None,
        "alternatives": [("pancrea", ["malignan", "carcinoma", "cancer", "adenocarcinoma"]), ("neoplasm of", ["pancreas"])],
        "gracious": [("pancrea", ["neoplasm", "tumor", "mass"])],
        "exclude": ["benign", "in situ"],
        "categories": {"C25"},
        "gracious_categories": {"C25", "C78", "D13", "D37", "K86"},
    },
    # ---- MIMIC_rewoo diseases (github.com/peterhan91/MIMIC_rewoo) and the MIMIC-III cardiac sets; same rule style ----
    "cholangitis": {
        "word": "cholangitis",
        "alternatives": [("bile duct", ["infect", "inflam", "abscess"]), ("biliary", ["sepsis", "infect"])],
        "gracious": [("choledocholithiasis", ["choledocholithiasis"]), ("bile duct", ["calculus", "obstruct", "stone"]), ("biliary", ["obstruct", "stricture"]), ("cholecystitis", ["cholecystitis"])],
        "exclude": ["sclerosing"],
        "categories": {"K83", "K80"},
        "gracious_categories": {"K83", "K80", "K81", "K82", "K85"},
    },
    "bowel_obstruction": {
        "word": None,
        "alternatives": [("obstruction", ["intestin", "bowel", "colon", "adhesion", "hernia", "postprocedural", "gallstone ileus"]), ("volvulus", ["volvulus"]), ("intussusception", ["intussusception"])],
        "gracious": [("ileus", ["ileus"]), ("impaction", ["intestin", "fecal", "feces"]), ("hernia", ["gangrene", "incarcerat", "strangulat"]), ("stenosis", ["intestin", "colon", "bowel", "anastomo"]), ("stricture", ["intestin", "colon", "bowel", "anastomo"])],
        "exclude": ["without obstruction", "urinary", "airway", "gastric outlet", "biliary", "bile duct"],
        "categories": {"K56"},
        "gracious_categories": {"K56", "K91", "K40", "K41", "K42", "K43", "K44", "K45", "K46", "K50", "K51"},
    },
    "meningitis": {
        "word": "meningitis",
        "alternatives": [("meningoencephalitis", ["meningoencephalitis"]), ("meninges", ["infect", "inflam"])],
        "gracious": [("encephalitis", ["encephalitis"]), ("meningism", ["meningism"]), ("central nervous system", ["infect"]), ("intracranial", ["abscess", "infect"])],
        "categories": {"G00", "G01", "G02", "G03", "A87", "A39"},
        "gracious_categories": {"G00", "G01", "G02", "G03", "A87", "A39", "G04", "G05", "G06", "B00", "B02", "A17", "B45", "B37"},
    },
    "pyelonephritis": {
        "word": "pyelonephritis",
        "alternatives": [("kidney", ["infect", "abscess"]), ("renal", ["abscess"]), ("pyonephrosis", ["pyonephrosis"])],
        "gracious": [("urinary tract infection", ["infection"]), ("urosepsis", ["urosepsis"]), ("cystitis", ["cystitis"]), ("tubulo-interstitial nephritis", ["nephritis"]), ("urinary", ["infect"])],
        "categories": {"N10", "N11", "N12"},
        "gracious_categories": {"N10", "N11", "N12", "N13", "N15", "N16", "N39", "N30"},
    },
    "sepsis": {
        "word": "sepsis",
        "alternatives": [("septic", ["septic"])],
        "gracious": [("bacteremia", ["bacteremia"]), ("systemic inflammatory response", ["systemic"]), ("infection following", ["procedure"])],
        "exclude": ["arthritis", "newborn", "puerperal"],
        "categories": {"A40", "A41", "R65"},
        "gracious_categories": {"A40", "A41", "R65", "R78", "T81", "A48", "B95", "B96"},
    },
    "subarachnoid_hemorrhage": {
        "word": None,
        "alternatives": [("subarachnoid", ["hemorrhage", "haemorrhage", "bleed"])],
        "gracious": [("intracranial", ["hemorrhage", "haemorrhage", "bleed"]), ("aneurysm", ["cerebral", "intracranial", "berry", "rupture"]), ("intracerebral", ["hemorrhage"]), ("subdural", ["hemorrhage"]), ("intraventricular", ["hemorrhage"])],
        "categories": {"I60"},
        "gracious_categories": {"I60", "S06", "I61", "I62", "I67", "Q28"},
    },
    "myocardial_infarction": {
        "word": r"(?:myocardial infarction|\bn?stemi)",  # "STEMI" as a word: not inside "systemic"
        "alternatives": [("infarction", ["myocard", "coronary", "wall"])],
        "gracious": [("acute coronary syndrome", ["syndrome"]), ("angina", ["unstable", "angina"]), ("coronary", ["thrombosis", "occlusion", "ischemi"]), ("ischemic heart", ["acute"]), ("ischaemic heart", ["acute"]), ("myocardial injury", ["injury"])],
        "exclude": ["old myocardial", "personal history", "subsequent episode of care"],
        "categories": {"I21", "I22"},
        "gracious_categories": {"I21", "I22", "I24", "I20", "I5A"},
    },
    "congestive_heart_failure": {
        "word": "heart failure",
        "alternatives": [("cardiac failure", ["failure"]), ("congestive", ["heart"]), ("ventricular failure", ["failure"])],
        "gracious": [("cardiomyopathy", ["cardiomyopathy"]), ("pulmonary edema", ["edema"]), ("pulmonary oedema", ["oedema"]), ("cardiogenic", ["shock"]), ("fluid overload", ["overload"]), ("cor pulmonale", ["pulmonale"]), ("hypervolemia", ["hypervolemia"])],
        "categories": {"I50", "I11", "I13"},
        "gracious_categories": {"I50", "I11", "I13", "I42", "I43", "J81", "R60", "E87", "I27", "I09"},
    },
    "aortic_stenosis": {
        "word": None,
        "alternatives": [("aortic", ["stenosis"])],
        "gracious": [("aortic", ["insufficiency", "regurgitation", "sclerosis", "valve disorder", "valve) disorder", "valve disease"])],  # ICD titles write "aortic (valve) insufficiency"
        "categories": {"I35", "I06", "Q23"},
        "gracious_categories": {"I35", "I06", "Q23", "I08", "I38", "I39"},
    },
    "mitral_regurgitation": {
        "word": None,
        "alternatives": [("mitral", ["regurgitation", "insufficiency", "incompetence"])],
        "gracious": [("mitral", ["disorder", "disease", "prolapse", "valve"])],
        "categories": {"I34", "I05", "I08"},
        "gracious_categories": {"I34", "I05", "I08", "I38", "I39", "Q23"},
    },
}


# Titles that name a disease the patient does not have now: a personal or family history, a sequela (ICD-10's "Sequelae of ..." and
# its 7th character S, "..., sequela") or an ICD-9 late effect. Checked on the whole title, before the "without" cut below.
NOT_CURRENT = ("personal history", "family history", "sequela", "late effect")


def pathology_match(title: str, pathology: str) -> Optional[str]:
    """'strict' / 'gracious' / None for an ICD title against MIMIC-CDM's pathology name rules."""
    rules = PATHOLOGY_RULES[pathology]
    t = title.lower()
    if any(x in t for x in NOT_CURRENT):
        return None
    t = re.split(r"\bwithout\b", t)[0]  # ICD titles negate with "without ..." (mirrors negex in MIMIC-CDM): "calculus of gallbladder without cholecystitis" is not cholecystitis
    if any(x in t for x in rules.get("exclude", [])):
        return None
    # no left word boundary: ICD titles use compound forms such as "Bronchopneumonia, unspecified organism" (J18.0 / 485),
    # which are the pathology itself; verified against d_icd_diagnoses that only those two titles change for any pathology
    if rules.get("word") and re.search(rf"{rules['word']}\b", t):
        return "strict"
    for loc, mods in rules["alternatives"]:
        if loc in t and any(m in t for m in mods):
            return "strict"
    for loc, mods in rules["gracious"]:
        if loc in t and any(m in t for m in mods):
            return "gracious"
    return None
