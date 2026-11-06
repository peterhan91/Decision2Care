"""Routing menus: every information-gathering step is a typed multiple-choice decision.

Level 1  next_action      Choice  over information sources (exam, vitals, labs, imaging, meds, diagnose)
Level 2  lab_group        Choice  over laboratory groups (panels, blood chemistry, hematology, blood gas,
                                  urine, stool, CSF, body fluids, microbiology)
Level 3  item_<id>        Noul    per item inside the chosen group  -> multi-select of individual tests
         pe_<system>      Noul    per physical-examination system   -> which parts of the exam to perform
         vital_<name>     Noul    per vital-sign item
         imaging_study    Choice  over every (modality, region) study seen in the cohort

Menus are built from the MIMIC item universes written by `build_dataset.py`, so the option space is
"all items", not a hand-picked subset: every lab item ever measured in MIMIC-IV (d_labitems_min_1),
every microbiology test, every imaging study.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from decision2care.paths import CDM_DIR
from decision2care.vendor.mimic_cdm.tools.utils import ADDITIONAL_LAB_TEST_MAPPING, UNIQUE_MODALITY_TO_ORGAN_MAPPING
from decision2care.vendor.mimic_cdm.utils.nlp import convert_labs_to_itemid

# ------------------------------------------------------------------------------------------------
# Information sources (next_action options). Descriptions are neutral: no pathology hints.
# ------------------------------------------------------------------------------------------------
ACTIONS = {
    "ask_patient": "Ask the patient a question and hear the answer: the complaint and its course, associated symptoms, past medical and surgical history, medications, allergies, social and family history, or what the findings so far raise.",
    "physical_examination": "Examine the patient at the bedside and document the findings (you choose which organ systems to examine).",
    "vital_signs": "Review the recorded vital signs: temperature, heart rate, blood pressure, respiratory rate, oxygen saturation, pain score.",
    "laboratory_tests": "Order laboratory tests: blood, urine, stool or body-fluid chemistry, hematology, coagulation, blood gas, or microbiology (cultures, smears, PCR).",
    "imaging": "Order one imaging study (radiograph, ultrasound, CT, MRI, nuclear or endoscopic imaging) and read the radiology report.",
    "ecg": "Obtain a 12-lead electrocardiogram and read the ECG report.",
    "echocardiogram": "Obtain an echocardiogram (transthoracic or transesophageal) and read the echocardiography report.",
    "home_medications": "Review the patient's pre-admission home medication list.",
    "ed_medications": "Review the medications the patient already received in the emergency department.",
    "llm_suggest": "Ask a colleague which further tests, imaging or examination would add the most diagnostic evidence now; you decide which of the suggested orders to place.",
    "final_diagnosis": "Stop gathering information and commit to the final principal diagnosis.",
}

# ------------------------------------------------------------------------------------------------
# Physical examination systems (headers used in MIMIC discharge summaries)
# ------------------------------------------------------------------------------------------------
PE_SYSTEMS: Dict[str, Tuple[str, List[str]]] = {
    "vitals": ("vital signs recorded at the time of the exam", ["vitals", "vital signs", "vs", "vitals on admission", "admission vitals"]),
    "general": ("general appearance and distress", ["general", "gen", "constitutional", "appearance", "general appearance"]),
    "heent": ("head, eyes, ears, nose and throat", ["heent", "head", "eyes", "ent", "mouth", "oropharynx", "head/neck", "head and neck"]),
    "neck": ("neck, lymph nodes, thyroid, jugular veins", ["neck", "lymph", "lymphatics", "lymph nodes", "lad"]),
    "cardiovascular": ("heart sounds, rhythm, murmurs, pulses", ["cv", "cardiac", "cardiovascular", "heart", "cor", "cvs"]),
    "respiratory": ("lungs and breathing", ["lungs", "pulm", "pulmonary", "resp", "respiratory", "chest", "lung"]),
    "abdomen": ("abdominal inspection, palpation (tenderness, guarding, rebound, masses), bowel sounds", ["abd", "abdomen", "abdominal", "gi", "abdomen/pelvis", "abd/pelvis"]),
    "back": ("back, spine and flanks", ["back", "spine", "flank", "flanks", "cva"]),
    "genitourinary_rectal": ("genitourinary, pelvic and rectal examination", ["gu", "genitourinary", "rectal", "genital", "pelvic", "genitalia", "dre"]),
    "extremities": ("extremities, edema, joints, peripheral perfusion", ["ext", "extremities", "extremity", "msk", "musculoskeletal", "extr", "limbs"]),
    "skin": ("skin, wounds, rashes, jaundice", ["skin", "derm", "integument", "integumentary"]),
    "neurologic": ("mental status, cranial nerves, motor and sensory function", ["neuro", "neurologic", "neurological", "cns", "neuro exam"]),
    "psychiatric": ("mood, affect, orientation", ["psych", "psychiatric", "mental status", "psychiatry"]),
}

VITAL_ITEMS: Dict[str, str] = {
    "temperature": "body temperature",
    "heart_rate": "heart rate (pulse)",
    "blood_pressure": "systolic and diastolic blood pressure",
    "respiratory_rate": "respiratory rate",
    "oxygen_saturation": "peripheral oxygen saturation (SpO2)",
    "pain_score": "patient-reported pain score (0-10)",
}

# ------------------------------------------------------------------------------------------------
# Laboratory groups. Items are resolved from d_labitems_min_1.csv (every item measured >= 1 time).
# ------------------------------------------------------------------------------------------------
LAB_GROUPS: Dict[str, Tuple[str, Optional[Tuple[str, ...]], Optional[Tuple[str, ...]]]] = {
    # key: (description, fluids, categories); None = special handling
    "common_panels": ("Standard panels ordered as a bundle: complete blood count, basic/comprehensive metabolic panel, liver function tests, renal panel, electrolytes, coagulation, urinalysis, lipids, iron, thyroid, pregnancy test, inflammatory markers, pancreatic enzymes, lactate, cardiac troponin, blood gas, blood/urine/stool cultures.", None, None),
    "blood_chemistry": ("Individual blood chemistry tests (electrolytes, renal and liver function, enzymes, proteins, glucose, lipids, hormones, tumor markers, vitamins, drug levels, toxicology).", ("Blood",), ("Chemistry",)),
    "blood_hematology": ("Individual blood hematology and coagulation tests (cell counts, differential, indices, smear morphology, PT/INR/PTT, fibrinogen, hemoglobin variants).", ("Blood",), ("Hematology",)),
    "blood_gas": ("Blood gas analysis items (pH, pCO2, pO2, bicarbonate, base excess, lactate, oxygen saturation, co-oximetry).", ("Blood", "Fluid"), ("Blood Gas",)),
    "urine": ("Urine tests (urinalysis components, urine chemistry, urine microscopy, urine electrolytes, urine drug screen, urine pregnancy test).", ("Urine",), None),
    "stool": ("Stool tests (occult blood, fecal fat, stool electrolytes, stool white cells).", ("Stool",), None),
    "cerebrospinal_fluid": ("Cerebrospinal fluid tests (cell counts, protein, glucose, lactate).", ("Cerebrospinal Fluid",), None),
    "body_fluids": ("Ascites/peritoneal, pleural, joint and other body-fluid tests (cell counts, albumin, protein, amylase, lipase, LDH, pH, crystals).", ("Ascites", "Pleural", "Joint Fluid", "Other Body Fluid", "Bone Marrow", "Q", "I"), None),
    "microbiology": ("Microbiology: cultures (blood, urine, stool, wound, fluid, respiratory), Gram stains, C. difficile, MRSA screen, viral PCR, serology.", None, None),
}

# Panel definitions. Names that exist as labels in lab_test_mapping.pkl (MIMIC-CDM panels) resolve
# through it; the rest are explicit lab names resolved with MIMIC-CDM's fuzzy matcher.
PANELS: Dict[str, Tuple[str, List[str]]] = {
    "Complete Blood Count (CBC)": ("white cells with differential, hemoglobin, hematocrit, platelets, red cell indices", ["Complete Blood Count (CBC)"]),
    "Basic Metabolic Panel (BMP)": ("sodium, potassium, chloride, bicarbonate, urea nitrogen, creatinine, glucose, calcium", ["Basic Metabolic Panel (BMP)"]),
    "Comprehensive Metabolic Panel (CMP)": ("BMP plus liver enzymes, bilirubin, total protein, albumin", ["Comprehensive Metabolic Panel (CMP)"]),
    "Liver Function Panel (LFP)": ("ALT, AST, alkaline phosphatase, GGT, bilirubin, total protein, albumin, PT", ["Liver Function Panel (LFP)"]),
    "Renal Function Panel (RFP)": ("urea nitrogen, creatinine, electrolytes, phosphate, albumin, glucose", ["Renal Function Panel (RFP)"]),
    "Electrolyte Panel": ("sodium, potassium, chloride, bicarbonate", ["Electrolyte Panel"]),
    "Magnesium and Phosphate": ("serum magnesium and phosphate", ["Magnesium", "Phosphate"]),
    "Urinalysis": ("dipstick and microscopy: color, specific gravity, pH, protein, glucose, ketones, blood, nitrite, leukocytes, cells, casts, crystals", ["Urinalysis"]),
    "Coagulation Profile": ("PT, INR, PTT", ["Coagulation Profile"]),
    "Lipid Profile": ("total cholesterol, LDL, HDL, triglycerides", ["Lipid Profile"]),
    "Iron Studies": ("iron, TIBC, transferrin, ferritin", ["Iron Studies"]),
    "Thyroid Function Test (TFT)": ("TSH, T4, free T4, T3", ["Thyroid Function Test (TFT)"]),
    "Pregnancy Test": ("urine hCG (qualitative)", ["Pregnancy Test", "HCG, Serum, Qualitative", "HCG"]),
    "Inflammatory markers": ("C-reactive protein and erythrocyte sedimentation rate", ["C-Reactive Protein (CRP)", "Erythrocyte Sedimentation Rate (ESR)"]),
    "Pancreatic enzymes": ("serum lipase and amylase", ["Lipase", "Amylase"]),
    "Lactate": ("blood lactate", ["Lactate"]),
    "Cardiac troponin": ("troponin T and troponin I", ["Troponin T", "Troponin I"]),
    "Blood gas": ("pH, pCO2, pO2, base excess, calculated bicarbonate, oxygen saturation", ["Blood pH", "Blood pCO2", "Blood pO2", "Blood Base Excess", "Blood Calculated Bicarbonate, Whole Blood", "Blood Oxygen Saturation"]),
    "Lactate dehydrogenase (LDH)": ("serum LDH", ["Lactate Dehydrogenase (LD)"]),
    "D-Dimer": ("plasma D-dimer", ["D-Dimer"]),
    "Ammonia": ("plasma ammonia", ["Ammonia"]),
    "Hemoglobin A1c": ("glycated hemoglobin", ["% Hemoglobin A1c"]),
    "Ethanol and toxicology screen": ("blood ethanol, acetaminophen, salicylate, urine drug screen", ["Ethanol", "Acetaminophen", "Salicylate"]),
    "Blood Culture": ("aerobic and anaerobic blood cultures", ["Blood Culture"]),
    "Urine Culture": ("urine culture", ["URINE CULTURE"]),
    "Stool Culture and C. difficile": ("stool culture, C. difficile testing, ova and parasites", ["Stool Culture", "C. difficile PCR", "O&P"]),
}

# ------------------------------------------------------------------------------------------------
# Imaging studies: descriptions per (modality, region); menu built from imaging_universe.csv
# ------------------------------------------------------------------------------------------------
MODALITY_DESCR = {
    "CT": "computed tomography", "Ultrasound": "ultrasound (sonography, incl. Doppler)", "Radiograph": "plain radiograph (x-ray)",
    "MRI": "magnetic resonance imaging", "MRCP": "MR cholangiopancreatography", "HIDA": "hepatobiliary scintigraphy (HIDA scan)",
    "EUS": "endoscopic ultrasound", "ERCP": "endoscopic retrograde cholangiopancreatography (fluoroscopic images)", "CTU": "CT urography",
    "Upper GI Series": "fluoroscopic upper GI contrast study", "Fluoroscopy": "fluoroscopy", "MRA": "MR angiography", "MRE": "MR enterography",
    "Carotid ultrasound": "carotid duplex ultrasound", "Mammogram": "mammography", "PTC": "percutaneous transhepatic cholangiography",
}
REGION_DESCR = {
    "Abdomen": "abdomen and pelvis (liver, gallbladder, pancreas, bowel, kidneys, bladder, pelvic organs)", "Chest": "chest (lungs, heart, mediastinum)",
    "Head": "head and brain", "Neck": "neck (soft tissue, thyroid, vessels)", "Spine": "spine", "Venous": "extremity veins (venous duplex for thrombosis)",
    "Scrotum": "scrotum and testes", "Hip": "hip", "Knee": "knee", "Shoulder": "shoulder", "Ankle": "ankle", "Foot": "foot", "Hand": "hand", "Wrist": "wrist",
    "Bone": "bone survey", "Femur": "femur", "Thigh": "thigh", "Finger": "finger", "Heel": "heel",
}
EXCLUDED_MODALITIES = {"Drainage", "Paracentesis"}  # interventional procedures, not diagnostic imaging

# MIMIC d_labitems rows that are not diagnostic tests: specimen-quality indices (H/L/I = hemolysis, lipemia, icterus),
# tube holds, specimen-type/void markers, and opaque send-out or instrument codes (HPE1, UTX4, STX2, COV8MC ...).
# Jev reads labels literally and answers "order it" for names it cannot interpret, so these are removed from the menus.
ADMIN_LABEL_RE = re.compile(
    r"^(H|L|I|Uhold|XUCU|RFXUCU|ARCH-\d+|PAN\d*|Problem( Specimen)?|Voided Specimen|Specimen Type|Urine Specimen Type|"
    r"Urine Comments|Urine tube, held|Anat Path Hold|Test Name|Comments?|wbcp|voided|FOCUSED ANALYSIS|ISOLATE FOR MIC|Problem)$|\bhold\b|^(EE|HPE|UTX|UCU|STX|COV|FLUA|FLUB|CDT)\d+\w*$|GENE (CT|ENDPT)",
    re.I,
)
LABEL_WHITELIST = {"C3", "C4", "pO2", "pCO2", "pH"}
# An order-set marker without a result of its own (one row in MIMIC-IV). When tests are ordered one at a time Jev reads it as
# "a CBC" and spends an action on it, so the single-item menu leaves it out; the multi-select menus keep it, so that the cached
# answers of earlier runs stay valid.
ORDER_MARKER_LABELS = {"CBC Only"}


def is_diagnostic_item(label: str) -> bool:
    return label in LABEL_WHITELIST or not ADMIN_LABEL_RE.search(label.strip())


@dataclass
class LabItem:
    itemid: int
    label: str
    fluid: str
    category: str
    count: int


@dataclass
class Menus:
    lab_test_mapping: pd.DataFrame
    lab_items: Dict[int, LabItem]
    lab_groups: Dict[str, List[int]]  # group key -> itemids (microbiology test ids included)
    panels: Dict[str, List[int]]  # panel name -> itemids
    imaging: Dict[str, Tuple[str, str, str]]  # option key -> (modality, region, description)
    micro_items: Dict[int, LabItem] = field(default_factory=dict)
    item_class: Dict[int, Tuple[int, ...]] = field(default_factory=dict)  # itemid -> every itemid of the same test (see _test_classes)

    # -------------------------------------------------------------- lab helpers
    def group_options(self) -> Dict[str, str]:
        return {k: LAB_GROUPS[k][0] for k in LAB_GROUPS if k in self.lab_groups and self.lab_groups[k]}

    def tests_of(self, itemids: List[int]) -> Dict[str, List[int]]:
        """Distinct tests among `itemids`, in their order: {label of the test's first item: every itemid of that test}."""
        out: Dict[str, List[int]] = {}
        seen = set()
        for i in itemids:
            it = self.lab_items.get(i) or self.micro_items.get(i)
            if it is not None and it.label in ORDER_MARKER_LABELS:
                continue
            cls = self.item_class.get(i, (i,))
            if cls not in seen:
                seen.add(cls)
                out.setdefault(self.item_label(i), []).extend(cls)
        return out

    def item_label(self, itemid: int) -> str:
        it = self.lab_items.get(itemid) or self.micro_items.get(itemid)
        if it is None:
            return str(itemid)
        return f"{it.label} ({it.fluid}, {it.category})" if it.fluid else f"{it.label} (microbiology)"

    def imaging_options(self) -> Dict[str, str]:
        return {k: v[2] for k, v in self.imaging.items()}

    # -------------------------------------------------------------- reproducibility: freeze the imaging menu of a run
    def imaging_snapshot(self) -> list:
        """The imaging menu as a JSON-serialisable list (order matters: it is part of every routing request)."""
        return [[k, v[0], v[1], v[2]] for k, v in self.imaging.items()]

    def load_imaging_snapshot(self, path) -> None:
        """Replace the imaging menu with the one saved by an earlier run (outputs/<run>/imaging_menu.json), so that re-scoring
        that run reuses its exact routing options even after a cohort build changed the imaging universes."""
        import json

        rows = json.load(open(path))
        self.imaging = {r[0]: (r[1], r[2], r[3]) for r in rows}


def _split_fluid_prefix(name: str) -> str:
    return name


def _test_classes(mapping: pd.DataFrame, items: Dict[int, LabItem]) -> Dict[int, Tuple[int, ...]]:
    """itemid -> the itemids that are the same test, most measured first. MIMIC holds several ids for one test (three for
    creatinine, four for the white cell count); ordering tests one at a time would return 'not available' for the wrong id.
    Two ids are the same test if MIMIC-CDM's mapping lists them as corresponding and they share the fluid, or if they share
    fluid, category and the label up to a parenthetical ('Alanine Aminotransferase' has 69 rows in MIMIC-IV, 'Alanine
    Aminotransferase (ALT)' 1.8 million)."""
    parent = {i: i for i in items}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_with_label: Dict[Tuple[str, str, str], int] = {}
    for i, it in items.items():
        name = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", it.label)).strip().lower() or it.label
        parent[find(i)] = find(first_with_label.setdefault((name, it.fluid, it.category), i))
    for r in mapping[mapping["itemid"].notna()].itertuples():
        i = int(r.itemid)
        for j in (int(x) for x in r.corresponding_ids):
            if i in items and j in items and items[i].fluid == items[j].fluid:
                parent[find(j)] = find(i)
    members: Dict[int, List[int]] = {}
    for i in items:
        members.setdefault(find(i), []).append(i)
    out: Dict[int, Tuple[int, ...]] = {}
    for ids in members.values():
        cls = tuple(sorted(ids, key=lambda i: -items[i].count))
        out.update({i: cls for i in ids})
    return out


def load_menus(cdm_dir: Path = CDM_DIR, min_item_count: int = 1, min_imaging_reports: int = 10, extra_imaging_dirs: Optional[List[Path]] = None) -> Menus:
    """Menus from the MIMIC item universes in `cdm_dir`. The imaging menu is the union of `cdm_dir`'s imaging universe and
    those of `extra_imaging_dirs` (other cohort directories, e.g. data/cdm_ext for the extension cohorts), counts summed;
    without extra directories the menu is exactly the abdominal-cohort one."""
    with open(cdm_dir / "lab_test_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    items_df = pd.read_csv(cdm_dir / "d_labitems_min_1.csv")
    items = {
        int(r.itemid): LabItem(int(r.itemid), str(r.label), str(r.fluid) if isinstance(r.fluid, str) else "", str(r.category) if isinstance(r.category, str) else "", int(r.count))
        for r in items_df.itertuples()
        if r.count >= min_item_count and is_diagnostic_item(str(r.label))
    }
    logger.info(f"lab item universe: {len(items)} diagnostic items ({int((items_df['count'] >= min_item_count).sum()) - len(items)} administrative/code items excluded)")
    micro_df = pd.read_csv(cdm_dir / "micro_tests_universe.csv")
    micro = {int(r.itemid): LabItem(int(r.itemid), str(r.label), "", "Microbiology", int(r.count)) for r in micro_df.itertuples() if is_diagnostic_item(str(r.label))}

    groups: Dict[str, List[int]] = {}
    for key, (descr, fluids, cats) in LAB_GROUPS.items():
        if key == "common_panels":
            continue
        if key == "microbiology":
            groups[key] = sorted(micro, key=lambda i: -micro[i].count)
            continue
        sel = [it for it in items.values() if (fluids is None or it.fluid in fluids) and (cats is None or it.category in cats)]
        groups[key] = [it.itemid for it in sorted(sel, key=lambda x: -x.count)]
    covered = {i for k, v in groups.items() if k != "microbiology" for i in v}
    leftovers = [i for i in items if i not in covered]
    if leftovers:
        groups["body_fluids"].extend(leftovers)  # odd fluids ('Q', 'I', 'Fluid') land here
        logger.debug(f"{len(leftovers)} lab items with unusual fluid assigned to body_fluids")

    panels: Dict[str, List[int]] = {}
    labels = set(mapping["label"].tolist())
    for name, (descr, members) in PANELS.items():
        ids: List[int] = []
        for m in members:
            if m in labels:
                ids.extend(mapping.loc[mapping["label"] == m, "corresponding_ids"].iloc[0])
            else:
                res = convert_labs_to_itemid([m], mapping)
                ids.extend([x for x in res if isinstance(x, int)])
        ids = [i for i in dict.fromkeys(ids) if i in items or i in micro]
        if not ids:
            logger.warning(f"panel '{name}' resolved to no measured items")
            continue
        panels[name] = ids
    groups["common_panels"] = []  # handled through `panels`

    counts: Dict[Tuple[str, str], int] = {}  # insertion order: base universe rows first, then studies only seen in extra cohorts
    for d in [cdm_dir, *(extra_imaging_dirs or [])]:
        f = Path(d) / "imaging_universe.csv"
        if not f.exists():
            logger.warning(f"no imaging_universe.csv in {d}; skipped for the imaging menu")
            continue
        for r in pd.read_csv(f).itertuples():
            key = (str(r.modality), str(r.region))
            counts[key] = counts.get(key, 0) + int(r.n_reports)
    imaging: Dict[str, Tuple[str, str, str]] = {}
    for (modality, region), n in counts.items():
        if n < min_imaging_reports or modality in EXCLUDED_MODALITIES:
            continue
        if region not in REGION_DESCR or modality not in MODALITY_DESCR:
            continue
        imaging[f"{modality} {region}"] = (modality, region, f"{MODALITY_DESCR[modality]} of the {REGION_DESCR[region]}")
    if extra_imaging_dirs:
        logger.info(f"imaging menu: {len(imaging)} studies (union with {[str(d) for d in extra_imaging_dirs]})")
    return Menus(lab_test_mapping=mapping, lab_items=items, lab_groups=groups, panels=panels, imaging=imaging, micro_items=micro,
                 item_class=_test_classes(mapping, {**items, **micro}))


# ------------------------------------------------------------------------------------------------
# Physical-examination text splitter (best effort; falls back to the full text)
# ------------------------------------------------------------------------------------------------
_PE_HEADER_TO_SYSTEM = {h.lower(): sys for sys, (_, hs) in PE_SYSTEMS.items() for h in hs}
_PE_HEADER_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s;,.)\]]))(" + "|".join(sorted((re.escape(h) for h in _PE_HEADER_TO_SYSTEM), key=len, reverse=True)) + r")\s*[:\-–]\s*",
    re.IGNORECASE,
)


def split_physical_exam(text: str) -> Dict[str, str]:
    """Split a MIMIC physical-exam blob into {system: text}; {'full_examination': text} if it cannot."""
    text = (text or "").strip()
    if not text:
        return {}
    matches = list(_PE_HEADER_RE.finditer(text))
    if len(matches) < 3:
        return {"full_examination": text}
    out: Dict[str, str] = {}
    preamble = text[: matches[0].start()].strip()
    if preamble:
        out["general"] = preamble
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sys_key = _PE_HEADER_TO_SYSTEM[m.group(1).lower()]
        seg = text[m.end():end].strip(" ,;.")
        if not seg:
            continue
        out[sys_key] = (out[sys_key] + " | " + seg) if sys_key in out else seg
    return out or {"full_examination": text}
