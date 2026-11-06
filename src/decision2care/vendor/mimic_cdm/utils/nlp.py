"""Trimmed copy of MIMIC-CDM `utils/nlp.py` (MIT, Hager et al. 2024).

Kept: negation-aware keyword checks (scispaCy + negspacy), lab-name -> itemid resolution,
primary-diagnosis extraction, and lab-result string formatting. Dropped: LLM tokenizer
utilities (transformers / tiktoken / exllama) and stop-word helpers that only served
free-text parsing of LLM output.
"""
from __future__ import annotations

import copy
import logging
import re
import string
from typing import List, Optional

import pandas as pd
from thefuzz import fuzz, process

from decision2care.vendor.mimic_cdm.tools.utils import FLUID_MAPPING, itemid_to_field

_NLP = None
UNMATCHED_LAB_NAMES: List[str] = []  # replaces the original `no_canonical_names.txt` side file


def get_nlp():
    """Lazily load `en_core_sci_lg` with the negex pipe (first call takes ~10 s)."""
    global _NLP
    if _NLP is None:
        import spacy
        from negspacy.negation import Negex  # noqa: F401  (registers the "negex" factory)

        _NLP = spacy.load("en_core_sci_lg")
        _NLP.add_pipe("negex", config={"chunk_prefix": ["no"]}, last=True)
    return _NLP


def treatment_alternative_procedure_checker(operation_keywords, text):
    for alternative_operations in operation_keywords:
        op_loc = alternative_operations["location"]
        for op_mod in alternative_operations["modifiers"]:
            for sentence in text.split("."):
                if keyword_positive(sentence, op_loc) and keyword_positive(sentence, op_mod):
                    return True
    return False


def keyword_positive(sentence: str, keyword: str) -> bool:
    """True if `keyword` occurs in `sentence` and is not negated (negex on scispaCy entities)."""
    doc = get_nlp()(sentence)
    for e in doc.ents:
        if keyword.lower() in e.text.lower():
            return not e._.negex
    return keyword.lower() in sentence.lower()


def remove_punctuation(input_string: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)


def contains(keyword: str, strings: List[str]) -> bool:
    return any(keyword_positive(s, keyword) for s in strings)


def diagnosis_checker(discharge_diagnosis: str, icd_diagnoses: List[str], keyword: str) -> bool:
    diags = copy.deepcopy(icd_diagnoses)
    diags.append(discharge_diagnosis)
    return contains(keyword, diags)


def procedure_checker(valid_procedures: List, done_procedures: List):
    for valid_procedure in valid_procedures:
        if isinstance(valid_procedure, int):
            if valid_procedure in done_procedures:
                return True
        else:
            for done_procedure in done_procedures:
                if keyword_positive(done_procedure, valid_procedure):
                    return True
    return False


def extract_short_and_long_name(test: str):
    match = re.search(r"\((.*)\)", test)
    if match:
        short_name = match.group(1).strip()
        long_name = test.replace(match.group(0), "").strip()
        long_name = long_name.replace("  ", " ").replace(" ,", ",")
        return short_name, long_name
    return test, test


def match_fluid(test: str):
    for fluid, keywords in FLUID_MAPPING.items():
        for keyword in keywords:
            if keyword.lower() in test.lower():
                return fluid, re.sub(keyword, "", test, flags=re.IGNORECASE).strip()
    return None, None


def convert_labs_to_itemid(tests: List[str], lab_test_mapping_df: pd.DataFrame):
    """Fuzzy-map free-text lab names to the expanded list of MIMIC itemids (MIMIC-CDM logic)."""
    logging.getLogger().setLevel(logging.ERROR)
    labels = lab_test_mapping_df["label"].tolist()
    all_tests = []
    for test_full in tests:
        fluid, test_no_fluid = match_fluid(test_full)
        test_short, test_long = extract_short_and_long_name(test_full)
        test_match, score = process.extractOne(test_full, labels, scorer=fuzz.ratio)
        if score < 90:
            test_match, score = process.extractOne(test_long, labels, scorer=fuzz.ratio)
            if score < 90:
                test_match, score = process.extractOne(test_short, labels, scorer=fuzz.ratio)
                if score < 100:
                    if fluid:
                        labels_fluid = lab_test_mapping_df[lab_test_mapping_df["fluid"] == fluid]["label"].tolist()
                        test_match, score = process.extractOne(test_no_fluid, labels_fluid, scorer=fuzz.ratio)
                        if score < 90:
                            test_match = ""
                            UNMATCHED_LAB_NAMES.append(test_full)
                    else:
                        test_match = ""
                        UNMATCHED_LAB_NAMES.append(test_full)
        if test_match:
            expanded_tests = lab_test_mapping_df.loc[lab_test_mapping_df["label"] == test_match, "corresponding_ids"].iloc[0]
            if fluid:
                expanded_tests = [
                    t
                    for t in expanded_tests
                    if (itemid_to_field(t, "fluid", lab_test_mapping_df) == fluid)
                    or (itemid_to_field(t, "fluid", lab_test_mapping_df) != itemid_to_field(t, "fluid", lab_test_mapping_df))
                ]
            all_tests.extend(expanded_tests)
        else:
            all_tests.append(test_full)
    logging.getLogger().setLevel(logging.INFO)
    return all_tests


def extract_primary_diagnosis(text: str) -> str:
    """First diagnosis mentioned in a discharge-diagnosis block (scispaCy noun chunks / entities)."""
    nlp = get_nlp()
    earliest_keyword_index = len(text)
    doc = nlp(text)
    diag = check_ents_for_diagnosis_noun_chunks(doc)
    if diag:
        earliest_keyword_index = min(earliest_keyword_index, text.find(diag))
    diag = check_ents_for_diagnosis_entities(doc)
    if diag:
        earliest_keyword_index = min(earliest_keyword_index, text.find(diag))
    for line in text.split("\n"):
        doc = nlp(line)
        diag = check_ents_for_diagnosis_noun_chunks(doc)
        if diag:
            earliest_keyword_index = min(earliest_keyword_index, text.find(diag))
            break
        diag = check_ents_for_diagnosis_entities(doc)
        if diag:
            earliest_keyword_index = min(earliest_keyword_index, text.find(diag))
            break
    prim_diag = text[earliest_keyword_index:]
    prim_diag = prim_diag.split("\n")[0]
    prim_diag = prim_diag.split(",")[0]
    prim_diag = re.split(r"\band\b", prim_diag)[0]
    prim_diag = re.split(r"\bor\b", prim_diag)[0]
    prim_diag = re.split(r"\bvs[.]?\b", prim_diag)[0]
    return prim_diag.strip()


def check_ents_for_diagnosis_entities(doc) -> Optional[str]:
    for e in doc.ents:
        d = e.text.lower()
        if "primary" not in d and "diagnosis" not in d and "diagnoses" not in d and "dx" not in d and d != "active" and d != "acute":
            return e.text
    return None


def check_ents_for_diagnosis_noun_chunks(doc) -> Optional[str]:
    for chunk in doc.noun_chunks:
        d = chunk.text.lower()
        d = re.sub(r"^[^a-zA-Z]+", "", d)
        if "primary" not in d and "diagnosis" not in d and "diagnoses" not in d and "dx" not in d and d != "active" and d != "acute":
            return d.strip()
    return None


def create_lab_test_string(
    test_id,
    lab_test_mapping_df,
    hadm_info,
    include_ref_range=False,
    bin_lab_results=False,
    bin_lab_results_abnormal=False,
    only_abnormal_labs=False,
):
    """Original MIMIC-CDM formatting: "(fluid) label: value[ | RR: [lo - hi]]\\n"."""
    lab_test_fluid = itemid_to_field(test_id, "fluid", lab_test_mapping_df)
    lab_test_label = itemid_to_field(test_id, "label", lab_test_mapping_df)
    lab_test_value = hadm_info["Laboratory Tests"].get(test_id, "N/A")
    rr_lower = hadm_info["Reference Range Lower"].get(test_id, None)
    rr_upper = hadm_info["Reference Range Upper"].get(test_id, None)
    has_rr = rr_lower == rr_lower and rr_upper == rr_upper and rr_lower is not None and rr_upper is not None
    if only_abnormal_labs and has_rr:
        try:
            v = float(str(lab_test_value).split()[0])
            if rr_lower < v < rr_upper:
                return ""
        except ValueError:
            pass
    if bin_lab_results_abnormal and has_rr:
        try:
            v = float(str(lab_test_value).split()[0])
            lab_test_value = "Abnormal" if (v < rr_lower or v > rr_upper) else "Normal"
        except ValueError:
            pass
    if bin_lab_results and has_rr:
        try:
            v = float(str(lab_test_value).split()[0])
            lab_test_value = "Low" if v < rr_lower else ("High" if v > rr_upper else "Normal")
        except ValueError:
            pass
    lab_test_str = f"({lab_test_fluid}) {lab_test_label}: {lab_test_value}"
    if include_ref_range and has_rr:
        lab_test_str += f" | RR: [{rr_lower} - {rr_upper}]"
    return lab_test_str + "\n"
