"""Instruction texts of the routing agent and the ICD scorer, as named prompt sets.

`default` reproduces the texts every run before 2026-10-01 used, byte for byte, so cached answers stay valid. `p1` to `p3` are
paraphrases written for the robustness evaluation (same meaning, different wording and sentence structure); option texts
(the actions, lab groups, imaging studies, ICD titles) and the treatment questions are never paraphrased.

Templates use str.format fields: {principal}, {title}, {code}, {descr}, {name}, {label}.
"""
from __future__ import annotations

from typing import Dict

DEFAULT: Dict[str, str] = {
    "route": (
        "You are the admitting physician working up this patient. The state holds everything documented "
        "so far (`history_of_present_illness`, `physical_examination`, `vital_signs`, `laboratory_results`, "
        "`imaging_results`, medications) and `workup_log` lists the steps already taken. Which single step "
        "should be taken next to reach the principal diagnosis efficiently? Do not repeat a step whose results are "
        "already documented. Choose `final_diagnosis` when the documented findings are sufficient to name the principal "
        "diagnosis with confidence, or when no further step would change it."
    ),
    "sufficient": (
        "Do the findings documented in the state suffice to name the principal diagnosis of this admission "
        "with high confidence, without any further examination, test or imaging?"
    ),
    "lab_group": "If laboratory tests are ordered next for this patient, from which group of tests should they be chosen?",
    "imaging": (
        "If one imaging study is ordered next for this patient, which study would be the most appropriate given the "
        "documented history and findings? Studies already obtained are listed in `imaging_results`; choose "
        "`none_indicated` if no further imaging would change the assessment."
    ),
    "lab_test": (
        "If one laboratory test is ordered next for this patient, which single test would be the most appropriate given the "
        "documented history and findings? Results already obtained are listed in `laboratory_results` and "
        "`microbiology_results`."
    ),
    "pe_noul": "If the patient is examined now, should the {descr} be examined and documented?",
    "vital_noul": "If vital signs are reviewed now, should the {descr} be reviewed?",
    "panel_noul": "Should the '{name}' panel ({descr}) be ordered now to work up this patient?",
    "lab_noul": "Should the laboratory test '{label}' be ordered now to work up this patient?",
    "micro_noul": "Should the microbiology test '{label}' be ordered now to work up this patient?",
    "item_true": "this specific test is clinically indicated now for this patient's presentation and its result would change the assessment",
    "item_false": "not indicated, redundant with results already documented, or not useful for the differential diagnosis",
    "principal": (
        "the principal diagnosis of this hospital admission, i.e. the condition established after study to be chiefly "
        "responsible for the presentation documented in the state (`history_of_present_illness`, `physical_examination`, "
        "`vital_signs`, `laboratory_results`, `microbiology_results`, `imaging_results`)"
    ),
    "dx_chapter": "Which ICD-10-CM chapter contains {principal}?",
    "dx_block": "The principal diagnosis of this admission lies in the ICD-10-CM chapter '{title}' ({code}). Which block within this chapter contains {principal}?",
    "dx_category": "The principal diagnosis of this admission lies in the ICD-10-CM block '{title}' ({code}). Which three-character category is {principal}?",
    "dx_leaf": "The principal diagnosis is in ICD-10-CM category {code} '{title}'. Which specific code within it best describes {principal}?",
    "dx_subgroup": "The principal diagnosis is in ICD-10-CM {code} '{title}'. Which sub-group or code within it best describes {principal}?",
    "dx_runoff": "Each of the following ICD-10-CM codes is a candidate for {principal}. Which one best describes it?",
}

P1: Dict[str, str] = {  # plain wording, instruction reordered
    "route": (
        "Pick the next step in this patient's diagnostic work-up. Everything found so far is recorded in the state, and "
        "`workup_log` shows what has already been done; do not choose a step whose results are already recorded. Aim to "
        "reach the principal diagnosis in as few steps as possible. Pick `final_diagnosis` if the findings already identify "
        "the principal diagnosis confidently or if no remaining step could change it."
    ),
    "sufficient": (
        "Is the principal diagnosis of this admission already clear with high confidence from the documented findings alone, "
        "so that no further examination, test or imaging is needed?"
    ),
    "lab_group": "Suppose laboratory tests are ordered next. Which group of tests should they come from?",
    "imaging": (
        "Suppose a single imaging study is ordered next. Given the history and findings so far, which study is most "
        "appropriate? Studies already done appear in `imaging_results`. Pick `none_indicated` if no further imaging would "
        "change the assessment."
    ),
    "lab_test": (
        "Suppose exactly one laboratory test is ordered next. Given the history and findings so far, which test is most "
        "appropriate? Results already available appear in `laboratory_results` and `microbiology_results`."
    ),
    "pe_noul": "During a physical examination now, should the {descr} be assessed and recorded?",
    "vital_noul": "When the vital signs are reviewed now, should the {descr} be included?",
    "panel_noul": "Is ordering the '{name}' panel ({descr}) indicated now for this patient's work-up?",
    "lab_noul": "Is the laboratory test '{label}' indicated now for this patient's work-up?",
    "micro_noul": "Is the microbiology test '{label}' indicated now for this patient's work-up?",
    "item_true": "the test is indicated by this presentation and its result could change the assessment",
    "item_false": "the test is not indicated, duplicates documented results, or would not help the differential diagnosis",
    "principal": (
        "the principal diagnosis of this admission, meaning the condition that, after study, is found to be chiefly "
        "responsible for the presentation recorded in the state"
    ),
    "dx_chapter": "In which ICD-10-CM chapter is {principal} classified?",
    "dx_block": "The principal diagnosis falls within ICD-10-CM chapter '{title}' ({code}). In which block of this chapter is {principal} classified?",
    "dx_category": "The principal diagnosis falls within ICD-10-CM block '{title}' ({code}). Which three-character category of this block is {principal}?",
    "dx_leaf": "The principal diagnosis falls within ICD-10-CM category {code} '{title}'. Which code in this category describes {principal} most accurately?",
    "dx_subgroup": "The principal diagnosis falls within ICD-10-CM {code} '{title}'. Which sub-group or code in it describes {principal} most accurately?",
    "dx_runoff": "The following ICD-10-CM codes are all candidates for {principal}. Which one describes it best?",
}

P2: Dict[str, str] = {  # formal, third person
    "route": (
        "Acting as the emergency physician responsible for this admission, determine the single most useful next action "
        "toward establishing the principal diagnosis. The documented history, examination, vital signs, laboratory and "
        "imaging results and medications are in the state; completed actions are listed in `workup_log` and should not be "
        "repeated. Select `final_diagnosis` once the available findings establish the principal diagnosis with confidence, "
        "or when further work-up would not alter it."
    ),
    "sufficient": (
        "Based solely on the findings documented so far, can the principal diagnosis of this admission be established with "
        "high confidence without additional examinations, laboratory tests or imaging?"
    ),
    "lab_group": "Should laboratory testing be the next step for this patient, which category of tests would be most appropriate to order from?",
    "imaging": (
        "Should imaging be performed next, which one study would be most appropriate in view of the documented history and "
        "findings? Previously obtained studies are recorded in `imaging_results`. Select `none_indicated` when additional "
        "imaging would not alter the assessment."
    ),
    "lab_test": (
        "Should a single laboratory test be ordered next, which one test would be most appropriate in view of the documented "
        "history and findings? Results already obtained are recorded in `laboratory_results` and `microbiology_results`."
    ),
    "pe_noul": "If a physical examination is performed at this point, is examination of the {descr} indicated?",
    "vital_noul": "If the vital signs are checked at this point, is review of the {descr} indicated?",
    "panel_noul": "Would the '{name}' panel ({descr}) be an appropriate order at this point in the work-up?",
    "lab_noul": "Would ordering the laboratory test '{label}' at this point be appropriate in working up this patient?",
    "micro_noul": "Would ordering the microbiology test '{label}' at this point be appropriate in working up this patient?",
    "item_true": "clinically warranted at this point for this presentation, with a result that would influence the assessment",
    "item_false": "not warranted, redundant given documented results, or uninformative for the differential diagnosis",
    "principal": (
        "this hospital admission's principal diagnosis, that is, the condition determined after work-up to be mainly "
        "responsible for the clinical presentation documented in the state (history, examination, vital signs, laboratory, "
        "microbiology and imaging results)"
    ),
    "dx_chapter": "Identify the ICD-10-CM chapter that contains {principal}.",
    "dx_block": "Given that the principal diagnosis belongs to ICD-10-CM chapter '{title}' ({code}), identify the block within this chapter that contains {principal}.",
    "dx_category": "Given that the principal diagnosis belongs to ICD-10-CM block '{title}' ({code}), identify the three-character category corresponding to {principal}.",
    "dx_leaf": "Given that the principal diagnosis belongs to ICD-10-CM category {code} '{title}', identify the specific code that best describes {principal}.",
    "dx_subgroup": "Given that the principal diagnosis belongs to ICD-10-CM {code} '{title}', identify the sub-group or code that best describes {principal}.",
    "dx_runoff": "Among the following candidate ICD-10-CM codes for {principal}, identify the one that describes it most accurately.",
}

P3: Dict[str, str] = {  # question first, concise
    "route": (
        "What should be done next for this patient? Choose the one step that most efficiently advances the work-up toward "
        "the principal diagnosis, given the findings in the state and the steps already listed in `workup_log` (never "
        "repeat one of those). If the findings are already enough to name the principal diagnosis with confidence, or no "
        "further step would change it, choose `final_diagnosis`."
    ),
    "sufficient": "Are the recorded findings enough to confidently name the principal diagnosis right now, with no more exams, tests or imaging?",
    "lab_group": "Which group of laboratory tests would be most useful to order next for this patient?",
    "imaging": (
        "Which single imaging study, if any, would best advance the assessment of this patient now? Studies already in "
        "`imaging_results` were done. Choose `none_indicated` if more imaging would not change anything."
    ),
    "lab_test": (
        "Which single laboratory test would best advance the assessment of this patient now? Results in "
        "`laboratory_results` and `microbiology_results` are already available."
    ),
    "pe_noul": "Should the {descr} be part of this patient's examination now?",
    "vital_noul": "Should the {descr} be reviewed now?",
    "panel_noul": "Order the '{name}' panel ({descr}) now for this patient?",
    "lab_noul": "Order the laboratory test '{label}' now for this patient?",
    "micro_noul": "Order the microbiology test '{label}' now for this patient?",
    "item_true": "useful now: indicated for this presentation and able to change the assessment",
    "item_false": "not useful now: not indicated, already covered by documented results, or irrelevant to the differential",
    "principal": (
        "the main condition responsible for this admission, as established after study from the documented history, "
        "examination, vital signs, laboratory, microbiology and imaging findings"
    ),
    "dx_chapter": "Which chapter of ICD-10-CM does {principal} belong to?",
    "dx_block": "Assume the principal diagnosis is in ICD-10-CM chapter '{title}' ({code}). Which block of that chapter does {principal} belong to?",
    "dx_category": "Assume the principal diagnosis is in ICD-10-CM block '{title}' ({code}). Which three-character category does {principal} belong to?",
    "dx_leaf": "Assume the principal diagnosis is in ICD-10-CM category {code} '{title}'. Which code of that category fits {principal} best?",
    "dx_subgroup": "Assume the principal diagnosis is in ICD-10-CM {code} '{title}'. Which sub-group or code of it fits {principal} best?",
    "dx_runoff": "Which of these candidate ICD-10-CM codes fits {principal} best?",
}

PROMPT_SETS: Dict[str, Dict[str, str]] = {"default": DEFAULT, "p1": P1, "p2": P2, "p3": P3}


def get_prompts(name: str = "default") -> Dict[str, str]:
    if name not in PROMPT_SETS:
        raise KeyError(f"unknown prompt set {name!r}; known: {sorted(PROMPT_SETS)}")
    p = PROMPT_SETS[name]
    missing = set(DEFAULT) - set(p)
    if missing:
        raise KeyError(f"prompt set {name!r} lacks {sorted(missing)}")
    return p
