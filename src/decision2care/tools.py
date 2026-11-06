"""Executable information-gathering tools over one MIMIC-CDM admission record, plus the structured
JSON state that is sent to Jev. Semantics follow MIMIC-CDM's `tools/Actions.py` (first matching
report per requested study, repeat requests walk forward chronologically, missing studies return
"not available") extended with vitals, medications and per-system physical examination.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from decision2care.menus import PE_SYSTEMS, VITAL_ITEMS, Menus, split_physical_exam
from decision2care.vendor.mimic_cdm.tools.utils import UNIQUE_TO_BROAD_MODALITY


HISTORY_FIELDS = {"dialogue": "History Dialogue", "dialogue_opus": "History Dialogue Opus"}  # tools.history -> the record field load_cohort fills


@dataclass
class ToolConfig:
    lab_format: str = "value_ref_flag"  # value | value_flag | value_ref_flag
    include_ed_medications: bool = True
    include_home_medications: bool = True
    include_vitals: bool = True
    include_demographics: bool = True  # master switch for the `patient` field
    include_sex: bool = True  # sex is normally visible in MIMIC HPIs ("Ms. ___", "M"); stating it avoids sex-inappropriate orders
    include_age: bool = False  # ages are censored in MIMIC notes; MIMIC-CDM models never saw them
    include_chief_complaint: bool = True
    hpi_cut_ed_course: bool = False  # drop the retrospective "In the ED ..." course from the HPI (stricter than MIMIC-CDM)
    max_state_chars: int = 90_000  # ~22k tokens; Jev allows 32k tokens for state + longest question
    max_report_chars: int = 6_000
    # robustness evaluation (2026-10-01); the defaults reproduce every earlier run
    include_hpi: bool = True  # False = history withheld (the on-premise paper's ungrounded patient agent)
    lab_panels: bool = True  # False = no pre-assembled panels, individual items only (MIRA's LabRequest design)
    state_field_order: str = "default"  # default | reversed (position sensitivity; the client salts the cache key)
    state_render: str = "json"  # json | text (the state as one text note with the same section names)
    # single-item ordering (2026-10-04): a laboratory action is a visit in which tests are ordered one at a time, each result
    # revealed before the next choice (agent._lab_visit). No panels and no multi-select, so every result in the state traces
    # back to one decision; the router sees the same laboratory action as in the multi-select mode
    lab_single_item: bool = False
    # radiology report sections (2026-10-04): findings = MIMIC-CDM's reports, IMPRESSION and other conclusion sections removed;
    # mira = the reports in MIRA's format, impression kept, with the disease name masked as MIRA's Methods describe
    # (decision2care/build_radiology_mira.py writes them next to the records). configs/default.yaml sets `mira`; the value here
    # stays `findings` so that the configs of earlier runs, which lack the key, keep replaying those runs.
    # mira_unmasked (2026-10-19) = the same reports without the masks: MIRA's released builder discards them, so this is the text
    # MIRA-v2's agents read (build_radiology_mira --unmasked), for comparisons with MIRA-v2 on its inputs
    radiology_sections: str = "findings"
    # laboratory values (2026-10-04): corrected = as decision2care/fix_lab_units.py leaves the records, a unit that is the text "N/A"
    # dropped ("NEG", not "NEG N/A"), as in MIMIC-CDM's own data, where pandas reads that unit as missing; as_built = the values
    # as the cohort builder wrote them, the default here and in configs/default.yaml (the baseline `micro_full`); the configs of
    # earlier runs lack the key
    lab_values: str = "as_built"
    # radiology reports of the stay (2026-10-04): as_built = the reports MIMIC-CDM's rules keep, with the labels as built;
    # full = also the entries decision2care/build_radiology_full.py adds to the records (marked `Added`: vein ultrasounds whose exam
    # name has no modality word, reports that are empty without the impression) and its corrected labels. A report without
    # text in the format of `radiology_sections` is never served. `as_built` is the default here and in configs/default.yaml
    # (the baseline `micro_full`); the configs of earlier runs lack the key
    radiology_notes: str = "as_built"
    # microbiology results (2026-10-04): full = every result of the stay, which the records hold once
    # decision2care/build_microbiology.py has completed them; as_built = the field as the cohort builder made it, cultures that grew
    # an organism only (it dropped every result without one, such as "NO GROWTH."): build_microbiology keeps it next to the
    # records. configs/default.yaml sets `full`; the value here stays `as_built` so that the configs of earlier runs, which
    # lack the key, keep replaying those runs
    microbiology: str = "as_built"
    # the history (2026-10-06): record = the history of present illness as the record holds it (every run before that date);
    # dialogue = the history taken in a dialogue with a patient agent that keeps the emergency-department course to itself, as
    # MIRA and the on-premise paper take it (decision2care/history_dialogue.py writes history_dialogue.json next to the records); the
    # dialogue takes the place of the history of present illness. dialogue = Jev asks from a fixed menu; dialogue_opus = an Opus
    # doctor with MIRA-v2's physician prompt asks in free text (history_dialogue_opus.json); interactive = the history is taken
    # during the work-up: asking the patient is one of Jev's actions (`ask_patient`), Opus proposes the questions and Jev chooses
    # (decision2care/interactive.py), and the dialogue so far takes the place of the history of present illness
    history: str = "record"
    max_questions: int = 15  # the questions the patient answers during the work-up (they are not counted in agent.max_rounds)
    # (2026-10-06) Jev's two optional tools, each an action it may choose or ignore: ask_patient = go back to the patient after a
    # dialogue history (dialogue | dialogue_opus; always on with history=interactive), the questions proposed by Opus
    # (decision2care/interactive.py); llm_suggest = Opus suggests further orders from Jev's menu, never a diagnosis, and Jev decides on
    # each (decision2care/llm_suggest.py), at most `max_suggests` calls per case
    ask_patient: bool = False
    llm_suggest: bool = False
    max_suggests: int = 2
    # llm_suggest's orders: True = Jev answers a typed Noul per suggested order and only the endorsed ones are carried out (the
    # smoke run tools_smoke); False = they are carried out as Opus names them, Jev deciding only when to call the tool (the
    # user's choice of 2026-10-06; configs/default.yaml). True here keeps the configs of earlier runs, which lack the key, replaying
    suggest_gate: bool = True
    suggest_mode: str = "broad"  # broad = up to three orders (cmp_record_tools) | decisive = the one test that would most change the diagnosis
    suggest_candidate: bool = False  # llm_suggest also names its most likely diagnosis as an ICD-10-CM code, one more option of Jev's final run-off
    # the benchmark masks the name of the admission's disease in every text with four underscores; a language model infers a
    # diagnosis behind the blank, Jev reads underscores. True = the state says what the blank stands for (not which diagnosis)
    mask_note: bool = False
    # ablation (2026-10-13): False = the source is not in the record, as tools.include_vitals and include_home_medications do it:
    # the action is not offered and the state has no field for it. include_labs covers laboratory tests and microbiology,
    # include_imaging every study
    include_exam: bool = True
    include_labs: bool = True
    include_imaging: bool = True


HPI_WITHHELD = "not available: no history was obtained for this patient"
FOLLOWUP_HEADER = "Questions to the patient during the work-up:"  # tools.ask_patient with the record's history
SECOND_LOOK_INTRO = "A colleague took a second look at this chart and noted: "  # a note of the revision (decision2care/llm_revision.py)
MASK_NOTE = ("In this record a blank of four underscores (____) stands for the name of a diagnosis that was removed from the text; "
             "three underscores (___) stand for a removed name, date or number.")


def render_state_text(st: Dict[str, Any]) -> str:
    """The JSON state as one plain-text note: a `name:` header per field, list items as dashed lines."""

    def fmt(v: Any) -> str:
        if isinstance(v, dict):
            return "\n".join(f"- {k}: {x}" for k, x in v.items())
        if isinstance(v, list):
            lines = []
            for x in v:
                if isinstance(x, dict) and "study" in x and "report" in x:
                    lines.append(f"- {x['study']}: {x['report']}")
                elif isinstance(x, dict):
                    lines.append("- " + "; ".join(f"{k}: {y}" for k, y in x.items()))
                else:
                    lines.append(f"- {x}")
            return "\n".join(lines)
        return str(v)

    return "\n\n".join(f"{k}:\n{fmt(v)}" for k, v in st.items())


_ED_START = re.compile(r"\b(?:in|at|to|on arrival (?:to|at)|upon arrival (?:to|at)|on presentation to) the (?:ED|ER|E\.D\.|emergency (?:department|room))\b|\bED course\b", re.I)
_ED_END = re.compile(r"\b(Past Medical History|PMH|Past Surgical History|Social History|Family History|Review of Systems|ROS|On (?:arrival|transfer) to the floor|On the floor|Upon arrival to the floor|Vitals on transfer)\b\s*:?", re.I)


def cut_ed_course(text: str) -> str:
    """Remove the retrospective emergency-department course (vitals, labs, imaging, treatment) from an HPI."""
    m = _ED_START.search(text)
    if not m:
        return text
    e = _ED_END.search(text, m.end())
    tail = text[e.start():] if e else ""
    return (text[: m.start()].rstrip() + " [emergency department course omitted] " + tail).strip()


def _flag_for(value: str, lo, hi, mimic_flag: Optional[str]) -> Tuple[str, str]:
    """(flag, ref) strings computed in code (Jev is not asked to compare numbers)."""
    ref = ""
    flag = ""
    has_rr = lo is not None and hi is not None and not (isinstance(lo, float) and math.isnan(lo)) and not (isinstance(hi, float) and math.isnan(hi))
    if has_rr:
        ref = f"ref {lo:g}-{hi:g}"
        try:
            v = float(str(value).split()[0])
            flag = "LOW" if v < lo else ("HIGH" if v > hi else "normal")
        except (ValueError, IndexError):
            flag = ""
    if not flag and isinstance(mimic_flag, str) and mimic_flag:
        flag = mimic_flag.upper()
    return flag, ref


class PatientCase:
    """One admission: hidden record + what the agent has uncovered so far."""

    def __init__(self, hadm_id: int, rec: Dict[str, Any], menus: Menus, cfg: Optional[ToolConfig] = None):
        self.hadm_id = hadm_id
        self.rec = rec
        self.menus = menus
        self.cfg = cfg or ToolConfig()
        if self.cfg.radiology_sections == "mira" and not rec.get("MIRA Reports Loaded"):
            raise ValueError("tools.radiology_sections=mira needs radiology_mira.json in the cohort directory (python -m decision2care.build_radiology_mira "
                             "--cohort-dir <dir>); for MIMIC-CDM's findings-only reports pass --set tools.radiology_sections=findings")
        if self.cfg.radiology_sections == "mira_unmasked" and not rec.get("MIRA Unmasked Reports Loaded"):
            raise ValueError("tools.radiology_sections=mira_unmasked needs radiology_mira_unmasked.json in the cohort directory "
                             "(python -m decision2care.build_radiology_mira --unmasked --cohort-dir <dir>)")
        if self.cfg.lab_values == "corrected":
            if not rec.get("Laboratory Units Corrected"):  # set by fix_lab_units
                raise ValueError("tools.lab_values=corrected needs records processed by python -m decision2care.fix_lab_units --cohort-dir <dir>; "
                                 "for the values as built pass --set tools.lab_values=as_built")
            self.labs: Dict[int, str] = rec["Laboratory Tests"]
        elif self.cfg.lab_values == "as_built":
            self.labs = {**rec["Laboratory Tests"], **(rec.get("Laboratory Tests As Built") or {})}
        else:
            raise ValueError(f"tools.lab_values must be corrected or as_built, not {self.cfg.lab_values!r}")
        notes = rec.get("Radiology") or []
        if self.cfg.radiology_notes == "full":
            if not rec.get("Radiology Complete"):  # set by build_radiology_full
                raise ValueError("tools.radiology_notes=full needs records completed by python -m decision2care.build_radiology_full --cohort-dir <dir> "
                                 "(then build_radiology_mira); for the reports as built pass --set tools.radiology_notes=as_built")
        elif self.cfg.radiology_notes == "as_built":
            notes = [({**x, "Modality": x["Modality As Built"]} if "Modality As Built" in x else x) for x in notes if not x.get("Added")]
        else:
            raise ValueError(f"tools.radiology_notes must be full or as_built, not {self.cfg.radiology_notes!r}")
        # only reports with text in the format served: an impression-only report has none as findings
        self.mira_field = {"mira": "Report MIRA", "mira_unmasked": "Report MIRA Unmasked"}.get(self.cfg.radiology_sections)
        self.radiology: List[dict] = [x for x in notes if ((x.get(self.mira_field) if self.mira_field else None) or x.get("Report") or "").strip()]
        complete = bool(rec.get("Microbiology Complete"))  # set by build_microbiology: the record holds every result of the stay
        if self.cfg.microbiology == "full":
            if not complete:
                raise ValueError("tools.microbiology=full needs records with the complete microbiology results (python -m decision2care.build_microbiology "
                                 "--cohort-dir <dir>); for the field as built, cultures with an organism only, pass --set tools.microbiology=as_built")
            self.micro: Dict[int, str] = rec.get("Microbiology") or {}
        elif self.cfg.microbiology == "as_built":
            if complete and "Microbiology As Built" not in rec:
                raise ValueError("tools.microbiology=as_built: the records hold the complete results and microbiology_as_built.json was not loaded from the "
                                 "cohort directory (python -m decision2care.build_microbiology --cohort-dir <dir> writes it again)")
            self.micro = (rec["Microbiology As Built"] if complete else rec.get("Microbiology")) or {}
        else:
            raise ValueError(f"tools.microbiology must be full or as_built, not {self.cfg.microbiology!r}")
        if self.cfg.history in HISTORY_FIELDS:
            if not rec.get(HISTORY_FIELDS[self.cfg.history]):
                raise ValueError(f"tools.history={self.cfg.history} needs the case's interview in the cohort directory (python -m "
                                 "decision2care.history_dialogue --all [--doctor opus]); for the record's history pass --set tools.history=record")
        elif self.cfg.history not in ("record", "interactive"):
            raise ValueError(f"tools.history must be record, dialogue, dialogue_opus or interactive, not {self.cfg.history!r}")
        # history=interactive: the dialogue so far, opened by the patient as in MIRA (the questions come from decision2care/interactive.py)
        self.interview: List[dict] = []
        # the encounter step by step in the order it happened, {"kind": patient | result | note, "log": the workup_log lines of
        # the step, "text": what it brought}: the chart as Opus's question writer reads it (decision2care/interactive.py)
        self.events: List[dict] = []
        self.notes: List[str] = []  # the revision's notes (decision2care/llm_revision.py), shown to Jev as `second_look`
        self.ask_declined_at: Optional[Tuple[int, int]] = None  # Jev chose none of the proposed questions at this chart version
        self.suggests_used = 0  # llm_suggest calls so far
        self.suggested_dx: List[str] = []  # tools.suggest_candidate: Opus's ICD-10-CM codes, options of the final run-off
        self.suggest_declined_at: Optional[Tuple[int, int]] = None  # Jev ordered none of the suggestions at this chart version
        if self.cfg.history == "interactive":
            from decision2care.history_dialogue import starter

            self.interview.append({"id": "starter", "doctor": None, "patient": starter(rec)})
        elif self.cfg.ask_patient and self.cfg.history in HISTORY_FIELDS:  # the dialogue of the history goes on
            self.interview = [dict(t) for t in rec[HISTORY_FIELDS[self.cfg.history] + " Turns"]]
        elif self.cfg.ask_patient:  # the record's history: the patient agent answers from it; its opening line only frames
            from decision2care.history_dialogue import starter  # the conversation, the state keeps the record's text and adds the questions

            self.interview = [{"id": "starter", "doctor": None, "patient": starter(rec)}]
        if self.cfg.history == "record" and self.interview:
            self.events.append({"kind": "patient", "log": [], "text": "History of present illness:\n" + self._record_history()})
        else:
            for t in self.interview:
                self.events.append({"kind": "patient", "log": [], "text": (f"Doctor: {t['doctor']}\n" if t.get("doctor") else "") + f"Patient: {t['patient']}"})
        self.n_interview_start = len(self.interview)
        if not self.cfg.include_labs:  # ablation: no laboratory or microbiology result to find
            self.labs, self.micro = {}, {}
        if not self.cfg.include_imaging:
            self.radiology = []
        self.pe_sections: Dict[str, str] = split_physical_exam(rec.get("Physical Examination", "")) if self.cfg.include_exam else {}
        self.vitals_sets: List[dict] = rec.get("Vital Signs", []) if self.cfg.include_vitals else []
        self.vitals_cursor = 0
        self.already_requested_scans: Dict[str, int] = {}
        # uncovered information
        self.pe_seen: Dict[str, str] = {}
        self.vitals_seen: List[dict] = []
        self.labs_seen: Dict[int, str] = {}
        self.micro_seen: Dict[int, str] = {}
        self.imaging_seen: List[dict] = []
        self.home_meds_seen = False
        self.ed_meds_seen = False
        self.unavailable: List[str] = []
        self.exhausted_studies: set = set()
        self.requested_itemids: set = set()  # every lab/micro item ever requested (found or not)
        self.ordered_panels: set = set()
        self.exhausted_lab_groups: set = set()
        self.imaging_exhausted = False  # set when Jev answers that no further imaging study is indicated
        # ECG / echocardiography reports (MIMIC-III cohorts imported from MIMIC_rewoo; MIMIC-IV records have none)
        self.ecg_reports: List[dict] = list(rec.get("ECG") or [])
        self.echo_reports: List[dict] = list(rec.get("Echo") or [])
        self.ecg_seen: List[str] = []
        self.echo_seen: List[str] = []
        self.log: List[str] = []
        self.trajectory: List[dict] = []

    # ------------------------------------------------------------------ availability
    def available_actions(self) -> Dict[str, str]:
        from decision2care.menus import ACTIONS

        acts: Dict[str, str] = {}
        if self.interview and self.questions_left() and self.ask_declined_at != self.chart_version():
            acts["ask_patient"] = ACTIONS["ask_patient"]
        if self.pe_sections and any(s not in self.pe_seen for s in self.pe_sections):
            acts["physical_examination"] = ACTIONS["physical_examination"]
        if self.vitals_cursor < len(self.vitals_sets):
            acts["vital_signs"] = ACTIONS["vital_signs"]
        if self.cfg.include_labs and self.available_lab_groups():
            acts["laboratory_tests"] = ACTIONS["laboratory_tests"]
        if self.cfg.include_imaging and self.remaining_imaging() and not self.imaging_exhausted:
            acts["imaging"] = ACTIONS["imaging"]
        if len(self.ecg_seen) < len(self.ecg_reports):
            acts["ecg"] = ACTIONS["ecg"]
        if len(self.echo_seen) < len(self.echo_reports):
            acts["echocardiogram"] = ACTIONS["echocardiogram"]
        if self.cfg.include_home_medications and self.rec.get("Home Medications") and not self.home_meds_seen:
            acts["home_medications"] = ACTIONS["home_medications"]
        if self.cfg.include_ed_medications and self.rec.get("ED Medications") and not self.ed_meds_seen:
            acts["ed_medications"] = ACTIONS["ed_medications"]
        if self.cfg.llm_suggest and self.suggests_used < self.cfg.max_suggests and self.suggest_declined_at != self.chart_version():
            acts["llm_suggest"] = ACTIONS["llm_suggest"]
        acts["final_diagnosis"] = ACTIONS["final_diagnosis"]
        return acts

    def remaining_items(self, group: str) -> List[int]:
        """Items of a lab group not yet requested (panels: names not yet ordered are handled by the agent)."""
        return [i for i in self.menus.lab_groups.get(group, []) if i not in self.requested_itemids]

    def remaining_panels(self) -> List[str]:
        return [n for n in self.menus.panels if n not in self.ordered_panels and not set(self.menus.panels[n]) <= self.requested_itemids]

    def remaining_tests(self, group: str) -> Dict[str, List[int]]:
        """Single-item ordering: the distinct tests of a lab group not requested yet, {test name: its itemids}."""
        return self.menus.tests_of(self.remaining_items(group))

    def available_lab_groups(self) -> Dict[str, str]:
        opts = {}
        panels = self.cfg.lab_panels and not self.cfg.lab_single_item  # a panel is a bundle of tests
        for k, d in self.menus.group_options().items():
            if k in self.exhausted_lab_groups:
                continue
            if k == "common_panels":
                if panels and self.remaining_panels():
                    opts[k] = d
            elif self.remaining_items(k):
                opts[k] = d
        if panels and "common_panels" not in opts and "common_panels" not in self.exhausted_lab_groups and self.remaining_panels():
            opts = {"common_panels": "Standard panels ordered as a bundle (CBC, metabolic panels, liver tests, coagulation, urinalysis, cultures, ...)", **opts}
        return opts

    def remaining_imaging(self) -> Dict[str, str]:
        return {k: d for k, (m, r, d) in self.menus.imaging.items() if k not in self.exhausted_studies}

    def unseen_pe_systems(self) -> List[str]:
        return [s for s in self.pe_sections if s not in self.pe_seen]

    def questions_left(self) -> int:
        """The questions the patient still answers during the work-up (the dialogue before it is not counted)."""
        return max(0, self.cfg.max_questions - (len(self.interview) - self.n_interview_start)) if self.interview else 0

    def add_event(self, n_log: int, text: str) -> None:
        """Record a step's result for the chart: the workup_log lines written since `n_log`, then the result."""
        self.events.append({"kind": "result", "log": self.log[n_log:], "text": text})

    def evidence_count(self) -> int:
        """Answers and results so far (the patient's opening line included), notes not counted."""
        return sum(e["kind"] != "note" for e in self.events)

    def chart_version(self) -> Tuple[int, int]:
        return self.evidence_count(), len(self.notes)

    def decline_ask(self) -> None:
        """Jev chose none of the proposed questions: asking leaves the menu until the chart changes."""
        self.ask_declined_at = self.chart_version()

    def decline_suggest(self) -> None:
        """Nothing was ordered: with the gate (Jev endorsed none of the orders) llm_suggest leaves the menu until the chart
        changes; without it (Opus had nothing new to suggest) for the rest of the case."""
        self.suggest_declined_at = self.chart_version()
        if not self.cfg.suggest_gate:
            self.suggests_used = self.cfg.max_suggests

    def add_note(self, note: str) -> None:
        self.notes.append(note)
        self.events.append({"kind": "note", "log": [], "text": SECOND_LOOK_INTRO + note})

    # ------------------------------------------------------------------ tools
    def do_ask_patient(self, question: str, answer: str) -> str:
        """history=interactive: one question and the patient's answer (the patient agent is decision2care/history_dialogue.PatientAgent)."""
        self.interview.append({"id": f"q{len(self.interview)}", "doctor": question, "patient": answer})
        self.log.append(f"Asked the patient: {question}")
        self._record("Ask Patient", {"question": question}, f"answer {len(answer)} chars")
        obs = f"Doctor: {question}\nPatient: {answer}"
        self.events.append({"kind": "patient", "log": [], "text": obs})
        return obs

    def do_physical_exam(self, systems: List[str]) -> str:
        systems = [s for s in systems if s in self.pe_sections and s not in self.pe_seen] or self.unseen_pe_systems()
        for s in systems:
            self.pe_seen[s] = self.pe_sections[s]
        names = ", ".join(systems)
        self.log.append(f"Physical examination: {names}")
        self._record("Physical Examination", {"systems": systems}, f"{len(systems)} systems")
        return "\n".join(f"{s}: {self.pe_sections[s]}" for s in systems)

    def do_vitals(self, items: List[str]) -> str:
        if self.vitals_cursor >= len(self.vitals_sets):
            self.log.append("Vital signs: no further measurements available")
            self._record("Vital Signs", {"items": items}, "none")
            return "No further vital sign measurements are available."
        full = self.vitals_sets[self.vitals_cursor]
        self.vitals_cursor += 1
        keep = [k for k in items if k in VITAL_ITEMS] or list(VITAL_ITEMS)
        shown = {"time": full.get("time", "")}
        for k in keep:
            if k in full:
                shown[k] = full[k]
        for k in ("rhythm",):
            if k in full:
                shown[k] = full[k]
        self.vitals_seen.append(shown)
        self.log.append(f"Vital signs reviewed ({shown['time']}): {', '.join(k for k in shown if k != 'time')}")
        self._record("Vital Signs", {"items": keep, "set": shown["time"]}, ", ".join(f"{k}={v}" for k, v in shown.items() if k != "time"))
        return "; ".join(f"{k}: {v}" for k, v in shown.items())

    def lab_line(self, itemid: int) -> Optional[str]:
        labs = self.labs
        if itemid not in labs:
            return None
        it = self.menus.lab_items.get(itemid)
        label = it.label if it else str(itemid)
        fluid = it.fluid if it else ""
        value = labs[itemid]
        lo = self.rec.get("Reference Range Lower", {}).get(itemid)
        hi = self.rec.get("Reference Range Upper", {}).get(itemid)
        flag, ref = _flag_for(value, lo, hi, self.rec.get("Laboratory Flags", {}).get(itemid))
        line = f"({fluid}) {label}: {value}" if fluid else f"{label}: {value}"
        extras = []
        if self.cfg.lab_format in ("value_flag", "value_ref_flag") and flag:
            extras.append(flag)
        if self.cfg.lab_format == "value_ref_flag" and ref:
            extras.append(ref)
        if extras:
            line += " [" + "; ".join(extras) + "]"
        return line

    def micro_line(self, test_itemid: int) -> Optional[str]:
        if test_itemid not in self.micro:
            return None
        it = self.menus.micro_items.get(test_itemid)
        label = it.label if it else str(test_itemid)
        return f"{label}: {self.micro[test_itemid]}"

    def do_labs(self, itemids: List[int], requested_names: List[str], panel_names: Optional[List[str]] = None, single: bool = False) -> str:
        """`single`: `itemids` are the ids of one test; the log says whether it was measured, and ids without a result are not listed."""
        found, missing, n_new = [], [], 0
        for i in dict.fromkeys(itemids):
            if i in self.labs:
                if i not in self.labs_seen:
                    self.labs_seen[i] = self.lab_line(i)
                    n_new += 1
                found.append(self.labs_seen[i])
            elif i in self.micro:
                if i not in self.micro_seen:
                    self.micro_seen[i] = self.micro_line(i)
                    n_new += 1
                found.append(self.micro_seen[i])
            else:
                missing.append(self.menus.item_label(i))
        self.requested_itemids.update(itemids)
        self.ordered_panels.update(panel_names or [])
        names = ", ".join(requested_names)
        suffix = f"{n_new} new results" if n_new else "no new results"
        if single:
            self.log.append(f"Laboratory test ordered: {names} ({'result received' if found else 'not available for this patient'})")
        else:
            self.log.append(f"Laboratory tests ordered: {names} ({suffix})")
        self._record("Laboratory Tests", {"action_input": list(dict.fromkeys(itemids)), "requested": requested_names, "n_new": n_new}, f"{len(found)} found ({n_new} new), {len(missing)} not available")
        if single:
            return "\n".join(found) if found else f"Not available for this patient: {names}"
        out = found[:]
        if missing:
            missing = list(dict.fromkeys(missing))
            shown = missing[:15]
            more = f" and {len(missing) - 15} more" if len(missing) > 15 else ""
            out.append(f"Not available for this patient: {', '.join(shown)}{more}")
        return "\n".join(out) if out else "No results."

    def do_imaging(self, study_key: str) -> str:
        modality, region, _ = self.menus.imaging[study_key]
        requested = f"{region} {modality}"
        idx = self.already_requested_scans.get(requested, 0)
        result = None
        for rad in self.radiology:
            if (rad["Modality"] == modality or UNIQUE_TO_BROAD_MODALITY.get(rad["Modality"]) == modality) and rad["Region"] == region:
                if idx == 0:
                    self.already_requested_scans[requested] = self.already_requested_scans.get(requested, 0) + 1
                    report = ((rad.get(self.mira_field) if self.mira_field else None) or rad["Report"]).strip()
                    if len(report) > self.cfg.max_report_chars:
                        report = report[: self.cfg.max_report_chars] + " ..."
                    self.imaging_seen.append({"study": f"{modality} {region}", "exam": rad.get("Exam Name", ""), "report": report})
                    self.log.append(f"Imaging: {modality} {region} (report received)")
                    self._record("Imaging", {"action_input": {"modality": modality, "region": region}}, f"report {len(report)} chars")
                    return f"{modality} {region}: {report}"
                idx -= 1
                result = "cannot be repeated (no further study of this type was performed)"
        if result is None:
            result = "not available (no such study was performed for this patient)"
        self.exhausted_studies.add(study_key)
        self.unavailable.append(f"{modality} {region}: {result}")
        self.log.append(f"Imaging: {modality} {region} - {result}")
        self._record("Imaging", {"action_input": {"modality": modality, "region": region}}, result)
        return f"{modality} {region}: {result}"

    def _next_report(self, reports: List[dict], seen: List[str], tool: str, label: str) -> str:
        report = str(reports[len(seen)].get("Report", "")).strip()
        if len(report) > self.cfg.max_report_chars:
            report = report[: self.cfg.max_report_chars] + " ..."
        seen.append(report)
        self.log.append(f"{label} obtained (report {len(seen)} of {len(reports)})")
        self._record(tool, {"action_input": None}, f"report {len(report)} chars")
        return f"{label}: {report}"

    def do_ecg(self) -> str:
        return self._next_report(self.ecg_reports, self.ecg_seen, "ECG", "ECG")

    def do_echo(self) -> str:
        return self._next_report(self.echo_reports, self.echo_seen, "Echocardiogram", "Echocardiogram")

    def do_home_meds(self) -> str:
        self.home_meds_seen = True
        meds = self.rec.get("Home Medications", [])
        self.log.append(f"Home medication list reviewed ({len(meds)} medications)")
        self._record("Home Medications", {}, f"{len(meds)} medications")
        return "\n".join(self._med_str(m) for m in meds) or "No home medications recorded."

    def do_ed_meds(self) -> str:
        self.ed_meds_seen = True
        meds = self.rec.get("ED Medications", [])
        self.log.append(f"Emergency-department medications reviewed ({len(meds)} administrations)")
        self._record("ED Medications", {}, f"{len(meds)} administrations")
        return "\n".join(f"{m['name']} ({m['time']})" for m in meds) or "No medications were given in the emergency department."

    @staticmethod
    def _med_str(m: dict) -> str:
        return f"{m['name']} [{m['class']}]" if m.get("class") else m["name"]

    def _record(self, tool: str, tool_input: dict, summary: str) -> None:
        self.trajectory.append({"round": len(self.trajectory) + 1, "tool": tool, "tool_input": tool_input, "observation": summary})

    # ------------------------------------------------------------------ state for Jev
    def state(self) -> Dict[str, Any]:
        r = self.rec
        st: Dict[str, Any] = {}
        if self.cfg.mask_note:
            st["record_note"] = MASK_NOTE
        if self.cfg.include_demographics:
            parts = []
            if self.cfg.include_age and r.get("Age") is not None:
                parts.append(f"{r['Age']}-year-old")
            if self.cfg.include_sex and r.get("Sex"):
                parts.append(str(r["Sex"]))
            if parts:
                st["patient"] = " ".join(parts) + (" patient" if not self.cfg.include_sex or not r.get("Sex") else "")
        if self.cfg.include_chief_complaint and r.get("ED Chief Complaint"):
            st["chief_complaint"] = r["ED Chief Complaint"]
        st["history_of_present_illness"] = self.history_text()
        if self.cfg.include_exam:
            st["physical_examination"] = dict(self.pe_seen) if self.pe_seen else "not examined yet"
        if self.cfg.include_vitals:
            st["vital_signs"] = self.vitals_seen if self.vitals_seen else ("not reviewed yet" if self.vitals_sets else "no emergency-department vital signs on record")
        if self.cfg.include_labs:
            st["laboratory_results"] = list(self.labs_seen.values()) if self.labs_seen else "no laboratory tests ordered yet"
        if self.micro_seen:
            st["microbiology_results"] = list(self.micro_seen.values())
        if self.cfg.include_imaging:
            st["imaging_results"] = [{"study": x["study"], "report": x["report"]} for x in self.imaging_seen] if self.imaging_seen else "no imaging ordered yet"
        if self.ecg_reports:  # fields exist only for records that carry such reports (MIMIC-III cohorts)
            st["ecg_results"] = list(self.ecg_seen) if self.ecg_seen else "no ECG obtained yet"
        if self.echo_reports:
            st["echocardiography_results"] = list(self.echo_seen) if self.echo_seen else "no echocardiogram obtained yet"
        if self.cfg.include_home_medications:
            meds = r.get("Home Medications", [])
            st["home_medications"] = [self._med_str(m) for m in meds] if self.home_meds_seen else ("not reviewed yet" if meds else "no home medication list on record")
        if self.ed_meds_seen:
            st["emergency_department_medications"] = [f"{m['name']} ({m['time']})" for m in r.get("ED Medications", [])]
        if self.unavailable:
            st["studies_not_available"] = list(self.unavailable)
        st["workup_log"] = [f"{i + 1}. {l}" for i, l in enumerate(self.log)] if self.log else "nothing done yet"
        if self.notes:
            st["second_look"] = [SECOND_LOOK_INTRO + n for n in self.notes]
        st = self._fit_budget(st)
        if self.cfg.state_field_order == "reversed":
            st = dict(reversed(list(st.items())))
        if self.cfg.state_render == "text":
            return render_state_text(st)
        return st

    def _record_history(self) -> str:
        if self.cfg.include_hpi:
            hpi = self.rec["Patient History"].strip()
            return cut_ed_course(hpi) if self.cfg.hpi_cut_ed_course else hpi
        return HPI_WITHHELD

    def history_text(self) -> str:
        """The history as the state shows it (`history_of_present_illness`); the same text as without tools.ask_patient until a
        question is asked."""
        from decision2care.history_dialogue import render

        if self.cfg.history == "interactive" or (self.interview and self.cfg.history in HISTORY_FIELDS):
            return render(self.interview)
        if self.cfg.history in HISTORY_FIELDS:
            return self.rec[HISTORY_FIELDS[self.cfg.history]]
        asked = self.interview[self.n_interview_start:]
        return self._record_history() + (f"\n\n{FOLLOWUP_HEADER}\n" + render(asked) if asked else "")

    def _fit_budget(self, st: Dict[str, Any]) -> Dict[str, Any]:
        import json

        def size(x):
            return len(json.dumps(x, ensure_ascii=False))

        if size(st) <= self.cfg.max_state_chars:
            return st
        # shrink imaging reports first, then lab lines (oldest first), then the exam
        if isinstance(st.get("imaging_results"), list):
            for x in st["imaging_results"]:
                if size(st) > self.cfg.max_state_chars:
                    x["report"] = x["report"][:1500] + " ..."
        if isinstance(st.get("laboratory_results"), list):
            while size(st) > self.cfg.max_state_chars and len(st["laboratory_results"]) > 40:
                st["laboratory_results"] = st["laboratory_results"][1:]
        if size(st) > self.cfg.max_state_chars:
            st["history_of_present_illness"] = st["history_of_present_illness"][:4000] + " ..."
        return st

    # ------------------------------------------------------------------ ground truth
    def ground_truth(self) -> Dict[str, Any]:
        codes = self.rec.get("ICD Codes", [])
        primary = next((c for c in codes if c["seq_num"] == 1), codes[0] if codes else None)
        return {
            "pathology": self.rec.get("Pathology"),
            "primary_code": primary["code"] if primary else None,
            "primary_version": primary["version"] if primary else None,
            "primary_title": primary["title"] if primary else None,
            "all_codes": [(c["code"], c["version"]) for c in codes],
            "discharge_diagnosis": self.rec.get("Discharge Diagnosis", ""),
        }
