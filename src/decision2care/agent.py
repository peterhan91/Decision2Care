"""Zero-shot clinical decision-making agent whose every step is a typed Jev routing decision.

Per round (one Jev request, speculative fan-out):
  next_action            Choice over the information sources still available
  sufficient_for_dx      Noul  (recorded; the routing itself follows next_action)
  pe_<system>            Noul  per unexamined organ system     (used if next_action == physical_examination)
  vital_<item>           Noul  per vital-sign item             (used if next_action == vital_signs)
  lab_group              Choice over laboratory groups          (used if next_action == laboratory_tests)
  imaging_study          Choice over every remaining study      (used if next_action == imaging)
Then, for laboratory tests, a second request asks one Noul per item of the chosen group
(or per panel), i.e. the individual tests are selected from the full MIMIC item list.
With `agent.lab_visit` the laboratory action is a visit: the results of the order are revealed and
Jev orders again on the updated state until it endorses nothing more.
With `tools.lab_single_item` the visit orders tests one at a time: one Choice ranks the distinct
tests of the chosen group and the Noul decides whether the best of them is indicated.
With `tools.history=interactive` (or `tools.ask_patient` after a dialogue history) asking the patient is one of the
actions (`ask_patient`): the interviewer (decision2care/interactive.py) has Opus propose questions, Jev choose one or none, and
the patient agent answer. Questions do not count as tool actions (`tools.max_questions` bounds them). With
`tools.llm_suggest`, `llm_suggest` has Opus suggest further orders and Jev decide on each (decision2care/llm_suggest.py). `run(case, resume=...)` continues a finished work-up, as the
revision does after a note (decision2care/llm_revision.py).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from decision2care.client import choice, noul
from decision2care.menus import PANELS, PE_SYSTEMS, VITAL_ITEMS, Menus
from decision2care.prompts import DEFAULT as _DEFAULT_PROMPTS
from decision2care.prompts import get_prompts
from decision2care.tools import PatientCase


@dataclass
class AgentConfig:
    max_rounds: int = 10
    noul_threshold: float = 0.5
    max_items_per_order: int = 40
    max_panels_per_order: int = 12
    # multi-select laboratory: stay in the laboratory and order again on the updated state until no panel or item is endorsed
    # (`_lab_visit`); False = one order per laboratory action, as in every run before 2026-10-04
    lab_visit: bool = False
    stop_when_only_final: bool = True
    prompt_set: str = "default"  # instruction wording (decision2care.prompts); paraphrase sets p1-p3 for the robustness evaluation


# Default instruction texts (aliases of decision2care.prompts.DEFAULT, kept for importers); the agent reads its prompt set.
ROUTE_INSTRUCTIONS = _DEFAULT_PROMPTS["route"]
SUFFICIENT_INSTRUCTIONS = _DEFAULT_PROMPTS["sufficient"]
LAB_GROUP_INSTRUCTIONS = _DEFAULT_PROMPTS["lab_group"]
IMAGING_INSTRUCTIONS = _DEFAULT_PROMPTS["imaging"]
NO_IMAGING_OPTION = {"none_indicated": "No further imaging study is indicated for this patient at this point."}
MAX_CHOICE_OPTIONS = 255  # a Jev Choice holds at most 255 options
TEST_RUNOFF_PER_CHUNK = 20
TEST_GATE_TOP = 3  # single-item ordering: the Noul is asked for this many of the most probable tests before the rest of the group
LAB_VISIT_MAX_ORDERS = 20  # multi-select laboratory visit: safety bound on the orders of one visit


def _yes(ans: Dict[str, Any], qid: str, thr: float) -> bool:
    a = ans.get(qid)
    return a is not None and a.get("type") == "noul" and a["noul"] >= thr


class RoutingAgent:
    def __init__(self, client, menus: Menus, cfg: AgentConfig = AgentConfig(), interviewer=None, suggester=None):
        self.client = client
        self.menus = menus
        self.cfg = cfg
        self.P = get_prompts(cfg.prompt_set)
        self.interviewer = interviewer  # ask_patient: decision2care.interactive.Interviewer
        self.suggester = suggester  # llm_suggest: decision2care.llm_suggest.Suggester

    # ------------------------------------------------------------------ question builders
    def _round_questions(self, case: PatientCase, actions: Dict[str, str]) -> Dict[str, dict]:
        q: Dict[str, dict] = {
            "next_action": choice(self.P["route"], actions),
            "sufficient_for_diagnosis": noul(self.P["sufficient"]),
        }
        if "physical_examination" in actions:
            for s in case.unseen_pe_systems():
                descr = PE_SYSTEMS[s][0] if s in PE_SYSTEMS else "complete examination"
                q[f"pe_{s}"] = noul(self.P["pe_noul"].format(descr=descr))
        if "vital_signs" in actions:
            for k, descr in VITAL_ITEMS.items():
                q[f"vital_{k}"] = noul(self.P["vital_noul"].format(descr=descr))
        if "laboratory_tests" in actions:
            opts = case.available_lab_groups()
            if len(opts) >= 2:
                q["lab_group"] = choice(self.P["lab_group"], opts)
        if "imaging" in actions:
            rem = case.remaining_imaging()
            if rem:
                q["imaging_study"] = choice(self.P["imaging"], {**rem, **NO_IMAGING_OPTION})
        return q

    def _item_questions(self, group: str, case: PatientCase) -> Dict[str, dict]:
        """One Noul per not-yet-requested item (or panel) of the group."""
        q: Dict[str, dict] = {}
        if group == "common_panels":
            for name in case.remaining_panels():
                descr = PANELS[name][0]
                q[f"panel::{name}"] = noul(self.P["panel_noul"].format(name=name, descr=descr))
            return q
        crit = {"true": self.P["item_true"], "false": self.P["item_false"]}
        for itemid in case.remaining_items(group):
            label = self.menus.item_label(itemid)
            if group == "microbiology":
                q[f"item::{itemid}"] = noul(self.P["micro_noul"].format(label=label), crit)
            else:
                q[f"item::{itemid}"] = noul(self.P["lab_noul"].format(label=label), crit)
        return q

    def _drop_redundant_panels(self, selected: List[str]):
        """Remove selected panels whose item set is contained in another selected panel (higher noul wins ties)."""
        names = [q.split("::", 1)[1] for q in selected]
        sets = {n: set(self.menus.panels[n]) for n in names}
        keep: List[str] = []
        dropped: List[str] = []
        for n in names:  # `selected` is sorted by noul, descending
            if any(sets[n] < sets[m] for m in names if m != n) or any(sets[n] == sets[k] for k in keep):
                dropped.append(n)
            else:
                keep.append(n)
        return [f"panel::{n}" for n in keep], dropped

    # ------------------------------------------------------------------ single-item laboratory ordering
    def _choose_test(self, state, names: List[str], group: str) -> Dict[str, Any]:
        """One Choice over test names (a name without a description is read as it stands). A group with more tests than one
        Choice holds is asked in chunks in one request, then a run-off among the best tests of each chunk."""
        tag = f"agent.test.{group}"
        if len(names) > MAX_CHOICE_OPTIONS:
            n_chunks = math.ceil(len(names) / MAX_CHOICE_OPTIONS)
            size = math.ceil(len(names) / n_chunks)
            qs = {f"chunk{i}": choice(self.P["lab_test"], dict.fromkeys(names[i * size:(i + 1) * size])) for i in range(n_chunks)}
            ans = self.client.ask(state, qs, tag=tag)
            keep = {k for a in ans.values() for k, _ in sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:TEST_RUNOFF_PER_CHUNK]}
            names = [k for k in names if k in keep]
        return self.client.ask(state, {"lab_test": choice(self.P["lab_test"], dict.fromkeys(names))}, tag=tag)["lab_test"]

    def _order_one_test(self, case: PatientCase, state, ranked_groups: List[str], tests_log: List[dict], retired: List[str]) -> Optional[str]:
        """One single-test order on `state`. One Choice ranks the distinct tests of the most probable lab group; a Choice is
        relative, so whether a test is indicated at all is asked with the Noul of the multi-select mode, and the most probable
        endorsed test is ordered. The Noul is asked for the `TEST_GATE_TOP` most probable tests first and for the rest of the
        group only if none of those is endorsed, so a group is retired only when no test in it is indicated (the multi-select
        rule); the next group by probability is then asked. Returns the observation, or None when every group is retired."""
        crit = {"true": self.P["item_true"], "false": self.P["item_false"]}
        for g in ranked_groups:
            tests = case.remaining_tests(g)
            if not tests:
                case.exhausted_lab_groups.add(g)
                continue
            a = self._choose_test(state, list(tests), g)
            probs = a["probabilities"]
            ranked = sorted(tests, key=lambda k: -probs.get(k, 0.0))  # ties: the more frequently measured test (menu order)
            wording = self.P["micro_noul" if g == "microbiology" else "lab_noul"]
            nouls: Dict[str, float] = {}
            name = None
            for part in (ranked[:TEST_GATE_TOP], ranked[TEST_GATE_TOP:]):
                if name is None and part:
                    gate = self.client.ask(state, {f"test::{k}": noul(wording.format(label=k), crit) for k in part}, tag=f"agent.test_gate.{g}")
                    nouls.update({k: gate[f"test::{k}"]["noul"] for k in part})
                    name = next((k for k in part if nouls[k] >= self.cfg.noul_threshold), None)
            if name is None:  # no test of this group is indicated for this patient
                case.exhausted_lab_groups.add(g)
                retired.append(g)
                continue
            obs = case.do_labs(tests[name], [name], single=True)
            tests_log.append({
                "group": g, "test": name, "rank": ranked.index(name) + 1, "p": round(probs.get(name, 0.0), 3), "noul": nouls[name],
                "confidence": a.get("confidence"), "n_itemids": len(tests[name]), "n_new": case.trajectory[-1]["tool_input"].get("n_new"),
                "top": [(k, round(probs.get(k, 0.0), 3), nouls[k]) for k in ranked[:TEST_GATE_TOP]],  # (test, Choice probability, Noul)
            })
            return obs
        return None

    def _order_endorsed(self, case: PatientCase, state, ranked_groups: List[str], orders_log: List[dict], retired: List[str]) -> Optional[str]:
        """One multi-select order on `state`: cascade through the groups by probability and order every endorsed panel or item
        (Noul >= threshold) of the first group in which Jev endorses at least one. A group without an endorsed item is retired.
        Returns the observation, or None when no remaining test in any group is indicated."""
        for g in ranked_groups:
            item_q = self._item_questions(g, case)
            if not item_q:
                case.exhausted_lab_groups.add(g)
                continue
            ans2 = self.client.ask(state, item_q, tag=f"agent.items.{g}")
            scored = sorted(((a["noul"], qid) for qid, a in ans2.items()), reverse=True)
            selected = [qid for p, qid in scored if p >= self.cfg.noul_threshold]
            if not selected:
                case.exhausted_lab_groups.add(g)  # nothing in this group is indicated for this patient
                retired.append(g)
                continue
            dropped: List[str] = []
            if g == "common_panels":  # Nouls are absolute: BMP, CMP and electrolytes all clear 0.5; keep only the superset
                selected, dropped = self._drop_redundant_panels(selected)
            selected = selected[: self.cfg.max_panels_per_order if g == "common_panels" else self.cfg.max_items_per_order]
            itemids: List[int] = []
            names: List[str] = []
            panel_names: List[str] = []
            for qid in selected:
                kind, key = qid.split("::", 1)
                if kind == "panel":
                    itemids.extend(self.menus.panels[key])
                    names.append(key)
                    panel_names.append(key)
                else:
                    itemids.append(int(key))
                    names.append(self.menus.item_label(int(key)))
            obs = case.do_labs(itemids, names, panel_names)
            orders_log.append({
                "group": g, "selected": names, "n_itemids": len(set(itemids)), "n_new": case.trajectory[-1]["tool_input"].get("n_new"),
                "item_nouls_top": [(qid.split("::", 1)[1], round(p, 3)) for p, qid in scored[:15]], "dropped_redundant_panels": dropped,
            })
            return obs
        return None

    def _lab_visit(self, case: PatientCase, state, ans: Dict[str, Any], rec: Dict[str, Any]) -> Optional[str]:
        """The laboratory action, one tool action. One order is placed on `state`, by default once: every endorsed panel or item
        of the best group (`_order_endorsed`). As a visit the order is repeated: its results are revealed, the lab groups are
        ranked again on the updated state and the next order is placed, until no test is indicated in any group (laboratory
        tests then leave the menu) or the bound on orders is reached. The router is not asked in between: asked after every
        single test it left the laboratory after a median of one test although the Noul still endorsed others.
          default                  one multi-select order
          `agent.lab_visit`        multi-select orders until nothing is endorsed, at most `LAB_VISIT_MAX_ORDERS`
          `tools.lab_single_item`  one test per order (`_order_one_test`), always as a visit, at most `max_items_per_order` tests
        Returns the observations, or None when no test is indicated at all."""
        single = case.cfg.lab_single_item
        order = self._order_one_test if single else self._order_endorsed
        max_orders = self.cfg.max_items_per_order if single else (LAB_VISIT_MAX_ORDERS if self.cfg.lab_visit else 1)
        orders_log: List[dict] = []
        retired: List[str] = []
        out: List[str] = []
        while len(out) < max_orders:
            avail = case.available_lab_groups()
            if not avail:
                break
            if not out and "lab_group" in ans:  # the round's own group Choice ranks the groups for the first order
                group_probs = rec["lab_group_probabilities"] = ans["lab_group"]["probabilities"]
            elif len(avail) >= 2:
                group_probs = self.client.ask(state, {"lab_group": choice(self.P["lab_group"], avail)}, tag="agent.lab_group")["lab_group"]["probabilities"]
            else:
                group_probs = dict.fromkeys(avail, 1.0)
            ranked_groups = [g for g, _ in sorted(group_probs.items(), key=lambda kv: -kv[1]) if g in avail]
            obs = order(case, state, ranked_groups, orders_log, retired)
            if obs is None:
                break
            out.append(obs)
            state = case.state()
        if single:
            rec["lab_tests"] = orders_log
        elif self.cfg.lab_visit:
            rec["lab_orders"] = orders_log
        if retired:
            rec["groups_without_indicated_items"] = retired
        if not out:
            # No remaining test in any group is indicated: laboratory tests leave the menu; no tool action consumed.
            rec["lab_group"] = None
            case.log.append("Laboratory tests: no further tests indicated")
            rec["input"] = {"group": None}
            rec["observation_chars"] = 0
            return None
        rec["lab_group"] = orders_log[0]["group"]
        if single:
            selected = [t["test"] for t in orders_log]
        else:
            selected = [n for o in orders_log for n in o["selected"]]
            dropped = [n for o in orders_log for n in o["dropped_redundant_panels"]]
            if dropped:
                rec["dropped_redundant_panels"] = dropped
            rec["item_nouls_top"] = orders_log[0]["item_nouls_top"]
        rec["input"] = {"group": orders_log[0]["group"], "selected": selected, "n_itemids": sum(o["n_itemids"] for o in orders_log)}
        rec["n_new_results"] = sum(o["n_new"] or 0 for o in orders_log)
        return "\n".join(out)

    # ------------------------------------------------------------------ main loop
    def run(self, case: PatientCase, resume: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The work-up of `case`; with `resume` (an earlier return value for the same case) it continues that work-up."""
        rounds: List[dict] = list(resume["rounds"]) if resume else []
        stop_reason = "max_rounds"
        n_actions = resume["n_rounds"] if resume else 0  # executed tool actions; MIMIC-CDM allows max_iterations=10 before forcing a diagnosis
        while True:
            if n_actions >= self.cfg.max_rounds:
                stop_reason = "max_rounds"
                break
            actions = case.available_actions()
            if self.cfg.stop_when_only_final and list(actions) == ["final_diagnosis"]:
                stop_reason = "nothing_left"
                break
            rnd = n_actions + 1
            state = case.state()
            questions = self._round_questions(case, actions)
            ans = self.client.ask(state, questions, tag="agent.route")
            act = ans["next_action"]["choice"]
            rec: Dict[str, Any] = {
                "round": rnd,
                "action": act,
                "action_probabilities": ans["next_action"]["probabilities"],
                "action_confidence": ans["next_action"].get("confidence"),
                "sufficient_for_diagnosis": ans["sufficient_for_diagnosis"]["noul"],
            }
            if act == "final_diagnosis":
                rounds.append(rec)
                stop_reason = "final_diagnosis"
                break
            if act == "ask_patient":  # not a tool action: the questions are bounded by tools.max_questions
                if self.interviewer is None:
                    raise RuntimeError("ask_patient needs the interviewer of decision2care/interactive.py (tools.history=interactive)")
                obs, info = self.interviewer.ask(self.client, case, state)
                rec.update(info)
                rec["observation_chars"] = len(obs or "")
                rounds.append(rec)
                continue
            thr = self.cfg.noul_threshold
            n_log = len(case.log)
            if act == "llm_suggest":  # Opus suggests orders, Jev decides on each; a tool action only when something is ordered
                if self.suggester is None:
                    raise RuntimeError("llm_suggest needs the suggester of decision2care/llm_suggest.py (tools.llm_suggest)")
                case.suggests_used += 1
                obs, info = self.suggester.suggest(self, case, state)
                rec.update(info)
                rec["observation_chars"] = len(obs or "")
                rounds.append(rec)
                if obs is None:
                    case.decline_suggest()
                    continue
                case.add_event(n_log, obs)
                n_actions += 1
                continue
            if act == "physical_examination":
                systems = [s for s in case.unseen_pe_systems() if _yes(ans, f"pe_{s}", thr)]
                rec["pe_nouls"] = {s: ans[f"pe_{s}"]["noul"] for s in case.unseen_pe_systems() if f"pe_{s}" in ans}
                obs = case.do_physical_exam(systems)
                rec["input"] = {"systems": systems or "all"}
            elif act == "vital_signs":
                items = [k for k in VITAL_ITEMS if _yes(ans, f"vital_{k}", thr)]
                rec["vital_nouls"] = {k: ans[f"vital_{k}"]["noul"] for k in VITAL_ITEMS if f"vital_{k}" in ans}
                obs = case.do_vitals(items)
                rec["input"] = {"items": items or "all"}
            elif act == "laboratory_tests":
                obs = self._lab_visit(case, state, ans, rec)
                if obs is None:
                    rounds.append(rec)
                    continue
            elif act == "imaging":
                rem = case.remaining_imaging()
                if "imaging_study" in ans:
                    study = ans["imaging_study"]["choice"]
                    rec["imaging_probabilities"] = ans["imaging_study"]["probabilities"]
                else:
                    study = next(iter(rem))
                if study == "none_indicated" or study not in rem:
                    # Jev sees no useful study left: imaging leaves the menu; no tool action consumed.
                    case.imaging_exhausted = True
                    case.log.append("Imaging: no further study indicated")
                    rec["input"] = {"study": None}
                    rec["observation_chars"] = 0
                    rounds.append(rec)
                    continue
                obs = case.do_imaging(study)
                rec["input"] = {"study": study}
            elif act == "ecg":
                obs = case.do_ecg()
                rec["input"] = {}
            elif act == "echocardiogram":
                obs = case.do_echo()
                rec["input"] = {}
            elif act == "home_medications":
                obs = case.do_home_meds()
                rec["input"] = {}
            elif act == "ed_medications":
                obs = case.do_ed_meds()
                rec["input"] = {}
            else:
                logger.warning(f"unknown action {act}")
                obs = ""
                rec["input"] = {}
            if obs:
                case.add_event(n_log, obs)
            rec["observation_chars"] = len(obs)
            rounds.append(rec)
            n_actions += 1
        return {"rounds": rounds, "n_rounds": n_actions, "n_decisions": len(rounds), "stop_reason": stop_reason, "trajectory": case.trajectory, "config": asdict(self.cfg)}
