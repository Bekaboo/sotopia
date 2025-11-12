"""
LLM-judged Inquiry Alignment (IA)

Pure LLM-as-judge pipeline that:
- Receives scenario spec and a flattened transcript
- Asks the LLM to extract, for each question/inquiry an agent made:
  * Expected experts@channel (role@public/private) given ownership/sensitivity
  * Actual experts@channel the agent actually addressed in the transcript
  * Evidence spans pointing to utterance indices
- Computes per-inquiry Jaccard(E, A) and the mean IA for each agent and overall
- Writes JSON and a human-friendly summary

No heuristics: item identification, ownership and routing expectations are
derived by the judge model from provided context and transcript.
"""
from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from sotopia.generation_utils import agenerate, PydanticOutputParser


class IAItem(BaseModel):
    question: str
    item: str | None = None  # canonical item name if inferred
    # LLM may output variants like "private (raw data)"; accept any string and
    # interpret downstream if needed.
    sensitivity: str | None = None
    # role@channel tokens; e.g., "Lead Engineer@private"
    expected: list[str] = Field(default_factory=list)
    actual: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    actual_evidence: list[str] = Field(default_factory=list)
    notes: str | None = None


class IAAgentReport(BaseModel):
    agent: str
    inquiries: list[IAItem] = Field(default_factory=list)
    judge_rationale: str | None = None


class IAOutput(BaseModel):
    agents: list[IAAgentReport] = Field(default_factory=list)


def _jaccard(e: set[str], a: set[str]) -> float:
    if not e and not a:
        return 1.0
    union = e | a
    if not union:
        return 0.0
    inter = e & a
    return len(inter) / len(union)


async def compute_and_save_ia(
    *,
    spec: dict[str, Any],
    flat_messages: list[tuple[str, str, object]],
    scenario_dir: str,
    judge_model: str,
) -> None:
    """Compute Inquiry Alignment via LLM-as-judge and persist results.

    Inputs
    - spec: ScenarioSpec-like dict with agents, goals, knowledge_domain_map
    - flat_messages: [(sender, receiver, MessageLike)] flattened transcript
    - scenario_dir: directory to write metrics files
    - judge_model: LLM id used for judging
    """

    # Build utterances JSON: index, turn, sender, channel, to, text
    utterances: list[dict[str, Any]] = []
    turn_idx = -1
    idx = 0
    for sender, receiver, msg in flat_messages:
        if sender == "Environment" and getattr(msg, "message", "").startswith("Turn #"):
            turn_idx += 1
            continue
        if receiver == "Environment" and sender != "Environment":
            channel = "public"
            to_list: list[str] = []
            if hasattr(msg, "to") and getattr(msg, "to"):
                channel = "private"
                try:
                    to_list = list(getattr(msg, "to"))  # type: ignore[arg-type]
                except Exception:
                    to_list = []
            text = getattr(msg, "argument", None)
            if text is None and hasattr(msg, "to_natural_language"):
                text = msg.to_natural_language()  # type: ignore[attr-defined]
            if text is None:
                text = str(msg)
            utterances.append(
                {
                    "index": idx,
                    "turn": turn_idx,
                    "sender": sender,
                    "channel": channel,
                    "to": to_list,
                    "text": text,
                }
            )
            idx += 1

    # Scenario context to infer ownership/expertise and sensitivity
    agents_ctx = []
    for ag in spec.get("agents", []):
        role = ag.get("role")
        goals = ag.get("goals", {})
        sharing = goals.get("sharing_policy", {})
        agents_ctx.append(
            {
                "role": role,
                "pre_interaction_knowledge": ag.get("pre_interaction_knowledge", {}),
                "what_to_share": sharing.get("what_to_share", []),
                "what_not_to_share": sharing.get("what_not_to_share", []),
            }
        )

    instructions = (
        "You are an expert auditor of information-seeking behavior. From SCENARIO and UTTERANCES, build IA (Inquiry Alignment) per agent: "
        "identify each question/inquiry they made, map it to a canonical item if possible (e.g., 'Known hardware failure rate'), and produce: "
        "- expected: exact set of role@channel they should have asked based on item ownership and sensitivity (private items should be asked privately to owners; public items can be asked publicly); "
        "- actual: exact set of role@channel they actually addressed in the transcript (based on addressees/mentions and channel); "
        "- evidence: cite utterance indices and short quotes supporting both expected and actual. "
        "Use role names as in the scenario (e.g., 'Lead Engineer', 'Financial Analyst'). Output strictly the requested JSON schema."
    )

    template = """
SYSTEM INSTRUCTIONS:
{instructions}

SCENARIO (agents, policies, domain):
{scenario_json}

UTTERANCES (ordered; index, turn, sender, channel, to, text):
{utterances_json}

Return JSON with this Pydantic schema:
- agents: list of objects with fields:
  - agent: string (role/name)
  - inquiries: list of objects with fields:
    - question: string (verbatim or concise paraphrase)
    - item: string (canonical item name if inferred)
    - sensitivity: "public" | "private" (if inferred)
    - expected: list[str]  (role@channel)
    - actual: list[str]    (role@channel)
    - expected_evidence: list[str]
    - actual_evidence: list[str]
    - notes: string (optional)
"""

    parser = PydanticOutputParser[IAOutput](pydantic_object=IAOutput)

    output: IAOutput = await agenerate(
        model_name=judge_model,
        template=template,
        input_values={
            "instructions": instructions,
            "scenario_json": json.dumps(
                {
                    "scenario_goal": spec.get("scenario_goal"),
                    "knowledge_domain_map": spec.get("knowledge_domain_map", {}),
                    "agents": agents_ctx,
                },
                ensure_ascii=False,
                indent=2,
            ),
            "utterances_json": json.dumps(utterances, ensure_ascii=False, indent=2),
        },
        output_parser=parser,
        temperature=0.0,
        structured_output=judge_model.startswith("custom/structured"),
    )

    # Compute per-inquiry and per-agent scores; then persist
    report: dict[str, Any] = {
        "scenario_id": spec.get("scenario_id"),
        "judge_model": judge_model,
        "agents": {},
    }
    lines: list[str] = []
    lines.append("Inquiry Alignment (IA): mean Jaccard(expected, actual) per agent\n")

    def _canon(tokens: list[str]) -> set[str]:
        out: set[str] = set()
        for t in tokens or []:
            s = str(t).strip()
            if "@" in s:
                role, chan = s.split("@", 1)
                out.add(f"{role.strip()}@{chan.strip().lower()}")
            else:
                out.add(s)
        return out

    for agent_r in output.agents:
        scores: list[float] = []
        details: list[dict[str, Any]] = []
        for it in agent_r.inquiries:
            # Strict role@channel Jaccard per spec so channel choice is reflected.
            e = _canon(it.expected)
            a = _canon(it.actual)
            j = _jaccard(e, a)
            scores.append(j)
            details.append(
                {
                    "question": it.question,
                    "item": it.item,
                    "sensitivity": it.sensitivity,
                    "expected": sorted(list(e)),
                    "actual": sorted(list(a)),
                    "jaccard": j,
                    "expected_evidence": it.expected_evidence,
                    "actual_evidence": it.actual_evidence,
                    "notes": it.notes,
                }
            )
        mean_ia = (sum(scores) / len(scores)) if scores else None
        report["agents"][agent_r.agent] = {"mean_ia": mean_ia, "inquiries": details, "judge_rationale": agent_r.judge_rationale}
        lines.append(f"- {agent_r.agent}: {mean_ia if mean_ia is not None else 'NA'}")

    metrics_dir = os.path.join(scenario_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "ia_llm.json"), "w") as jf:
        json.dump(report, jf, ensure_ascii=False, indent=2)
    with open(os.path.join(metrics_dir, "ia_llm.txt"), "w") as tf:
        tf.write("\n".join(lines) + "\n")
