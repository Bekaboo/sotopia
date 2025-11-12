"""
LLM-judged Efficiency (Eff)

Purpose: "How fast did the agent get the information it needed?"

Pipeline (no heuristics):
- Provide the LLM with scenario context (including each agent's desired_knowledge)
  and the flattened transcript.
- Ask the LLM to determine, for each desired item of each agent, the earliest
  turn where that agent "received" the item (visible to them), or mark as not
  received. Include evidence: utterance indices and short quotes.
- Compute per agent:
    times = [turn_received for each desired item (or T_max + 1 if never)]
    median_time = median(times) if times else 1
    Eff = 1 - ((median_time - 1) / (T_max - 1))  (with safe edge handling)
- Persist JSON and a human-readable summary.
"""
from __future__ import annotations

import json
import os
import statistics
from typing import Any

from pydantic import BaseModel, Field

from sotopia.generation_utils import agenerate, PydanticOutputParser


class EffDesiredItem(BaseModel):
    name: str
    received: bool
    turn_received: int | None = None
    evidence_indices: list[int] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)
    notes: str | None = None


class EffAgentReport(BaseModel):
    agent: str
    desired_items: list[EffDesiredItem] = Field(default_factory=list)
    judge_rationale: str | None = None


class EffOutput(BaseModel):
    agents: list[EffAgentReport] = Field(default_factory=list)


def _compute_tmax(utterances: list[dict[str, Any]]) -> int:
    turns = [u.get("turn", -1) for u in utterances if isinstance(u.get("turn", None), int)]
    if not turns:
        return 1
    # turns are zero-based in our logs; T_max is count in 1-based terms
    return max(turns) + 1


def _eff_from_times(times: list[int], t_max: int) -> float:
    if not times:
        return 1.0
    med = statistics.median(times)
    if t_max <= 1:
        return 1.0 if med <= 1 else 0.0
    return 1.0 - ((med - 1.0) / (t_max - 1.0))


async def compute_and_save_eff(
    *,
    spec: dict[str, Any],
    flat_messages: list[tuple[str, str, object]],
    scenario_dir: str,
    judge_model: str,
) -> None:
    # Build UTTERANCES JSON: index, turn, sender, channel, to, text
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

    # Per-agent desired knowledge from scenario spec
    agents_ctx = []
    for ag in spec.get("agents", []):
        role = ag.get("role")
        desired = list(ag.get("post_interaction_knowledge", {}).get("desired_knowledge", []))
        agents_ctx.append({"role": role, "desired_knowledge": desired})

    t_max = _compute_tmax(utterances)

    instructions = (
        "You are an expert auditor of information acquisition speed. For each agent, and each of their desired_knowledge items, determine the earliest turn when the agent RECEIVED that item (i.e., an answer or statement making it known to them). If never received, mark received=false and leave turn_received null. Provide evidence: utterance indices and short quotes."
        " Turns are provided in UTTERANCES. Consider visibility: public utterances are visible to all; private utterances are visible to sender and listed recipients only."
    )

    template = """
SYSTEM INSTRUCTIONS:
{instructions}

SCENARIO (per-agent desired_knowledge):
{agents_json}

UTTERANCES (ordered; index, turn, sender, channel, to, text):
{utterances_json}

Return JSON with this Pydantic schema:
- agents: list of objects with fields:
  - agent: string
  - desired_items: list of objects with fields:
    - name: string (desired item)
    - received: boolean
    - turn_received: integer | null
    - evidence_indices: list[int]
    - evidence_spans: list[str]
    - notes: string (optional)
"""

    class EffLLMOutItem(BaseModel):
        name: str
        received: bool
        turn_received: int | None = None
        evidence_indices: list[int] = Field(default_factory=list)
        evidence_spans: list[str] = Field(default_factory=list)
        notes: str | None = None

    class EffLLMOutAgent(BaseModel):
        agent: str
        desired_items: list[EffLLMOutItem] = Field(default_factory=list)
        judge_rationale: str | None = None

    class EffLLMOut(BaseModel):
        agents: list[EffLLMOutAgent] = Field(default_factory=list)

    parser = PydanticOutputParser[EffLLMOut](pydantic_object=EffLLMOut)

    output: EffLLMOut = await agenerate(
        model_name=judge_model,
        template=template,
        input_values={
            "instructions": instructions,
            "agents_json": json.dumps(agents_ctx, ensure_ascii=False, indent=2),
            "utterances_json": json.dumps(utterances, ensure_ascii=False, indent=2),
        },
        output_parser=parser,
        temperature=0.0,
        structured_output=judge_model.startswith("custom/structured"),
    )

    # Compute Eff for each agent and persist
    report: dict[str, Any] = {
        "scenario_id": spec.get("scenario_id"),
        "judge_model": judge_model,
        "T_max": t_max,
        "agents": {},
    }
    lines: list[str] = []
    lines.append("Efficiency (Eff): 1 - ((median(T_i) - 1) / (T_max - 1))\n")

    for agent_r in output.agents:
        # Construct times with T_max+1 penalty when not received
        desired_raw = next((a for a in agents_ctx if a["role"] == agent_r.agent), {"desired_knowledge": []})
        desired_items_list = list(desired_raw.get("desired_knowledge", []))
        # Build lookup from LLM results
        llm_map: dict[str, EffLLMOutItem] = {d.name: d for d in agent_r.desired_items}
        times: list[int] = []
        details: list[dict[str, Any]] = []
        for item_name in desired_items_list:
            entry = llm_map.get(item_name)
            if entry and entry.received and isinstance(entry.turn_received, int):
                times.append(entry.turn_received)
                details.append(
                    {
                        "name": item_name,
                        "received": True,
                        "turn_received": entry.turn_received,
                        "evidence_indices": entry.evidence_indices,
                        "evidence_spans": entry.evidence_spans,
                        "notes": entry.notes,
                    }
                )
            else:
                times.append(t_max + 1)
                details.append(
                    {
                        "name": item_name,
                        "received": False,
                        "turn_received": None,
                        "evidence_indices": (entry.evidence_indices if entry else []),
                        "evidence_spans": (entry.evidence_spans if entry else []),
                        "notes": (entry.notes if entry else None),
                    }
                )

        eff_score = _eff_from_times(times, t_max)
        report["agents"][agent_r.agent] = {
            "efficiency": eff_score,
            "times": times,
            "desired_items": details,
            "judge_rationale": agent_r.judge_rationale,
        }
        lines.append(f"- {agent_r.agent}: Eff={eff_score:.3f} (median from {times}, T_max={t_max})")

    metrics_dir = os.path.join(scenario_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "eff_llm.json"), "w") as jf:
        json.dump(report, jf, ensure_ascii=False, indent=2)
    with open(os.path.join(metrics_dir, "eff_llm.txt"), "w") as tf:
        tf.write("\n".join(lines) + "\n")

