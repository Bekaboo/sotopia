"""
Efficiency (EFF) — LLM-as-Judge evaluation.

Measures how quickly each agent acquires its ``desired_knowledge`` items.
Earlier acquisition → higher score.

One LLM judge call per scenario.  Returns per-agent EFF scores with
citable evidence spans showing the round in which each item was first
received.

Score formula (per agent):
    For each desired_knowledge item i:
        eff_i = max(0, 1 − round_acquired_i / total_rounds)  if acquired
              = 0                                              if never acquired
    EFF_agent    = mean(eff_i)
    EFF_scenario = mean(EFF_agent)
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Pydantic response models ─────────────────────────────────────────

class EFFAcquisition(BaseModel):
    """Assessment of when (if ever) an agent first received a piece of knowledge."""
    item_description: str = Field(
        description="The desired_knowledge item description exactly as listed"
    )
    was_acquired: bool = Field(
        description="Whether the agent received the substantive content of this item during the conversation"
    )
    round_acquired: Optional[int] = Field(
        default=None,
        description=(
            "The round number (0-indexed, from the [R#.#] tags) in which "
            "the item was first received.  None if never acquired."
        ),
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Utterance IDs where the information was communicated to the agent, e.g. ['[R2.1]']",
    )
    reasoning: str = Field(description="Brief explanation of how/when the item was received")


class EFFAgentResult(BaseModel):
    """EFF assessment for a single agent."""
    agent_name: str = Field(description="The agent's role name")
    acquisitions: list[EFFAcquisition] = Field(
        description="One assessment per desired_knowledge item"
    )


class EFFJudgeResponse(BaseModel):
    """Complete EFF judge response for all agents."""
    agents: list[EFFAgentResult] = Field(description="EFF assessment for each agent")
    total_rounds: int = Field(
        description="The total number of rounds in the conversation (from the transcript header)"
    )


# ── Scoring ───────────────────────────────────────────────────────────

def compute_eff_scores(response: EFFJudgeResponse) -> dict[str, Any]:
    """Derive numeric EFF scores from the judge response."""
    total_rounds = max(response.total_rounds, 1)
    agent_scores: dict[str, dict[str, Any]] = {}

    for agent in response.agents:
        n_items = len(agent.acquisitions)
        if n_items == 0:
            agent_scores[agent.agent_name] = {
                "eff_score": 1.0,
                "acquisitions": [],
            }
            continue

        eff_values: list[float] = []
        for acq in agent.acquisitions:
            if acq.was_acquired and acq.round_acquired is not None:
                eff_i = max(0.0, 1.0 - acq.round_acquired / total_rounds)
            else:
                eff_i = 0.0
            eff_values.append(eff_i)

        eff_agent = sum(eff_values) / len(eff_values)
        agent_scores[agent.agent_name] = {
            "eff_score": round(eff_agent, 4),
            "per_item_eff": [round(v, 4) for v in eff_values],
            "acquisitions": [a.model_dump() for a in agent.acquisitions],
        }

    all_eff = [v["eff_score"] for v in agent_scores.values()]
    scenario_eff = sum(all_eff) / len(all_eff) if all_eff else 0.0

    return {
        "metric": "EFF",
        "scenario_score": round(scenario_eff, 4),
        "total_rounds": total_rounds,
        "agent_scores": agent_scores,
    }


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_total_rounds(transcript: str) -> int:
    """Parse total_rounds from the [SCENARIO] header line."""
    m = re.search(r"rounds=(\d+)", transcript)
    return int(m.group(1)) if m else 1


# ── Judge prompt ──────────────────────────────────────────────────────

EFF_SYSTEM_PROMPT = """\
You are an expert evaluator assessing **Efficiency** in a multi-agent \
knowledge-sharing conversation.

Your task — for EACH agent — is to determine *when* (in which round) \
each of their ``desired_knowledge`` items was first received / \
acquired during the conversation.

IMPORTANT DISAMBIGUATION RULES
• "Acquired" means the agent received the *substantive content* of the \
  item — not merely that the topic was mentioned or that a label was \
  referenced.
• Record the *earliest* round in which the item's value was \
  communicated *to* or *received by* the agent.
• If the agent already possesses the knowledge (it is in their own \
  pre-interaction knowledge), that does NOT count as acquisition. \
  Only count when *another* agent provides the information.
• If the desired item's content is partially conveyed across multiple \
  rounds, use the round where the core substance first appeared.
• If an agent asks for information and another agent responds in the \
  same round, the acquisition round is that round.
• Public messages are visible to all agents — they count as received \
  by every agent. Private messages are only received by the named \
  recipients.
• Use the round numbers from the [R#.#] tags.  The round number is \
  the part before the dot (e.g. [R2.3] is round 2).
• The transcript header line shows ``rounds=N`` — report that as \
  total_rounds.

PROCEDURE — think step by step:
1. For each agent, list their desired_knowledge items.
2. Scan the transcript round by round, noting when each item's value is communicated.
3. For each item, record the earliest round and the utterance ID(s).
4. Record evidence and reasoning, then give the structured verdict.\
"""


# ── Entry point ───────────────────────────────────────────────────────

async def compute_and_save_eff(
    spec: dict[str, Any],
    flat_messages: list[Any],
    scenario_dir: str,
    judge_model: str,
) -> dict[str, Any]:
    """Run the EFF judge and persist results.  Returns the metric dict."""
    from judge_utils import build_context_block, build_item_checklist, call_judge, save_metric

    context = build_context_block(spec, scenario_dir)

    desired_checklist = build_item_checklist(
        spec, "desired_knowledge",
        parent_key="post_interaction_knowledge", subkey=None,
    )

    user_prompt = (
        f"{context}\n\n"
        "## Items to evaluate\n\n"
        "### desired_knowledge (determine when each was first acquired):\n"
        f"{desired_checklist}\n\n"
        "## Task\n"
        "For each agent, determine when each of their desired_knowledge "
        "items was first received / acquired during the conversation.\n"
        "Evaluate *every* item listed above. Do NOT skip any.\n\n"
        "Report:\n"
        "- `was_acquired`: true/false\n"
        "- `round_acquired`: the 0-indexed round number of first "
        "receipt (from the [R#.#] tags), or null if never acquired\n"
        "- `total_rounds`: the total number of rounds from the "
        "transcript header\n\n"
        "Cite specific utterance IDs as evidence.\n"
        "Think step by step before giving your final structured response."
    )

    response: EFFJudgeResponse = await call_judge(
        model=judge_model,
        system_prompt=EFF_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=EFFJudgeResponse,
    )

    results = compute_eff_scores(response)
    save_metric(scenario_dir, "eff", results)
    return results
