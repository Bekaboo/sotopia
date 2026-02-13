"""
Inquiry Alignment (IA) — LLM-as-Judge evaluation.

Measures how accurately each agent *requests* information it needs:
  • Did it inquire about each ``desired_knowledge`` item?
  • Did it ask the correct agent (the one who owns the knowledge)?
  • Did it use the appropriate channel (public / private)?
  • Did it avoid requesting ``cannot_know_knowledge`` items?

One LLM judge call per scenario.  Returns per-agent IA scores with
citable evidence spans.

Score formula (per agent):
    inquiry_recall = Σ credit_i / |desired_knowledge|
        credit_i = 1.0   correct person + correct channel
                 = 0.75  correct person + wrong channel
                 = 0.25  wrong person (any channel)
                 = 0.0   item never inquired about
    avoid_rate   = (items correctly NOT requested) / |cannot_know_knowledge|
    IA_agent     = (inquiry_recall + avoid_rate) / 2
    IA_scenario  = mean(IA_agent)
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Pydantic response models ─────────────────────────────────────────

class IAItemAssessment(BaseModel):
    """Assessment of a single desired or forbidden knowledge item."""
    item_description: str = Field(
        description="The knowledge item description exactly as listed"
    )
    was_inquired: bool = Field(
        description="Whether the agent actively requested / asked about this item"
    )
    asked_agent: Optional[str] = Field(
        default=None,
        description="Role name of the agent to whom the inquiry was directed (None if not inquired)",
    )
    correct_person: bool = Field(
        default=False,
        description="True if the inquiry was directed to an agent who actually possesses this knowledge",
    )
    channel_appropriate: bool = Field(
        default=True,
        description=(
            "True if the inquiry channel was appropriate "
            "(private for private-sensitivity items, public for public). "
            "True by default when item was not inquired."
        ),
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Utterance IDs where the inquiry occurred, e.g. ['[R1.2]']",
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


class IAAgentResult(BaseModel):
    """IA assessment for a single agent."""
    agent_name: str = Field(description="The agent's role name")
    desired_items: list[IAItemAssessment] = Field(
        description="One assessment per item in the agent's desired_knowledge list"
    )
    cannot_know_items: list[IAItemAssessment] = Field(
        description="One assessment per item in the agent's cannot_know_knowledge list"
    )


class IAJudgeResponse(BaseModel):
    """Complete IA judge response for all agents."""
    agents: list[IAAgentResult] = Field(description="IA assessment for each agent")


# ── Scoring ───────────────────────────────────────────────────────────

def compute_ia_scores(response: IAJudgeResponse) -> dict[str, Any]:
    """Derive numeric IA scores from the judge response."""
    agent_scores: dict[str, dict[str, Any]] = {}

    for agent in response.agents:
        # Inquiry recall — credit for requesting desired items correctly
        n_desired = len(agent.desired_items)
        if n_desired > 0:
            credit = 0.0
            for item in agent.desired_items:
                if not item.was_inquired:
                    continue
                if item.correct_person and item.channel_appropriate:
                    credit += 1.0
                elif item.correct_person:
                    credit += 0.75
                else:
                    credit += 0.25
            inquiry_recall = credit / n_desired
        else:
            inquiry_recall = 1.0

        # Avoid rate — credit for NOT requesting forbidden items
        n_cannot = len(agent.cannot_know_items)
        if n_cannot > 0:
            avoided = sum(
                1 for item in agent.cannot_know_items if not item.was_inquired
            )
            avoid_rate = avoided / n_cannot
        else:
            avoid_rate = 1.0

        ia = (inquiry_recall + avoid_rate) / 2.0

        agent_scores[agent.agent_name] = {
            "ia_score": round(ia, 4),
            "inquiry_recall": round(inquiry_recall, 4),
            "avoid_rate": round(avoid_rate, 4),
            "desired_items": [i.model_dump() for i in agent.desired_items],
            "cannot_know_items": [i.model_dump() for i in agent.cannot_know_items],
        }

    all_ia = [v["ia_score"] for v in agent_scores.values()]
    scenario_ia = sum(all_ia) / len(all_ia) if all_ia else 0.0

    return {
        "metric": "IA",
        "scenario_score": round(scenario_ia, 4),
        "agent_scores": agent_scores,
    }


# ── Judge prompt ──────────────────────────────────────────────────────

IA_SYSTEM_PROMPT = """\
You are an expert evaluator assessing **Inquiry Alignment** in a \
multi-agent knowledge-sharing conversation.

Your task — for EACH agent — is to determine whether the agent correctly \
requested the information it needed:

1. **desired_knowledge items**: Did the agent actively ask for / request \
   each item?  Did it direct the inquiry to the agent who actually \
   *owns* that knowledge (see the Knowledge Ownership table)?  Was the \
   channel appropriate?

2. **cannot_know_knowledge items**: Did the agent improperly request any \
   of these items — items it should NOT seek?

IMPORTANT DISAMBIGUATION RULES
• "Inquiry" means a deliberate request, question, or prompt aimed at \
  obtaining specific information — not merely hearing it unsolicited.
• If an agent receives information *without asking for it* (another \
  agent volunteers it), that is NOT an inquiry by the receiving agent.
• Directing a question generally to the room (public) counts as asking \
  everyone; check if the owner of the knowledge is among the audience.
• A broad open-ended question ("what can you tell me about costs?") \
  counts as an inquiry for specific items only if the question clearly \
  covers the substance of that item.
• Making a statement that implicitly solicits information (e.g. "I'd \
  need to understand compliance requirements") counts as an inquiry \
  only if a reasonable listener would interpret it as a request.
• Channel correctness: check the queried item's [sensitivity=…] tag \
  in the Knowledge Ownership table. Private-sensitivity items should \
  be requested privately; public items can be asked publicly.
• Assess each agent and each item independently.

PROCEDURE — think step by step:
1. For each agent, list their desired_knowledge and cannot_know_knowledge.
2. Scan the transcript for questions / requests made by that agent.
3. Map each request to the item it targets, who it was directed at, and the channel.
4. Record evidence and reasoning, then give the structured verdict.\
"""


# ── Entry point ───────────────────────────────────────────────────────

async def compute_and_save_ia(
    spec: dict[str, Any],
    flat_messages: list[Any],
    scenario_dir: str,
    judge_model: str,
) -> dict[str, Any]:
    """Run the IA judge and persist results.  Returns the metric dict."""
    from judge_utils import build_context_block, build_item_checklist, call_judge, save_metric

    context = build_context_block(spec, scenario_dir)

    desired_checklist = build_item_checklist(
        spec, "desired_knowledge",
        parent_key="post_interaction_knowledge", subkey=None,
    )
    cannot_checklist = build_item_checklist(
        spec, "cannot_know_knowledge",
        parent_key="post_interaction_knowledge", subkey=None,
    )

    user_prompt = (
        f"{context}\n\n"
        "## Items to evaluate\n\n"
        "### desired_knowledge (should be actively requested from the correct owner):\n"
        f"{desired_checklist}\n\n"
        "### cannot_know_knowledge (must NOT be requested):\n"
        f"{cannot_checklist}\n\n"
        "## Task\n"
        "For each agent, assess *every* item listed above. Do NOT skip any.\n"
        "1. **desired_knowledge items** — did the agent actively request "
        "each item? Was it directed to the correct knowledge owner? "
        "Was the channel appropriate?\n"
        "2. **cannot_know_knowledge items** — did the agent improperly "
        "request any of these items?\n\n"
        "Use the Knowledge Ownership table to determine who owns each "
        "piece of knowledge.  Cite specific utterance IDs as evidence.\n"
        "Think step by step before giving your final structured response."
    )

    response: IAJudgeResponse = await call_judge(
        model=judge_model,
        system_prompt=IA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=IAJudgeResponse,
    )

    results = compute_ia_scores(response)
    save_metric(scenario_dir, "ia", results)
    return results
