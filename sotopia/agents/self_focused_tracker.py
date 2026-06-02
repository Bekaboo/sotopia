"""
Self-Focused Scratchpad Tracker — Ablation control for ToM-Belief.

Maintains the SAME persistent structured state as ToM-Belief but WITHOUT
any other-agent modeling (beliefs about others).  This isolates whether
gains come from the structured scratchpad effect or from genuine
Theory-of-Mind reasoning.

Key difference from ToM-Belief:
  - NO AgentBelief per other agent (knows / does_not_know / wants / thinks_about_me)
  - INSTEAD: PlanningState (next_actions / blockers / conversation_strategy)
  - SharingProgress, Memory, PrivacyRisks remain IDENTICAL

If ToM-Belief > Self-Focused: other-agent modeling provides value (genuine ToM).
If ToM-Belief ≈ Self-Focused: gains are from structured scratchpad (Clever Hans).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sotopia.generation_utils.generate import agenerate
from sotopia.generation_utils.output_parsers import PydanticOutputParser


# ── Structured data models ───────────────────────────────────────────────

class PlanningState(BaseModel):
    """Self-focused planning (replaces AgentBelief about others)."""
    model_config = ConfigDict(extra="forbid")

    next_actions: list[str]
    blockers: list[str]
    conversation_strategy: str


class MemoryItem(BaseModel):
    """A single event in the memory buffer."""
    model_config = ConfigDict(extra="forbid")

    turn: int
    event: str


class PrivacyRisk(BaseModel):
    """A detected privacy risk."""
    model_config = ConfigDict(extra="forbid")

    turn: int
    description: str


class SharingProgress(BaseModel):
    """Tracks what the agent has/hasn't shared and acquired."""
    model_config = ConfigDict(extra="forbid")

    items_shared: list[str]
    items_not_yet_shared: list[str]
    items_acquired: list[str]
    items_still_needed: list[str]
    objective_progress: str


class SelfFocusedState(BaseModel):
    """Complete self-focused state (no other-agent beliefs)."""
    model_config = ConfigDict(extra="forbid")

    planning: PlanningState
    sharing_progress: SharingProgress
    memory: list[MemoryItem]
    privacy_risks: list[PrivacyRisk]


# ── Limits ───────────────────────────────────────────────────────────────

MAX_MEMORY_ITEMS = 10
MAX_PRIVACY_RISKS = 5


# ── Prompt templates ─────────────────────────────────────────────────────

SELF_FOCUSED_INIT_SYSTEM = """\
You are a task-planning state initializer.  Given the scenario background \
that a particular agent sees at the start of a multi-agent conversation, \
produce their INITIAL self-focused state as structured JSON.

For planning, analyze the agent's goals and determine:
- next_actions: list of 2-3 immediate actions the agent should take
- blockers: list of things that might prevent progress (empty at start)
- conversation_strategy: one sentence describing how to approach the conversation

For sharing_progress, analyze the agent's goals and pre_interaction_knowledge:
- items_shared: [] (nothing shared yet)
- items_not_yet_shared: list ALL items from 'MAY share' that the agent could share
- items_acquired: [] (nothing acquired yet)
- items_still_needed: list ALL items the agent needs to acquire per their objective
- objective_progress: "Not started"

Set memory to an empty list and privacy_risks to an empty list.

Be concise — each list item should be one short sentence.\
"""

SELF_FOCUSED_UPDATE_SYSTEM = """\
You are a task-planning state updater.  You maintain a running self-focused \
state for a specific agent in a multi-agent conversation.

You will receive:
1. The agent's ROLE and GOALS
2. The agent's CURRENT STATE as JSON
3. NEW MESSAGES since the last update

Produce an UPDATED state as structured JSON by:

## Planning (IMPORTANT — drives strategic behavior)
- Update next_actions based on what has happened and what still needs to happen.
- Update blockers if new obstacles or dependencies emerged.
- Revise conversation_strategy based on how the conversation is progressing.

## Sharing progress (CRITICAL — this drives the agent's behavior)
- Move items from items_not_yet_shared to items_shared when the agent shared them.
- Move items from items_still_needed to items_acquired when the agent received them.
- Update objective_progress honestly: "Not started" / "In progress" / \
"Mostly complete" / "Complete — ready to leave".
- BE HONEST: if information was exchanged, reflect it immediately.

## Memory & risks
- Adding important new events to memory (keep at most {max_memory} items).
- Adding privacy risks if anyone probed for restricted info (keep at most {max_risks}).

RULES:
- If something has NOT changed, keep the previous entries.
- Each list item = one short sentence.
- Do NOT add duplicate entries that are semantically identical to existing ones.
- Focus on PROGRESS: the agent needs to know what to share next, not just what to protect.\
"""


# ── Rendering ────────────────────────────────────────────────────────────

def render_self_focused_state(state: SelfFocusedState) -> str:
    """Render state to text for agent context injection."""
    lines: list[str] = []

    # Sharing progress first — this is what drives behavior
    sp = state.sharing_progress
    lines.append("## Your Progress")
    lines.append(f"  Objective status: {sp.objective_progress}")
    if sp.items_still_needed:
        lines.append(f"  STILL NEED: {'; '.join(sp.items_still_needed)}")
    else:
        lines.append("  STILL NEED: (nothing — objective met!)")
    if sp.items_not_yet_shared:
        lines.append(f"  CAN STILL SHARE: {'; '.join(sp.items_not_yet_shared)}")
    else:
        lines.append("  CAN STILL SHARE: (all shareable items shared)")
    if sp.items_acquired:
        lines.append(f"  ACQUIRED: {'; '.join(sp.items_acquired)}")
    if sp.items_shared:
        lines.append(f"  SHARED: {'; '.join(sp.items_shared)}")

    # Planning (replaces "Beliefs About Others" section)
    lines.append("\n## Your Plan")
    p = state.planning
    lines.append(f"  STRATEGY: {p.conversation_strategy}")
    if p.next_actions:
        lines.append(f"  NEXT ACTIONS: {'; '.join(p.next_actions)}")
    if p.blockers:
        lines.append(f"  BLOCKERS: {'; '.join(p.blockers)}")

    if state.memory:
        lines.append("\n## Key Events")
        for m in state.memory:
            lines.append(f"  [Turn {m.turn}] {m.event}")

    if state.privacy_risks:
        lines.append("\n## Privacy Risks")
        for r in state.privacy_risks:
            lines.append(f"  [Turn {r.turn}] {r.description}")

    return "\n".join(lines)


# ── Tracker ──────────────────────────────────────────────────────────────

class SelfFocusedTracker:
    """Persistent self-focused state (no other-agent modeling)."""

    def __init__(self, agent_name: str, model_name: str) -> None:
        self.agent_name = agent_name
        self.model_name = model_name
        self._use_structured = True
        self.state: SelfFocusedState = SelfFocusedState(
            planning=PlanningState(
                next_actions=[],
                blockers=[],
                conversation_strategy="",
            ),
            sharing_progress=SharingProgress(
                items_shared=[],
                items_not_yet_shared=[],
                items_acquired=[],
                items_still_needed=[],
                objective_progress="Not started",
            ),
            memory=[],
            privacy_risks=[],
        )
        self._initialized: bool = False
        self._last_inbox_len: int = 0

    async def initialize(self, background: str, agent_goal: str) -> None:
        """Create the initial state from scenario background."""
        if self._initialized:
            return

        user_prompt = (
            f"## Agent Being Modeled\n"
            f"Role: {self.agent_name}\n\n"
            f"## Agent's Goals & Policies\n"
            f"{agent_goal}\n\n"
            f"## Scenario Background\n"
            f"{background}\n\n"
            f"Produce the initial self-focused state for {self.agent_name}."
        )

        prompt = SELF_FOCUSED_INIT_SYSTEM + "\n\n" + user_prompt
        template = "{prompt_text}"

        try:
            result: SelfFocusedState = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={"prompt_text": prompt},
                output_parser=PydanticOutputParser[SelfFocusedState](
                    pydantic_object=SelfFocusedState
                ),
                temperature=0.2,
                structured_output=self._use_structured,
            )
            self.state = result
        except Exception:
            pass  # Graceful degradation — empty state

        self._initialized = True
        self._last_inbox_len = 1  # background message consumed

    async def update(
        self,
        agent_goal: str,
        inbox: list[tuple[str, object]],
    ) -> str:
        """Update state based on new messages and return rendered text."""
        new_messages = inbox[self._last_inbox_len:]
        self._last_inbox_len = len(inbox)

        if not new_messages:
            return render_self_focused_state(self.state)

        new_msgs_text = "\n".join(
            f"{msg.to_natural_language()}" for _, msg in new_messages
        )

        current_state_json = self.state.model_dump_json(indent=2)

        update_system = SELF_FOCUSED_UPDATE_SYSTEM.format(
            max_memory=MAX_MEMORY_ITEMS,
            max_risks=MAX_PRIVACY_RISKS,
        )

        user_prompt = (
            f"## Agent Being Modeled\n"
            f"Role: {self.agent_name}\n\n"
            f"## Agent's Goals & Policies\n"
            f"{agent_goal}\n\n"
            f"## Current State (JSON)\n"
            f"{current_state_json}\n\n"
            f"## New Messages (since last update)\n"
            f"{new_msgs_text}\n\n"
            f"Produce the updated state."
        )

        prompt = update_system + "\n\n" + user_prompt
        template = "{prompt_text}"

        try:
            result: SelfFocusedState = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={"prompt_text": prompt},
                output_parser=PydanticOutputParser[SelfFocusedState](
                    pydantic_object=SelfFocusedState
                ),
                temperature=0.2,
                structured_output=self._use_structured,
            )
            # Enforce hard caps
            result.memory = result.memory[-MAX_MEMORY_ITEMS:]
            result.privacy_risks = result.privacy_risks[-MAX_PRIVACY_RISKS:]
            self.state = result
        except Exception:
            pass  # keep previous state on failure

        return render_self_focused_state(self.state)
