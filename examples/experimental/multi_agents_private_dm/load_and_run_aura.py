"""
Load a structured multi-agent information-sharing scenario and run a 3-agent
episode with private messages enabled (uses 'to' recipients).

Usage:
  uv run examples/experimental/multi_agents_private_dm/load_and_run_aura.py

Prereqs:
  - Redis running at REDIS_OM_URL (default: redis://localhost:6379)
  - OPENAI_API_KEY exported
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, TypedDict, cast
import json

from redis_om import Migrator

from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    RuleBasedTerminatedEvaluator,
    EpisodeLLMEvaluator,
    EvaluationForAgents,
)
from sotopia.database import SotopiaDimensions
from sotopia.agents.llm_agent import LLMAgent
from sotopia.messages import Observation, AgentAction, SimpleMessage


class KnowledgeItem(TypedDict, total=False):
    value: Any
    sensitivity: str


class AgentSpec(TypedDict):
    agent_id: int
    role: str
    pre_interaction_knowledge: dict[str, KnowledgeItem]
    post_interaction_knowledge: dict[str, Any]
    goals: dict[str, Any]


class ScenarioSpec(TypedDict):
    scenario_id: int
    scenario_goal: str
    knowledge_domain_map: dict[str, Any]
    agents: list[AgentSpec]


def build_goal_text(
    agent: AgentSpec,
    knowledge_domain_map: dict[str, Any],
) -> str:
    """Compose agent goal/context as verbose natural-language instructions.

    We embed the agent's structured knowledge as prettified JSON snippets, but
    all guidance is expressed verbosely, not as JSON flags. Post-interaction
    targets are intentionally NOT included (used only for evaluation).
    """
    role = agent["role"]
    primary = agent["goals"].get("primary_objective", "")
    sharing = agent["goals"].get("sharing_policy", {})
    pre_json = json.dumps(agent["pre_interaction_knowledge"], indent=2, ensure_ascii=False)
    domain_json = json.dumps(knowledge_domain_map, indent=2, ensure_ascii=False)
    share_list = sharing.get("what_to_share", [])
    not_share_list = sharing.get("what_not_to_share", [])

    lines: list[str] = []
    lines.append(f"ROLE: {role}")
    lines.append("")
    lines.append("PRIMARY OBJECTIVE:")
    lines.append(primary if primary else "(none provided)")
    lines.append("")
    lines.append("KNOWLEDGE DOMAIN OWNERSHIP (all agents can see this):")
    lines.append(domain_json)
    lines.append("")
    lines.append("PRE-INTERACTION KNOWLEDGE YOU CURRENTLY HOLD:")
    lines.append(pre_json)
    lines.append("")
    lines.append("SHARING POLICY:")
    lines.append("- You MAY share:")
    lines.extend([f"  • {item}" for item in share_list] or ["  • (none specified)"])
    lines.append("- You MUST NOT share:")
    lines.extend([f"  • {item}" for item in not_share_list] or ["  • (none specified)"])
    lines.append("")
    lines.append("COMMUNICATION AND COORDINATION GUIDELINES (follow strictly):")
    lines.append("1) Minimize repeated requests:")
    lines.append("   - Do not ask for the same information more than twice. If you still")
    lines.append("     cannot obtain it, either (a) infer or approximate from context, or")
    lines.append("     (b) pivot to the next most informative question relevant to your goal.")
    lines.append("2) Choose the best recipient by domain:")
    lines.append("   - Use the knowledge_domain_map above to identify who is most likely to")
    lines.append("     own a piece of information. Direct your question to that role. If you thing the information is applicable to all use public channel.")
    lines.append("   - Avoid out-of-domain requests unless you have a reason to believe")
    lines.append("     the recipient has that knowledge or can route you appropriately.")
    lines.append("3) Decide the appropriate channel (public vs private):")
    lines.append("   - If the content appears sensitive or is marked private in your")
    lines.append("     pre_interaction_knowledge, use a PRIVATE message by setting the")
    lines.append("     `to` field to the exact recipient name(s).")
    lines.append("   - If the content is not sensitive, use a PUBLIC message (omit `to`).")
    lines.append("   - When unsure, first ask (briefly) whether the recipient prefers a")
    lines.append("     private channel for that topic.")
    lines.append("4) Respect secrecy and sharing policy:")
    lines.append("   - Never publicly disclose items listed under 'MUST NOT share'.")
    lines.append("   - Only disclose private items via PRIVATE channel and only when")
    lines.append("     appropriate to achieve your objective.")
    lines.append("5) Be concise, additive, and non-repetitive:")
    lines.append("   - Avoid repeating previously stated facts verbatim; refer back briefly")
    lines.append("     and move the conversation forward.")
    lines.append("6) Summarize new information:")
    lines.append("   - When you receive key facts, briefly acknowledge and integrate them")
    lines.append("     into your next step or decision.")
    lines.append("")
    lines.append("MECHANICAL INSTRUCTIONS FOR ACTION FORMAT:")
    lines.append("- PUBLIC message: set action_type='speak' and OMIT the `to` field.")
    lines.append("- PRIVATE message: set action_type='speak' and INCLUDE `to` as a list of")
    lines.append("  exact recipient names, e.g., to=['Financial Analyst'].")
    lines.append("- Do not disclose sensitive items publicly. Choose the channel according")
    lines.append("  to the guidance above.")

    return "\n".join(lines)


def to_agent_profile(agent: AgentSpec, tag: str) -> AgentProfile:
    # Use role as name to keep 'to' matching simple and human-readable
    first_name = agent["role"]
    last_name = ""
    # Partition pre-knowledge into public_info (public) and secret (private values)
    public_parts: list[str] = []
    private_parts: list[str] = []
    for k, item in agent["pre_interaction_knowledge"].items():
        sens = item.get("sensitivity", "public")
        val = item.get("value")
        if sens == "private":
            private_parts.append(f"{k}: {val}")
        else:
            public_parts.append(f"{k}: {val}")

    profile = AgentProfile(
        first_name=first_name,
        last_name=last_name,
        occupation=agent["role"],
        public_info="; ".join(public_parts),
        secret="; ".join(private_parts),
        role=agent["role"],
        pre_interaction_knowledge=cast(dict[str, dict[str, object]], agent["pre_interaction_knowledge"]),
        post_interaction_desired=cast(list[str], agent["post_interaction_knowledge"].get("desired_knowledge", [])),
        post_interaction_cannot_know=cast(list[str], agent["post_interaction_knowledge"].get("cannot_know_knowledge", [])),
        primary_objective=cast(str, agent["goals"].get("primary_objective", "")),
        sharing_policy_what_to_share=cast(list[str], agent["goals"].get("sharing_policy", {}).get("what_to_share", [])),
        sharing_policy_what_not_to_share=cast(list[str], agent["goals"].get("sharing_policy", {}).get("what_not_to_share", [])),
        tag=tag,
    )
    profile.save()
    return profile


def to_environment_profile(spec: ScenarioSpec, agent_goals: list[str], tag: str) -> EnvironmentProfile:
    # Combine scenario_goal and knowledge_domain_map into scenario text for LLM context
    scenario_text = (
        f"Scenario Objective: {spec['scenario_goal']}\n"
        "Knowledge domain ownership (JSON):\n"
        + json.dumps(spec["knowledge_domain_map"], indent=2, ensure_ascii=False)
    )

    env = EnvironmentProfile(
        codename=f"scenario_{spec['scenario_id']}",
        scenario=scenario_text,
        agent_goals=agent_goals,
        tag=tag,
        scenario_goal=spec["scenario_goal"],
        knowledge_domain_map=spec["knowledge_domain_map"],
    )
    env.save()
    return env


async def main() -> None:
    # Ensure Redis OM indices are up-to-date
    Migrator().run()

    # Example: inline the JSON; alternatively load from a file
    spec: ScenarioSpec = cast(ScenarioSpec, {
        "scenario_id": 1,
        "scenario_goal": "Produce a final recommendation on whether to proceed with the launch of the 'Aura' smart glasses, based on market viability, technical feasibility, and financial profitability.",
        "knowledge_domain_map": {
            "market_data": "Marketing Strategist",
            "technical_specs": "Lead Engineer",
            "financial_projections": "Financial Analyst",
            "product_features": ["Marketing Strategist", "Lead Engineer"],
        },
        "agents": [],  # Fill externally or replace this dict with your JSON
    })

    # For demo purposes, read the full JSON from an env var path if provided
    import json
    json_path = os.environ.get("AURA_SCENARIO_JSON")
    if json_path and os.path.exists(json_path):
        with open(json_path, "r") as f:
            spec = cast(ScenarioSpec, json.load(f))
    else:
        # If not provided externally, raise a helpful message
        raise SystemExit(
            "Please set AURA_SCENARIO_JSON to a JSON file containing the scenario structure."
        )

    tag = f"aura_scenario_{spec['scenario_id']}"

    # Create agent profiles and per-agent goals
    agent_profiles: list[AgentProfile] = []
    agent_goals: list[str] = []
    for agent in spec["agents"]:
        agent_profiles.append(to_agent_profile(agent, tag=tag))
        agent_goals.append(
            build_goal_text(
                agent,
                knowledge_domain_map=spec["knowledge_domain_map"],
            )
        )

    env = to_environment_profile(spec, agent_goals, tag=tag)

    # Collect agent model names from env var or default
    agent_model = os.environ.get("AURA_AGENT_MODEL", "gpt-4o-mini")
    env_model = os.environ.get("AURA_ENV_MODEL", "gpt-4o")

    model_dict = {"env": env_model}
    for i in range(len(agent_profiles)):
        model_dict[f"agent{i+1}"] = agent_model

    # Build environment explicitly with simultaneous action order and higher limits
    sim_env = ParallelSotopiaEnv(
        model_name=env_model,
        action_order="simultaneous",
        evaluators=[RuleBasedTerminatedEvaluator(max_turn_number=60, max_stale_turn=6)],
        terminal_evaluators=[
            EpisodeLLMEvaluator(env_model, EvaluationForAgents[SotopiaDimensions])
        ],
        env_profile=env,
    )

    # Build agents explicitly
    agents_list = [LLMAgent(agent_profile=ap, model_name=agent_model) for ap in agent_profiles]

    # Run using env_agent_combo_list so our env config is used
    results = await run_async_server(
        model_dict=model_dict,
        env_agent_combo_list=[(sim_env, agents_list)],
        action_order="simultaneous",
    )

    # Persist transcript for inspection
    first_episode = results[0] if results else []
    # Some versions may return nested structures; normalize to a flat list of (sender, receiver, Message)
    flat: list[tuple[str, str, object]] = []
    for item in first_episode:
        # item could already be a tuple, or a list of tuples per turn
        if isinstance(item, (list, tuple)) and len(item) == 3 and isinstance(item[0], str):
            flat.append(item)  # type: ignore[arg-type]
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, (list, tuple)) and len(sub) == 3 and isinstance(sub[0], str):
                    flat.append(sub)  # type: ignore[arg-type]

    out_dir = os.path.dirname(os.environ.get("AURA_SCENARIO_JSON", "aura.json")) or "."
    txt_path = os.path.join(out_dir, "aura_transcript.txt")
    jsonl_path = os.path.join(out_dir, "aura_transcript.jsonl")

    import json

    with open(txt_path, "w") as f_txt, open(jsonl_path, "w") as f_jsonl:
        for sender, receiver, msg in flat:
            # Natural language line
            if hasattr(msg, "to_natural_language"):
                line = f"{sender} -> {receiver}: {msg.to_natural_language()}\n"  # type: ignore[attr-defined]
            else:
                line = f"{sender} -> {receiver}: {str(msg)}\n"
            f_txt.write(line)

            # Structured JSONL
            entry: dict[str, Any] = {"sender": sender, "receiver": receiver}
            if isinstance(msg, AgentAction):
                entry.update(
                    {
                        "type": "agent_action",
                        "action_type": msg.action_type,
                        "argument": msg.argument,
                        "to": msg.to,
                    }
                )
            elif isinstance(msg, SimpleMessage):
                entry.update({"type": "message", "text": msg.to_natural_language()})
            else:
                entry.update({"type": "unknown", "repr": str(msg)})
            f_jsonl.write(json.dumps(entry) + "\n")

    print(f"Transcript written to: {txt_path} and {jsonl_path}")

    # Produce a more readable, turn-grouped transcript and per-agent views
    pretty_path = os.path.join(out_dir, "aura_transcript_pretty.txt")
    views_dir = os.path.join(out_dir, "aura_transcript_views")
    os.makedirs(views_dir, exist_ok=True)

    # Derive agent list from senders (exclude 'Environment')
    agent_names = []
    for sender, _, _ in flat:
        if sender != "Environment" and sender not in agent_names:
            agent_names.append(sender)

    # Build pretty transcript grouped by turns using ONLY SimpleMessage Turn markers
    def to_str(msg_obj: object) -> str:
        try:
            return msg_obj.to_natural_language()  # type: ignore[attr-defined]
        except Exception:
            return str(msg_obj)

    turns: list[list[str]] = []
    current: list[str] = []
    for sender, receiver, msg in flat:
        # Turn boundaries are emitted as SimpleMessage("Turn #n") by the env
        if sender == "Environment" and isinstance(msg, SimpleMessage) and msg.message.startswith("Turn #"):
            if current:
                turns.append(current)
                current = []
            current.append(msg.message.strip())
            continue
        # Only log actual agent actions (not Observations or placeholders)
        if receiver == "Environment" and sender != "Environment" and isinstance(msg, AgentAction):
            if msg.action_type == "none":
                continue
            if msg.to:
                current.append(f"{sender} [{msg.action_type} private to={msg.to}]: {msg.argument}")
            else:
                current.append(f"{sender} [{msg.action_type}]: {msg.argument}")
    if current:
        turns.append(current)

    with open(pretty_path, "w") as f:
        for i, block in enumerate(turns):
            f.write(f"=== Turn {i} ===\n")
            for line in block:
                f.write(line + "\n")
        if not turns:
            f.write("No turns detected. Raw lines were written to aura_transcript.txt.\n")

    # Per-agent view: what each agent could see (public + messages addressed to them + their own)
    for viewer in agent_names:
        view_path = os.path.join(views_dir, f"{viewer.replace(' ', '_').lower()}_view.txt")
        with open(view_path, "w") as vf:
            vf.write(f"Perceived transcript for {viewer}\n\n")
            turn_idx = -1
            for sender, receiver, msg in flat:
                if sender == "Environment" and isinstance(msg, SimpleMessage) and msg.message.startswith("Turn #"):
                    turn_idx += 1
                    vf.write(f"\n=== Turn {turn_idx} ===\n")
                    continue
                if receiver == "Environment" and sender != "Environment" and isinstance(msg, AgentAction):
                    to_list = msg.to or []
                    if (not to_list) or (viewer in to_list) or (sender == viewer):
                        if msg.action_type == "none":
                            continue
                        if to_list:
                            vf.write(f"{sender} [{msg.action_type} private to={to_list}]: {msg.argument}\n")
                        else:
                            vf.write(f"{sender} [{msg.action_type}]: {msg.argument}\n")

    print(f"Also wrote: {pretty_path} and per-agent views in {views_dir}")

    # Structured per-agent directional views: received_from_{X}.txt and sent_to_{Y}.txt
    structured_dir = os.path.join(out_dir, "aura_transcript_views_structured")
    os.makedirs(structured_dir, exist_ok=True)

    # Preprocess actions with turn indices
    actions_with_turn: list[dict[str, Any]] = []
    cur_turn = -1
    for sender, receiver, msg in flat:
        if sender == "Environment" and isinstance(msg, SimpleMessage) and msg.message.startswith("Turn #"):
            cur_turn += 1
            continue
        if receiver == "Environment" and sender != "Environment" and isinstance(msg, AgentAction):
            if msg.action_type == "none":
                continue
            actions_with_turn.append(
                {
                    "turn": cur_turn,
                    "sender": sender,
                    "action_type": msg.action_type,
                    "argument": msg.argument,
                    "to": msg.to or [],
                    "public": not bool(msg.to),
                }
            )

    for viewer in agent_names:
        others = [n for n in agent_names if n != viewer]
        vdir = os.path.join(structured_dir, viewer.replace(" ", "_").lower())
        os.makedirs(vdir, exist_ok=True)

        # Open files for received_from_* and sent_to_* (including ALL for public sends)
        files: dict[str, Any] = {}
        try:
            for name in others:
                files[f"recv_{name}"] = open(os.path.join(vdir, f"received_from_{name.replace(' ', '_').lower()}.txt"), "w")
                files[f"sent_{name}"] = open(os.path.join(vdir, f"sent_to_{name.replace(' ', '_').lower()}.txt"), "w")
            files["sent_ALL"] = open(os.path.join(vdir, "sent_to_ALL.txt"), "w")

            # Iterate actions and route lines
            for act in actions_with_turn:
                turn = act["turn"]
                sender = act["sender"]
                to_list = act["to"]
                arg = act["argument"]
                a_type = act["action_type"]
                is_public = act["public"]

                # Received by viewer from others (public or private to viewer)
                if sender != viewer:
                    if is_public or (viewer in to_list):
                        key = f"recv_{sender}"
                        if key in files:
                            files[key].write(f"Turn {turn}: {sender} [{a_type}{' private' if not is_public else ''}]: {arg}\n")

                # Sent by viewer
                if sender == viewer:
                    if is_public:
                        files["sent_ALL"].write(f"Turn {turn}: {viewer} [{a_type}]: {arg}\n")
                    else:
                        # Write once per explicit recipient
                        for rec in to_list:
                            key = f"sent_{rec}"
                            if key in files:
                                files[key].write(f"Turn {turn}: {viewer} [{a_type} -> {rec}]: {arg}\n")
        finally:
            for f in files.values():
                try:
                    f.close()
                except Exception:
                    pass

    print(f"Structured per-agent directional views in {structured_dir}")


if __name__ == "__main__":
    asyncio.run(main())
