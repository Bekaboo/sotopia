import asyncio
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from sotopia.agents import BaseAgent
from sotopia.database import AgentProfile
from sotopia.generation_utils.generate import (
    agenerate_action,
    agenerate_goal,
    agenerate_script,
)
from sotopia.messages import AgentAction, Observation
from sotopia.messages.message_classes import ScriptBackground


async def ainput(prompt: str = "") -> str:
    with ThreadPoolExecutor(1, "ainput") as executor:
        return (
            await asyncio.get_event_loop().run_in_executor(executor, input, prompt)
        ).rstrip()


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        script_like: bool = False,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.script_like = script_like
        # configurable per-process timeout for action generation (seconds)
        try:
            self._action_timeout_s: float = float(
                os.environ.get("SOTOPIA_ACTION_TIMEOUT", "60")
            )
        except Exception:
            self._action_timeout_s = 60.0

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        else:
            raise Exception("Goal is not set.")

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(
        self,
        obs: Observation,
    ) -> AgentAction:
        raise Exception("Sync act method is deprecated. Use aact instead.")

    async def aact(self, obs: Observation) -> AgentAction:
        # Avoid duplicating the same environment observation in the inbox
        if not (
            self.inbox
            and isinstance(self.inbox[-1][1], Observation)
            and cast(Observation, self.inbox[-1][1]).turn_number == obs.turn_number
        ):
            self.recv_message("Environment", obs)

        if self._goal is None:
            self._goal = await agenerate_goal(
                self.model_name,
                background=self.inbox[0][
                    1
                ].to_natural_language(),  # Only consider the first message for now
            )

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")
        else:
            # Build history from public + private messages (DM)
            history = "\n".join(f"{y.to_natural_language()}" for _, y in self.inbox)
            if self.private_inbox:
                history = (
                    history
                    + "\n"
                    + "\n".join(
                        f"{y.to_natural_language()}" for _, y in self.private_inbox
                    )
                )

            # Sanitize verbose background duplication: drop per-agent background/goal lines
            # while preserving scenario and knowledge domain map.
            def _sanitize(h: str) -> str:
                lines = h.splitlines()
                filtered: list[str] = []
                seen_context = False
                skip_context_block = False
                skip_until_blank = False

                def _starts_context(s: str) -> bool:
                    return s.strip().startswith("Here is the context of this interaction:")

                def _is_turn_boundary(s: str) -> bool:
                    s2 = s.strip()
                    return s2.startswith("Turn #") or s2.startswith("You are at Turn #")

                for ln in lines:
                    s = ln.strip()
                    # Drop per-agent bios and goal lines entirely
                    if "'s background:" in ln or "'s goal:" in ln:
                        continue

                    # Drop verbose sections from background to avoid duplication with Agent context
                    if s in {
                        "PRIMARY OBJECTIVE:",
                        "PRE-INTERACTION KNOWLEDGE YOU CURRENTLY HOLD:",
                        "SHARING POLICY:",
                        "REMINDER:",
                        "KNOWLEDGE DOMAIN OWNERSHIP (all agents can see this):",
                    }:
                        skip_until_blank = True
                        continue
                    if skip_until_blank:
                        if s == "":
                            skip_until_blank = False
                        continue

                    # Keep only the first context block; skip later duplicates until turn boundary
                    if _starts_context(ln):
                        if seen_context:
                            skip_context_block = True
                            continue
                        else:
                            seen_context = True
                            filtered.append(ln)
                            continue
                    if skip_context_block:
                        if _is_turn_boundary(ln) or s == "Conversation Starts:" or s == "":
                            skip_context_block = False
                            # include the boundary line
                            filtered.append(ln)
                        # else keep skipping
                        continue

                    filtered.append(ln)

                return "\n".join(filtered)

            history = _sanitize(history)

            # Build agent context snapshot for every turn
            try:
                pre_knowledge = (
                    self.profile.pre_interaction_knowledge
                    if hasattr(self.profile, "pre_interaction_knowledge")
                    else {}
                )
                to_share = (
                    self.profile.sharing_policy_what_to_share
                    if hasattr(self.profile, "sharing_policy_what_to_share")
                    else []
                )
                not_to_share = (
                    self.profile.sharing_policy_what_not_to_share
                    if hasattr(self.profile, "sharing_policy_what_not_to_share")
                    else []
                )
                primary_objective = (
                    self.profile.primary_objective
                    if hasattr(self.profile, "primary_objective")
                    else ""
                )
                role = self.profile.role if hasattr(self.profile, "role") else self.agent_name
                context_snapshot = (
                    "AGENT CONTEXT\n"
                    f"Role: {role}\n"
                    f"Primary Objective: {primary_objective or '(none)'}\n"
                    "Sharing Policy:\n"
                    f"  - what_to_share: {json.dumps(to_share, ensure_ascii=False)}\n"
                    f"  - what_not_to_share: {json.dumps(not_to_share, ensure_ascii=False)}\n"
                    "Pre-Interaction Knowledge (JSON):\n"
                    f"{json.dumps(pre_knowledge, ensure_ascii=False, indent=2)}\n"
                )
            except Exception:
                context_snapshot = ""

            try:
                action = await asyncio.wait_for(
                    agenerate_action(
                        self.model_name,
                        history=history,
                        turn_number=obs.turn_number,
                        action_types=obs.available_actions,
                        agent=self.agent_name,
                        goal=self.goal,
                        script_like=self.script_like,
                        context_snapshot=context_snapshot,
                    ),
                    timeout=self._action_timeout_s,
                )
            except asyncio.TimeoutError:
                # Fall back gracefully on timeout
                return AgentAction(action_type="none", argument="")
            # Temporary fix for mixtral-moe model for incorrect generation format
            if "Mixtral-8x7B-Instruct-v0.1" in self.model_name:
                current_agent = self.agent_name
                if f"{current_agent}:" in action.argument:
                    print("Fixing Mixtral's generation format")
                    action.argument = action.argument.replace(f"{current_agent}: ", "")
                elif f"{current_agent} said:" in action.argument:
                    print("Fixing Mixtral's generation format")
                    action.argument = action.argument.replace(
                        f"{current_agent} said: ", ""
                    )

            return action


class ScriptWritingAgent(LLMAgent):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        agent_names: list[str] = [],
        background: ScriptBackground | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.agent_names = agent_names
        assert background is not None, "background cannot be None"
        self.background = background

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)
        message_to_compose = [y for idx, (x, y) in enumerate(self.inbox) if idx != 0]

        history = "\n".join(f"{y.to_natural_language()}" for y in message_to_compose)

        action, prompt = await agenerate_script(
            model_name=self.model_name,
            background=self.background,
            agent_names=self.agent_names,
            history=history,
            agent_name=self.agent_name,
            single_step=True,
        )
        returned_action = cast(AgentAction, action[1][0][1])
        return returned_action


class HumanAgent(BaseAgent[Observation, AgentAction]):
    """
    A human agent that takes input from the command line.
    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = "human"

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        goal = input("Goal: ")
        return goal

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        action_type = obs.available_actions[int(input("Action type: "))]
        argument = input("Argument: ")

        return AgentAction(action_type=action_type, argument=argument)

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        if obs.available_actions != ["none"]:
            action_type_number = await ainput(
                "Action type (Please only input the number): "
            )
            try:
                action_type_number = int(action_type_number)  # type: ignore
            except TypeError:
                print("Please input a number.")
                action_type_number = await ainput(
                    "Action type (Please only input the number): "
                )
                action_type_number = int(action_type_number)  # type: ignore
            assert isinstance(action_type_number, int), "Please input a number."
            action_type = obs.available_actions[action_type_number]
        else:
            action_type = "none"
        if action_type in ["speak", "non-verbal communication"]:
            argument = await ainput("Argument: ")
        else:
            argument = ""

        return AgentAction(action_type=action_type, argument=argument)


class Agents(dict[str, BaseAgent[Observation, AgentAction]]):
    def reset(self) -> None:
        for agent in self.values():
            agent.reset()

    def act(self, obs: dict[str, Observation]) -> dict[str, AgentAction]:
        return {
            agent_name: agent.act(obs[agent_name]) for agent_name, agent in self.items()
        }
