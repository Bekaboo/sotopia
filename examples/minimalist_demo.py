"""
This demo serves as a minimal example of how to use the sotopia library.
"""

# 1. Import the sotopia library
# 1.1. Import the `run_async_server` function: In sotopia, we use Python Async
#     API to optimize the throughput.
import asyncio
import logging

from rich.logging import RichHandler

from sotopia.database.persistent_profile import (AgentProfile,
                                                 EnvironmentProfile,
                                                 RelationshipType)
# 1.2. Import the `UniformSampler` class: In sotopia, we use samplers to sample
#     the social tasks.
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server

# 2. Create agent profiles and environment scenarios


# 2.1. Create a simple environment scenario
env_profile = EnvironmentProfile(
    codename="simple_negotiation",
    source="minimalist_demo",
    scenario="""Two people are negotiating over the price of an item.
Person A is selling a rare collectible item and wants to get at least $500 for it.
Person B is interested in buying the item but doesn't want to spend more than $400.
They need to come to an agreement or walk away.""",
    agent_goals=[
        "Sell the collectible item for at least $500, preferably more.",
        "Buy the collectible item for no more than $400.",
    ],
    relationship=RelationshipType.stranger,
    tag="negotiation",
)
env_profile.save()


# 2.2. Create agent profiles
agent_profile_1 = AgentProfile(
    first_name="Alex",
    last_name="Smith",
    personality_and_values="A seller who is willing to negotiate and find a middle ground.",
    tag="seller",
)

agent_profile_2 = AgentProfile(
    first_name="Jordan",
    last_name="Taylor",
    personality_and_values="A buyer who knows what they want and is firm on their budget.",
    tag="buyer",
)

agent_profile_1.save()
agent_profile_2.save()

# 3. Configure the logging and run the server
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

# 3.2. Run the simulation
asyncio.run(
    run_async_server(
        model_dict={
            "env": "gpt-4",
            "agent1": "gpt-4o-mini",
            "agent2": "gpt-4o-mini",
        },
        sampler=UniformSampler(
            env_candidates=[env_profile],
            agent_candidates=[agent_profile_1, agent_profile_2],
        ),
    )
)
