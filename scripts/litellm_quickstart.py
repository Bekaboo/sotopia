from litellm import acompletion
import logging
import asyncio

logging.basicConfig()

log = logging.getLogger("sotopia.generation")
log.setLevel(logging.INFO)


async def main():
    # Use the /responses route (required for reasoning models to get reasoning content)
    # https://docs.litellm.ai/docs/providers/openai#getting-reasoning-content-in-chatcompletions
    model = "openai/responses/gpt-5"

    response = await acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "What is the 10th element in the Fibonacci sequence?",
            }
        ],
        reasoning_effort="low",
    )

    log.info("Full response: %s", response)
    log.info("Answer: %s", response.choices[0].message["content"])
    log.info("Reasoning: %s", response.choices[0].message.get("reasoning_content"))


if __name__ == "__main__":
    asyncio.run(main())
