from sotopia.generation_utils import Script, generate_episode
from sotopia.generation_utils.generate import (
    ListOfIntOutputParser,
    generate,
)
from sotopia.messages import ScriptEnvironmentResponse


def test_generate_episode() -> None:
    """
    Test that the scenario generator works
    """
    scenario = generate_episode("gpt-3.5-turbo")
    assert isinstance(scenario, Script)


def test_generate_list_integer() -> None:
    """
    Test that the integer generator works
    """
    length, lower, upper = 5, -10, 10
    l = generate(
        "gpt-3.5-turbo",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(length, (lower, upper)),
    )
    assert isinstance(l, list)
    assert len(l) == length
    assert all(isinstance(i, int) for i in l)
    assert all(lower <= i <= upper for i in l)
