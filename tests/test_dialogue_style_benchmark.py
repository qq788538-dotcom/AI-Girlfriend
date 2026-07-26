from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/benchmark-dialogue-style.py"
SPEC = importlib.util.spec_from_file_location("benchmark_dialogue_style", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_amused_scenario_allows_qualified_real_world_uncertainty() -> None:
    assert MODULE.dialogue_violations(
        "amused",
        "你还真是粗心。希望路上没人盯着你的袜子看。",
    ) == []


def test_jealousy_scenario_requires_a_relationship_cue() -> None:
    assert "缺少恋人式吃醋反应" in MODULE.dialogue_violations(
        "jealousy",
        "聊了什么有意思的事呀？",
    )
    assert MODULE.dialogue_violations(
        "jealousy",
        "哦，聊得挺投入啊。怎么还来找我了？",
    ) == []


def test_amused_scenario_rejects_invented_bystander_reaction() -> None:
    violations = MODULE.dialogue_violations(
        "amused",
        "路上没人盯着你的脚看啦。",
    )

    assert "编造路人反应" in violations


def test_unknown_nickname_rejects_homophone_hallucination() -> None:
    violations = MODULE.dialogue_violations(
        "unknown_nickname",
        "你说的是那个总想偷秘方的痞老板吧？",
    )

    assert "未知称呼应承认不确定" in violations
    assert "未知称呼脑补成相似角色" in violations


def test_dialogue_style_rejects_multiple_questions() -> None:
    assert "连续追问" in MODULE.dialogue_violations(
        "amused",
        "什么时候发现的？后来换了吗？",
    )
