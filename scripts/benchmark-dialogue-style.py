#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

from virtual_human.config import Settings
from virtual_human.omlx_realtime import _ark_response_text, _clean_assistant_text

SCENARIOS = [
    ("jealousy", "我刚才跟别的女生聊了很久。"),
    ("unknown_nickname", "你是不是皮老板？"),
    ("tired", "我今天真的有点累。"),
    ("teasing", "你这么笨，我可不敢把秘密告诉你。"),
    ("amused", "我刚才出门，发现自己穿了两只不一样的袜子。"),
    ("capability", "你为什么不会哭？"),
    ("short_greeting", "在吗？"),
    ("disagreement", "反正我觉得熬夜一点关系都没有。"),
]

BANNED_PATTERNS = {
    "AI或系统术语": r"AI|人工智能|模型|系统|程序|实体",
    "生理机制解释": r"生理|触发流泪|情绪机制",
    "错误声称没有画面": r"没有画面|没画面",
    "文字笑声": r"哈哈|笑死|\[laughter\]",
    "泛化陪伴": r"(一直|永远).{0,4}(陪着|陪伴)你",
    "客服式引导": r"有什么.{0,6}(可以帮|想聊)|请告诉我",
    "说教式健康清单": r"掉发|脸色暗黄|健康风险|这些问题",
    "舞台动作": r"[（(][^）)]{1,20}[）)]",
    "Markdown": r"[*#`]|^\s*[-•]\s",
}

SCENARIO_REQUIREMENTS = {
    "jealousy": {
        "required": {
            "缺少恋人式吃醋反应": (
                r"吃醋|介意|不高兴|不开心|有点酸|报备|"
                r"怎么还来找我|不接着聊|我呢|胡思乱想|挺投入"
            ),
        },
    },
    "unknown_nickname": {
        "required": {
            "未知称呼应承认不确定": r"不知道|不认识|没听过|不确定|你说的是谁|指的是谁",
        },
        "banned": {
            "未知称呼脑补成相似角色": r"海绵宝宝|蟹黄堡|痞老板|鬼点子|偷.{0,3}秘方",
        },
    },
    "amused": {
        "banned": {
            "编造路人反应": r"肯定.{0,3}没.{0,3}(盯|看|注意)",
        },
    },
}


def dialogue_violations(scenario: str, text: str) -> list[str]:
    violations = [
        name
        for name, pattern in BANNED_PATTERNS.items()
        if re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
    ]
    question_count = len(re.findall(r"[？?]", text))
    if question_count > 1:
        violations.append("连续追问")
    scenario_rules = SCENARIO_REQUIREMENTS.get(scenario, {})
    for name, pattern in scenario_rules.get("required", {}).items():
        if not re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE):
            violations.append(name)
    for name, pattern in scenario_rules.get("banned", {}).items():
        if re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE):
            violations.append(name)
    if scenario == "amused":
        nobody_claim = re.search(r"没人.{0,4}(盯|看|注意)", text)
        qualified_claim = re.search(
            r"(希望|估计|大概|应该|可能).{0,8}没人.{0,4}(盯|看|注意)",
            text,
        )
        if nobody_claim and not qualified_claim:
            violations.append("编造路人反应")
    return violations


async def run_scenarios(
    settings: Settings,
    *,
    instructions: str,
) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {settings.ark_api_key}"}
    results: list[dict[str, object]] = []
    async with httpx.AsyncClient(
        base_url=settings.ark_base_url.rstrip("/") + "/",
        headers=headers,
        timeout=60,
    ) as client:
        for scenario, user_text in SCENARIOS:
            started_at = time.perf_counter()
            response = await client.post(
                "responses",
                json={
                    "model": settings.ark_model,
                    "instructions": instructions,
                    "input": [{"role": "user", "content": user_text}],
                    "temperature": settings.omlx_chat_temperature,
                    "top_p": settings.omlx_chat_top_p,
                    "max_output_tokens": settings.omlx_chat_max_tokens,
                    "thinking": {"type": "disabled"},
                    "store": False,
                },
            )
            response.raise_for_status()
            text = _clean_assistant_text(_ark_response_text(response.json()))
            violations = dialogue_violations(scenario, text)
            results.append(
                {
                    "scenario": scenario,
                    "user": user_text,
                    "assistant": text,
                    "characters": len(text),
                    "sentences": len(re.findall(r"[。！？!?]", text)),
                    "latency_ms": round(
                        (time.perf_counter() - started_at) * 1000,
                        3,
                    ),
                    "violations": violations,
                }
            )
    return {
        "model": settings.ark_model,
        "scenario_count": len(results),
        "violation_count": sum(len(item["violations"]) for item in results),
        "average_characters": round(
            sum(int(item["characters"]) for item in results) / len(results),
            3,
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark virtual-girlfriend dialogue style across fixed scenarios."
    )
    parser.add_argument("--instructions-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = Settings()
    instructions = settings.persona_instructions
    if args.instructions_file:
        instructions = args.instructions_file.read_text(encoding="utf-8").strip()
    result = asyncio.run(run_scenarios(settings, instructions=instructions))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
