from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.shared.logging import get_agent_logger, log_event, log_llm_result
from agents.shared.models import get_llm
from agents.shared.utils import get_formatted_recent_history

from .prompts import PLANNER_PROMPT, PLANNER_REVISE_PROMPT

logger = get_agent_logger("deep_research", "planner")


@dataclass
class ResearchPlan:
    """Structured research plan produced by the planner."""

    goal: str
    steps: List[str] = field(default_factory=list)
    done_when: str = ""

    def step_list(self) -> str:
        """Return steps as a numbered list string for display / prompts."""
        return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(self.steps))


async def make_plan(state, goal: str) -> ResearchPlan:
    """Generate a research plan for the given confirmed goal.

    Falls back to a minimal two-step plan if the LLM output cannot be parsed.
    """
    log_event(logger, "PLAN_MAKE_START", goal=goal[:200])
    llm = get_llm(strong=True)

    history_limit = 7
    conversation_history = get_formatted_recent_history(state=state, max_messages=history_limit)

    response = await llm.ainvoke(
        [
            SystemMessage(content=PLANNER_PROMPT.format(
                conversation_history=conversation_history,
                history_limit=history_limit
            )),
            HumanMessage(content=f"Research goal: {goal}"),
        ]
    )

    data = _parse_plan_json(response.content)
    if data is None:
        log_event(logger, "PLAN_MAKE_FALLBACK", goal=goal[:200])
        return ResearchPlan(
            goal=goal,
            steps=[
                f"Research: {goal}",
                "Synthesise all findings into a final answer.",
            ],
            done_when="The goal is fully answered with supporting evidence.",
        )

    plan = ResearchPlan(
        goal=goal,
        steps=data.get("steps", [f"Research: {goal}"]),
        done_when=data.get("done_when", "The goal is fully answered with supporting evidence."),
    )
    log_event(logger, "PLAN_MAKE_DONE", goal=goal[:200], step_count=len(plan.steps))
    return plan


async def revise_plan(
    state,
    goal: str,
    existing_steps: List[str],
    done_when: str,
    revision_notes: str,
) -> ResearchPlan:
    """Revise an existing plan in place based on user feedback.

    Unaffected steps should remain unchanged. If parsing fails, the original
    plan is returned unchanged.
    """
    log_event(logger, "PLAN_REVISE_START", goal=goal[:200], revision=revision_notes[:200])
    llm = get_llm(strong=True)

    existing_steps_text = "\n".join(
        f"{i + 1}. {step}" for i, step in enumerate(existing_steps)
    )

    history_limit = 7
    conversation_history = get_formatted_recent_history(state=state, max_messages=history_limit)

    response = await llm.ainvoke(
        [
            SystemMessage(
                content=PLANNER_REVISE_PROMPT.format(
                    goal=goal,
                    conversation_history=conversation_history,
                    history_limit=history_limit,
                    existing_steps=existing_steps_text,
                    done_when=done_when,
                    revision_notes=revision_notes,
                )
            ),
            HumanMessage(content="Revise the plan and return the full updated plan as JSON."),
        ]
    )

    data = _parse_plan_json(response.content)
    if data is None:
        log_event(logger, "PLAN_REVISE_FALLBACK", goal=goal[:200])
        return ResearchPlan(goal=goal, steps=existing_steps, done_when=done_when)

    plan = ResearchPlan(
        goal=goal,
        steps=data.get("steps", existing_steps),
        done_when=data.get("done_when", done_when),
    )
    log_event(logger, "PLAN_REVISE_DONE", goal=goal[:200], step_count=len(plan.steps))
    return plan


def _parse_plan_json(raw: str) -> Optional[dict]:
    """Parse planner JSON output, tolerating fenced markdown."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError, ValueError):
        return None
