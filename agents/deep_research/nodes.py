from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from webbrowser import get

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig, interrupt

from agents.shared.memory import (
    save_message_idempotent,
    save_tool_call,
    save_tool_result,
)
from agents.shared.logging import get_agent_logger
from agents.shared.models import get_llm
from agents.shared.tools import research_tools
from agents.shared.utils import get_formatted_recent_history

from .planner import ResearchPlan, make_plan, revise_plan
from .prompts import (
    EXECUTOR_PROMPT,
    FINISHER_PROMPT,
    GOAL_CONFIRMATION_MESSAGE,
    GOAL_CONFIRMATION_PROMPT,
    GOAL_UPDATER_PROMPT,
    CLARIFIER_PROMPT,
    PLAN_CONFIRMATION_MESSAGE,
    PLAN_CONFIRMATION_PROMPT,
    REFLECTOR_PROMPT,
)
from .state import ResearchState

MAX_ITERATIONS = 10

logger = get_agent_logger("deep_research", "nodes")

_tool_node = ToolNode(tools=research_tools)
_llm_with_tools = get_llm(strong=True).bind_tools(research_tools)



def _get_thread_id(state: ResearchState, config: RunnableConfig) -> str:
    """Resolve thread_id from config first, then state."""
    return (
        config.get("configurable", {}).get("thread_id", "")
        or config.get("metadata", {}).get("thread_id", "")
        or state.get("thread_id", "")
    )


def _get_latest_human_text(state: ResearchState) -> str:
    """Return the latest human message content from state, if any."""
    for msg in reversed(list(state.get("messages", []))):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


# ============================================================================
# Routers
# ============================================================================

def entry_router(state: ResearchState) -> str:
    plan_confirmed = state.get("plan_confirmed", False)
    goal_confirmed = state.get("goal_confirmed", False)

    if plan_confirmed:
        logger.info("ROUTE source=entry_router target=execute_step reason=plan already confirmed goal_confirmed=%s plan_confirmed=%s", goal_confirmed, plan_confirmed)
        return "execute_step"

    if goal_confirmed:
        logger.info("ROUTE source=entry_router target=create_plan reason=goal confirmed but plan not confirmed goal_confirmed=%s plan_confirmed=%s", goal_confirmed, plan_confirmed)
        return "create_plan"

    logger.info("ROUTE source=entry_router target=clarify_goal reason=goal not yet confirmed goal_confirmed=%s plan_confirmed=%s", goal_confirmed, plan_confirmed)
    return "clarify_goal"


def goal_confirmation_router(state: ResearchState) -> str:
    goal_confirmed = state.get("goal_confirmed", False)

    if goal_confirmed:
        logger.info("ROUTE source=goal_confirmation_router target=create_plan reason= goal_confirmed=%s", goal_confirmed)
        return "create_plan"

    logger.info("ROUTE source=goal_confirmation_router target=clarify_goal reason= goal_confirmed=%s revision_notes=%s", goal_confirmed, state.get("goal_revision_notes", ""))
    return "clarify_goal"


def plan_confirmation_router(state: ResearchState) -> str:
    plan_confirmed = state.get("plan_confirmed", False)

    if plan_confirmed:
        logger.info("ROUTE source=plan_confirmation_router target=execute_step reason= plan_confirmed=%s", plan_confirmed)
        return "execute_step"

    logger.info("ROUTE source=plan_confirmation_router target=create_plan reason= plan_confirmed=%s revision_notes=%s", plan_confirmed, state.get("plan_revision_notes", ""))
    return "create_plan"


def execution_router(state: ResearchState) -> str:
    is_done = state.get("is_done", False)
    iteration = state.get("iteration", 0)
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if is_done:
        logger.info("ROUTE source=execution_router target=finish reason=reflector marked done iteration=%s current_step=%s total_steps=%s", iteration, current_step, len(plan))
        return "finish"

    if iteration >= MAX_ITERATIONS:
        logger.info("ROUTE source=execution_router target=finish reason=max iterations reached iteration=%s max_iterations=%s", iteration, MAX_ITERATIONS)
        return "finish"

    if current_step >= len(plan):
        logger.info("ROUTE source=execution_router target=finish reason=all plan steps consumed current_step=%s total_steps=%s", current_step, len(plan))
        return "finish"

    logger.info("ROUTE source=execution_router target=execute_step reason=continue research iteration=%s current_step=%s total_steps=%s", iteration, current_step, len(plan))
    return "execute_step"


# ============================================================================
# Goal clarification / confirmation
# ============================================================================

async def clarify_goal(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("NODE_ENTER node=clarify_goal thread_id=%s", state.get("thread_id", ""))

    original_question = state.get("original_question", "")
    current_goal = state.get("goal", "")
    goal_revision_notes = state.get("goal_revision_notes", "").strip()

    if not original_question:
        original_question = _get_latest_human_text(state)

    llm = get_llm(strong=True)

    # ------------------------------------------------------------------
    # First pass: no existing goal yet
    # ------------------------------------------------------------------
    if not current_goal:
        logger.debug("BRANCH node=clarify_goal branch=first_pass has_current_goal=False original_question=%s", original_question)

        history_limit = 7
        conversation_history = get_formatted_recent_history(state=state, max_messages=history_limit)

        response = await llm.ainvoke(
            [
                SystemMessage(content=CLARIFIER_PROMPT.format(
                    conversation_history=conversation_history,
                    history_limit=history_limit
                )),
                HumanMessage(content=original_question),
            ]
        )

        data = _parse_json(
            response.content,
            fallback={
                "refined_goal": f"Research goal: {original_question}",
                "questions": [],
                "goal_ready": True,
                "reason": "Fallback used due to parse failure.",
            },
        )
        logger.debug("STATE clarify_goal.clarifier_output %s", data)

        refined_goal = data.get("refined_goal", f"Research goal: {original_question}")
        questions = data.get("questions", []) or []
        goal_ready = bool(data.get("goal_ready", not questions))

        logger.debug("BRANCH node=clarify_goal branch=first_pass_result goal_ready=%s question_count=%s", goal_ready, len(questions))

        return {
            "original_question": original_question,
            "goal": refined_goal,
            "clarifying_questions": questions,
            "goal_ready": goal_ready,
            "goal_confirmed": False,
            "goal_revision_notes": "",
            "user_clarification": "",
        }

    # ------------------------------------------------------------------
    # Revision pass: goal exists and the user supplied feedback
    # ------------------------------------------------------------------
    user_feedback = goal_revision_notes or _get_latest_human_text(state)

    logger.debug("BRANCH node=clarify_goal branch=revision_pass has_current_goal=True has_goal_revision_notes=%s", bool(goal_revision_notes))

    history_limit = 7
    conversation_history = get_formatted_recent_history(state=state, max_messages=history_limit)

    response = await llm.ainvoke(
        [
            SystemMessage(
                content=GOAL_UPDATER_PROMPT.format(
                    original_question=original_question,
                    refined_goal=current_goal,
                    conversation_history=conversation_history,
                    history_limit=history_limit,
                    clarifying_questions="\n".join(
                        f"- {q}" for q in state.get("clarifying_questions", [])
                    ) or "(none)",
                    user_clarification=user_feedback,
                )
            ),
            HumanMessage(content="Update the research goal based on the user's latest feedback."),
        ]
    )

    data = _parse_json(
        response.content,
        fallback={
            "updated_goal": current_goal,
            "questions": state.get("clarifying_questions", []),
            "goal_ready": True,
        },
    )
    logger.debug("STATE clarify_goal.goal_updater_output %s", data)

    updated_goal = data.get("updated_goal", current_goal)
    questions = data.get("questions", []) or []
    goal_ready = bool(data.get("goal_ready", not questions))

    logger.debug("BRANCH node=clarify_goal branch=revision_pass_result goal_ready=%s question_count=%s", goal_ready, len(questions))

    return {
        "original_question": original_question,
        "goal": updated_goal,
        "clarifying_questions": questions,
        "goal_ready": goal_ready,
        "goal_confirmed": False,
        "goal_revision_notes": "",
        "user_clarification": "",
    }


async def check_goal_confirmation(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("NODE_ENTER node=check_goal_confirmation thread_id=%s", state.get("thread_id", ""))

    thread_id = (
        config.get("configurable", {}).get("thread_id", "")
        or config.get("metadata", {}).get("thread_id", "")
    )

    original_question = state.get("original_question", "")
    goal = state.get("goal", "")
    questions = state.get("clarifying_questions", [])

    if questions:
        questions_block = (
            "**Open clarification points:**\n"
            + "\n".join(f"- {question}" for question in questions)
        )
    else:
        questions_block = ""

    message = GOAL_CONFIRMATION_MESSAGE.format(
        original_question=original_question,
        goal=goal,
        questions_block=questions_block,
    )

    await save_message_idempotent(thread_id, "assistant", message)

    logger.debug("BRANCH node=check_goal_confirmation branch=interrupt goal_ready=%s question_count=%s", state.get("goal_ready", False), len(questions))

    user_reply = interrupt(
        {
            "type": "goal_confirmation",
            "goal": goal,
            "clarifying_questions": questions,
            "message": message,
        }
    )

    user_reply_text = user_reply if isinstance(user_reply, str) else str(user_reply)
    logger.debug("BRANCH node=check_goal_confirmation branch=resumed user_reply=%s", user_reply_text)

    llm = get_llm(strong=True)

    history_limit = 7
    conversation_history = get_formatted_recent_history(state=state, max_messages=history_limit)

    response = await llm.ainvoke(
        [
            SystemMessage(
                content=GOAL_CONFIRMATION_PROMPT.format(
                    goal=goal,
                    conversation_history=conversation_history,
                    history_limit=history_limit,
                    questions="\n".join(f"- {q}" for q in questions) or "(none)",
                    user_reply=user_reply_text,
                )
            ),
            HumanMessage(content="Return JSON only."),
        ]
    )

    data = _parse_json(
        response.content,
        fallback={
            "is_confirmed": False,
            "revision_notes": user_reply_text,
            "reason": "Fallback used due to parse failure.",
        },
    )
    logger.debug("STATE check_goal_confirmation.confirmation_parse %s", data)

    is_confirmed = bool(data.get("is_confirmed", False))
    revision_notes = (data.get("revision_notes") or "").strip()

    if is_confirmed:
        if revision_notes:
            logger.debug("BRANCH node=check_goal_confirmation branch=confirmed_with_context revision_notes=%s", revision_notes)
            return {
                "goal_confirmed": True,
                "goal_revision_notes": revision_notes,
            }

        logger.debug("BRANCH node=check_goal_confirmation branch=confirmed")
        return {
            "goal_confirmed": True,
            "goal_revision_notes": "",
        }

    logger.debug("BRANCH node=check_goal_confirmation branch=not_confirmed revision_notes=%s", revision_notes or user_reply_text)
    return {
        "goal_confirmed": False,
        "goal_revision_notes": revision_notes or user_reply_text,
    }


# ============================================================================
# Planning / plan confirmation
# ============================================================================

async def create_plan(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("NODE_ENTER node=create_plan thread_id=%s", state.get("thread_id", ""))

    goal = state.get("goal", "")
    existing_plan = state.get("plan", [])
    done_when = state.get("done_when", "")
    plan_revision_notes = state.get("plan_revision_notes", "").strip()
    goal_revision_notes = state.get("goal_revision_notes", "").strip()

    planner_goal = goal
    if goal_revision_notes:
        planner_goal = f"{goal}\n\nAdditional confirmed context from user:\n{goal_revision_notes}"

    if existing_plan and plan_revision_notes:
        logger.debug("BRANCH node=create_plan branch=revise_existing_plan existing_plan_len=%s has_revision_notes=True", len(existing_plan))
        plan: ResearchPlan = await revise_plan(
            state=state,
            goal=planner_goal,
            existing_steps=existing_plan,
            done_when=done_when,
            revision_notes=plan_revision_notes,
        )
    else:
        logger.debug("BRANCH node=create_plan branch=make_new_plan existing_plan_len=%s has_revision_notes=%s", len(existing_plan), bool(plan_revision_notes))
        plan = await make_plan(state=state, goal=planner_goal)

    logger.debug("BRANCH node=create_plan branch=plan_ready step_count=%s done_when=%s", len(plan.steps), plan.done_when)

    return {
        "plan": plan.steps,
        "done_when": plan.done_when,
        "current_step": 0,
        "findings": state.get("findings", ""),
        "is_done": False,
        "iteration": 0 if not existing_plan or plan_revision_notes else state.get("iteration", 0),
        "next_focus": "",
        "plan_confirmed": False,
        "plan_revision_notes": "",
    }


async def check_plan_confirmation(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("NODE_ENTER node=check_plan_confirmation thread_id=%s", state.get("thread_id", ""))

    thread_id = (
        config.get("configurable", {}).get("thread_id", "")
        or config.get("metadata", {}).get("thread_id", "")
    )

    goal = state.get("goal", "")
    plan = state.get("plan", [])
    done_when = state.get("done_when", "")

    plan_steps = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    message = PLAN_CONFIRMATION_MESSAGE.format(
        goal=goal,
        plan_steps=plan_steps,
        done_when=done_when,
    )

    await save_message_idempotent(thread_id, "assistant", message)

    logger.debug("BRANCH node=check_plan_confirmation branch=interrupt step_count=%s", len(plan))

    user_reply = interrupt(
        {
            "type": "plan_confirmation",
            "goal": goal,
            "plan": plan,
            "done_when": done_when,
            "message": message,
        }
    )

    user_reply_text = user_reply if isinstance(user_reply, str) else str(user_reply)
    logger.debug("BRANCH node=check_plan_confirmation branch=resumed user_reply=%s", user_reply_text)

    llm = get_llm(strong=True)

    history_limit = 7
    conversation_history = get_formatted_recent_history(state=state, max_messages=history_limit)

    response = await llm.ainvoke(
        [
            SystemMessage(
                content=PLAN_CONFIRMATION_PROMPT.format(
                    goal=goal,
                    plan=plan_steps,
                    conversation_history=conversation_history,
                    history_limit=10,
                    user_reply=user_reply_text,
                )
            ),
            HumanMessage(content="Return JSON only."),
        ]
    )

    data = _parse_json(
        response.content,
        fallback={
            "is_confirmed": False,
            "revision_notes": user_reply_text,
            "requires_plan_revision": True,
            "reason": "Fallback used due to parse failure.",
        },
    )
    logger.debug("STATE check_plan_confirmation.confirmation_parse %s", data)

    is_confirmed = bool(data.get("is_confirmed", False))
    revision_notes = (data.get("revision_notes") or "").strip()
    requires_plan_revision = bool(data.get("requires_plan_revision", False))

    if is_confirmed and not requires_plan_revision:
        logger.debug("BRANCH node=check_plan_confirmation branch=confirmed_no_revision is_confirmed=%s requires_plan_revision=%s", is_confirmed, requires_plan_revision)
        return {
            "plan_confirmed": True,
            "plan_revision_notes": "",
        }

    if is_confirmed and requires_plan_revision:
        logger.debug("BRANCH node=check_plan_confirmation branch=confirmed_but_revision_needed revision_notes=%s", revision_notes)
        return {
            "plan_confirmed": False,
            "plan_revision_notes": revision_notes or user_reply_text,
        }

    logger.debug("BRANCH node=check_plan_confirmation branch=not_confirmed revision_notes=%s", revision_notes or user_reply_text)
    return {
        "plan_confirmed": False,
        "plan_revision_notes": revision_notes or user_reply_text,
    }


# ============================================================================
# Execution / reflection / finish
# ============================================================================

async def execute_step(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("NODE_ENTER node=execute_step thread_id=%s", state.get("thread_id", ""))

    goal = state.get("goal", "")
    plan = state.get("plan", [])
    current_idx = state.get("current_step", 0)
    findings = state.get("findings", "")
    next_focus = state.get("next_focus", "")
    iteration = state.get("iteration", 0)

    if current_idx >= len(plan):
        logger.debug("BRANCH node=execute_step branch=no_remaining_steps current_idx=%s total_steps=%s", current_idx, len(plan))
        return {"is_done": True}

    current_step_text = plan[current_idx]
    completed_steps = (
        "\n".join(f"✓ {plan[i]}" for i in range(current_idx))
        if current_idx > 0
        else "(none yet)"
    )

    logger.debug("BRANCH node=execute_step branch=run_step current_idx=%s current_step=%s iteration=%s", current_idx, current_step_text, iteration)

    prompt = EXECUTOR_PROMPT.format(
        goal=goal,
        completed_steps=completed_steps,
        findings=findings[-3000:] if findings else "(none yet)",
        current_step=current_step_text,
        next_focus=next_focus or "none",
    )

    step_findings = await _run_with_tools(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Execute step {current_idx + 1}: {current_step_text}"),
        ],
        config,
    )

    step_header = f"\n\n--- Step {current_idx + 1}: {current_step_text} ---\n"
    new_findings = findings + step_header + step_findings

    logger.debug("BRANCH node=execute_step branch=step_complete next_step_index=%s findings_length=%s", current_idx + 1, len(new_findings))

    return {
        "findings": new_findings,
        "current_step": current_idx + 1,
        "iteration": iteration + 1,
        "messages": [AIMessage(content=f"Step {current_idx + 1} completed: {current_step_text}")],
    }


async def reflect(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("NODE_ENTER node=reflect thread_id=%s", state.get("thread_id", ""))

    goal = state.get("goal", "")
    plan = state.get("plan", [])
    findings = state.get("findings", "")
    iteration = state.get("iteration", 0)
    current_step = state.get("current_step", 0)
    done_when = state.get("done_when", f"All {len(plan)} steps are complete.")

    prompt = REFLECTOR_PROMPT.format(
        goal=goal,
        done_when=done_when,
        plan="\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan)),
        steps_completed=current_step,
        total_steps=len(plan),
        iteration=iteration,
        max_iterations=MAX_ITERATIONS,
        findings=findings[-4000:] if findings else "(none yet)",
    )

    llm = get_llm(strong=True)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    data = _parse_json(
        response.content,
        fallback={
            "is_done": False,
            "reason": "Fallback used due to parse failure.",
            "next_focus": "",
        },
    )
    logger.debug("STATE reflect.reflector_output %s", data)

    is_done = bool(data.get("is_done", False))
    next_focus = data.get("next_focus", "")
    reason = data.get("reason", "")

    logger.debug("BRANCH node=reflect branch=reflection_result is_done=%s next_focus=%s reason=%s", is_done, next_focus, reason)

    status_msg = "Research complete." if is_done else f"Continuing research: {reason}"

    return {
        "is_done": is_done,
        "next_focus": next_focus,
        "messages": [AIMessage(content=status_msg)],
    }


async def finish(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("NODE_ENTER node=finish thread_id=%s", state.get("thread_id", ""))

    goal = state.get("goal", "")
    findings = state.get("findings", "")
    thread_id = _get_thread_id(state, config)

    logger.debug("BRANCH node=finish branch=synthesise_final_answer findings_length=%s thread_id=%s", len(findings), thread_id)

    llm = get_llm(strong=True)
    response = await llm.ainvoke(
        [
            SystemMessage(content=FINISHER_PROMPT.format(goal=goal, findings=findings)),
            HumanMessage(content="Write the final research answer now."),
        ]
    )

    content = response.content if isinstance(response.content, str) else str(response.content)

    if thread_id:
        await save_message_idempotent(thread_id, "assistant", content)

    logger.debug("BRANCH node=finish branch=final_answer_ready answer_length=%s", len(content))

    return {
        "final_answer": content,
        "messages": [AIMessage(content=content)],
    }


# ============================================================================
# Internal helpers
# ============================================================================

async def _run_with_tools(messages: List[Any], config: RunnableConfig) -> str:
    current_messages = list(messages)
    thread_id = (
        config.get("configurable", {}).get("thread_id", "")
        or config.get("metadata", {}).get("thread_id", "")
    )

    for attempt in range(5):
        logger.debug("BRANCH node=_run_with_tools branch=llm_turn attempt=%s", attempt + 1)

        response = await _llm_with_tools.ainvoke(current_messages, config=config)

        content = response.content if isinstance(response.content, str) else str(response.content)
        tool_calls = getattr(response, "tool_calls", []) or []

        assistant_msg_id: Optional[str] = None
        if thread_id:
            assistant_msg_id = await save_message_idempotent(thread_id, "assistant", content)
            for tool_call in tool_calls:
                await save_tool_call(
                    message_id=assistant_msg_id,
                    call_id=tool_call.get("id", ""),
                    tool_input=tool_call,
                )

        current_messages.append(response)

        if not tool_calls:
            logger.debug("BRANCH node=_run_with_tools branch=llm_finished_without_tools attempt=%s content_length=%s", attempt + 1, len(content))
            return content

        logger.debug("BRANCH node=_run_with_tools branch=execute_tools attempt=%s tool_call_count=%s", attempt + 1, len(tool_calls))

        tool_result = await _tool_node.ainvoke({"messages": current_messages}, config)
        tool_messages = tool_result.get("messages", [])

        for tool_message in tool_messages:
            if assistant_msg_id:
                await save_tool_result(
                    message_id=assistant_msg_id,
                    call_id=tool_message.tool_call_id,
                    output=tool_message.content,
                )

        current_messages.extend(tool_messages)

    last = current_messages[-1]
    fallback_content = getattr(last, "content", "") or ""
    logger.debug("BRANCH node=_run_with_tools branch=max_attempts_reached content_length=%s", len(str(fallback_content)))
    return fallback_content


def _parse_json(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError, IndexError, AttributeError):
        return fallback
