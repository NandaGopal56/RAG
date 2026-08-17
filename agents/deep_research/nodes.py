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
from agents.shared.logging import (
    get_agent_logger,
    log_branch,
    log_llm_result,
    log_node_enter,
    log_node_exit,
    log_route,
)
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
    """Route a fresh or resumed invocation to the correct graph stage.

    Order matters:
        1. If the plan is confirmed, research execution is already in progress.
        2. Else if the goal is confirmed, the next stage is planning.
        3. Otherwise continue / start the goal clarification loop.
    """
    plan_confirmed = state.get("plan_confirmed", False)
    goal_confirmed = state.get("goal_confirmed", False)

    if plan_confirmed:
        log_route(logger, "entry_router", "execute_step", reason="plan already confirmed", goal_confirmed=goal_confirmed, plan_confirmed=plan_confirmed)
        return "execute_step"

    if goal_confirmed:
        log_route(logger, "entry_router", "create_plan", reason="goal confirmed but plan not confirmed", goal_confirmed=goal_confirmed, plan_confirmed=plan_confirmed)
        return "create_plan"

    log_route(logger, "entry_router", "clarify_goal", reason="goal not yet confirmed", goal_confirmed=goal_confirmed, plan_confirmed=plan_confirmed)
    return "clarify_goal"


def goal_confirmation_router(state: ResearchState) -> str:
    """Route after the goal confirmation interrupt."""
    goal_confirmed = state.get("goal_confirmed", False)

    if goal_confirmed:
        log_route(logger, "goal_confirmation_router", "create_plan", reason="", goal_confirmed=goal_confirmed)
        return "create_plan"

    log_route(logger, "goal_confirmation_router", "clarify_goal", reason="", goal_confirmed=goal_confirmed, revision_notes=state.get("goal_revision_notes", ""))
    return "clarify_goal"


def plan_confirmation_router(state: ResearchState) -> str:
    """Route after the plan confirmation interrupt."""
    plan_confirmed = state.get("plan_confirmed", False)

    if plan_confirmed:
        log_route(logger, "plan_confirmation_router", "execute_step", reason="", plan_confirmed=plan_confirmed)
        return "execute_step"

    log_route(logger, "plan_confirmation_router", "create_plan", reason="", plan_confirmed=plan_confirmed, revision_notes=state.get("plan_revision_notes", ""))
    return "create_plan"


def execution_router(state: ResearchState) -> str:
    """Route after reflection to either continue execution or finish.

    Finish conditions:
        - reflector explicitly marked research as done
        - max iteration limit reached
        - all plan steps already consumed
    """
    is_done = state.get("is_done", False)
    iteration = state.get("iteration", 0)
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if is_done:
        log_route(logger, "execution_router", "finish", reason="reflector marked done", iteration=iteration, current_step=current_step, total_steps=len(plan))
        return "finish"

    if iteration >= MAX_ITERATIONS:
        log_route(logger, "execution_router", "finish", reason="max iterations reached", iteration=iteration, max_iterations=MAX_ITERATIONS)
        return "finish"

    if current_step >= len(plan):
        log_route(logger, "execution_router", "finish", reason="all plan steps consumed", current_step=current_step, total_steps=len(plan))
        return "finish"

    log_route(logger, "execution_router", "execute_step", reason="continue research", iteration=iteration, current_step=current_step, total_steps=len(plan))
    return "execute_step"


# ============================================================================
# Goal clarification / confirmation
# ============================================================================

async def clarify_goal(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Refine the research goal based on the original question and user feedback.

    This node does not ask for final approval itself. Instead, it prepares the
    latest goal draft and any open clarification questions, then hands off to
    `check_goal_confirmation`, which pauses for explicit user confirmation.

    Two modes are supported:
        1. First pass:
           - no goal exists yet, so the original user question is analysed
             via `CLARIFIER_PROMPT`.
        2. Revision pass:
           - the user has responded to a goal confirmation prompt with either
             extra context or corrections, so the goal is updated via
             `GOAL_UPDATER_PROMPT`.

    Returns:
        State updates for `goal`, `clarifying_questions`, `goal_ready`, and
        consumed clarification fields.
    """
    log_node_enter(logger, "clarify_goal", thread_id=state.get("thread_id", ""), state=state)

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
        log_branch(logger, "clarify_goal", "first_pass", has_current_goal=False, original_question=original_question)

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
        log_llm_result(logger, "clarify_goal", "clarifier_output", data)

        refined_goal = data.get("refined_goal", f"Research goal: {original_question}")
        questions = data.get("questions", []) or []
        goal_ready = bool(data.get("goal_ready", not questions))

        log_branch(logger, "clarify_goal", "first_pass_result", goal_ready=goal_ready, question_count=len(questions))

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

    log_branch(logger, "clarify_goal", "revision_pass", has_current_goal=True, has_goal_revision_notes=bool(goal_revision_notes))

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
    log_llm_result(logger, "clarify_goal", "goal_updater_output", data)

    updated_goal = data.get("updated_goal", current_goal)
    questions = data.get("questions", []) or []
    goal_ready = bool(data.get("goal_ready", not questions))

    log_branch(logger, "clarify_goal", "revision_pass_result", goal_ready=goal_ready, question_count=len(questions))

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
    """Pause for explicit user confirmation of the refined research goal.

    First execution:
        - Builds a user-facing confirmation message from the current goal and
          any open clarification questions.
        - Calls `interrupt(...)` to pause the graph.

    On resume:
        - The value returned by `interrupt()` is the user's reply.
        - An LLM interprets whether the reply:
            a) confirms the goal,
            b) confirms the goal while adding extra context, or
            c) provides corrections / context without confirming.
        - If the reply is not a clear confirmation, the graph routes back to
          `clarify_goal` with `goal_revision_notes`.
    """
    log_node_enter(logger, "check_goal_confirmation", thread_id=state.get("thread_id", ""), state=state)

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
    
    log_branch(logger, "check_goal_confirmation", "interrupt", goal_ready=state.get("goal_ready", False), question_count=len(questions))

    user_reply = interrupt(
        {
            "type": "goal_confirmation",
            "goal": goal,
            "clarifying_questions": questions,
            "message": message,
        }
    )

    user_reply_text = user_reply if isinstance(user_reply, str) else str(user_reply)
    log_branch(logger, "check_goal_confirmation", "resumed", user_reply=user_reply_text)

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
    log_llm_result(logger, "check_goal_confirmation", "confirmation_parse", data)

    is_confirmed = bool(data.get("is_confirmed", False))
    revision_notes = (data.get("revision_notes") or "").strip()

    if is_confirmed:
        if revision_notes:
            # Confirmed, but the user also added context. We treat it as
            # confirmed and carry the context forward to planning. If you want
            # a stricter loop, route back to clarify_goal here instead.
            log_branch(logger, "check_goal_confirmation", "confirmed_with_context", revision_notes=revision_notes)
            return {
                "goal_confirmed": True,
                "goal_revision_notes": revision_notes,
            }

        log_branch(logger, "check_goal_confirmation", "confirmed")
        return {
            "goal_confirmed": True,
            "goal_revision_notes": "",
        }

    log_branch(logger, "check_goal_confirmation", "not_confirmed", revision_notes=revision_notes or user_reply_text)
    return {
        "goal_confirmed": False,
        "goal_revision_notes": revision_notes or user_reply_text,
    }


# ============================================================================
# Planning / plan confirmation
# ============================================================================

async def create_plan(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Create or revise the research plan.

    Behaviour:
        - If `plan_revision_notes` exists, revise the existing plan in place.
        - Otherwise generate a new plan from the confirmed goal.

    This node does not persist a user-facing plan message. The plan is shown
    to the user by `check_plan_confirmation`, which owns the plan review loop.
    """
    log_node_enter(logger, "create_plan", thread_id=state.get("thread_id", ""), state=state)

    goal = state.get("goal", "")
    existing_plan = state.get("plan", [])
    done_when = state.get("done_when", "")
    plan_revision_notes = state.get("plan_revision_notes", "").strip()
    goal_revision_notes = state.get("goal_revision_notes", "").strip()

    # If the user confirmed the goal and added context, but that context was
    # not folded into the goal string, append it to the planning objective so
    # the planner still sees it.
    planner_goal = goal
    if goal_revision_notes:
        planner_goal = f"{goal}\n\nAdditional confirmed context from user:\n{goal_revision_notes}"

    if existing_plan and plan_revision_notes:
        log_branch(logger, "create_plan", "revise_existing_plan", existing_plan_len=len(existing_plan), has_revision_notes=True)
        plan: ResearchPlan = await revise_plan(
            state=state,
            goal=planner_goal,
            existing_steps=existing_plan,
            done_when=done_when,
            revision_notes=plan_revision_notes,
        )
    else:
        log_branch(logger, "create_plan", "make_new_plan", existing_plan_len=len(existing_plan), has_revision_notes=bool(plan_revision_notes))
        plan = await make_plan(state=state, goal=planner_goal)

    log_branch(logger, "create_plan", "plan_ready", step_count=len(plan.steps), done_when=plan.done_when)

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
    """Pause for explicit user confirmation of the research plan.

    First execution:
        - Presents the current plan via interrupt.

    On resume:
        - Interprets the user's reply with an LLM.
        - Handles three cases:
            1. confirmed, no revision needed -> proceed to execution
            2. confirmed, but revision needed -> revise plan and ask again
            3. not confirmed -> revise plan and ask again
    """
    log_node_enter(logger, "check_plan_confirmation", thread_id=state.get("thread_id", ""), state=state)

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

    log_branch(logger, "check_plan_confirmation", "interrupt", step_count=len(plan))

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
    log_branch(logger, "check_plan_confirmation", "resumed", user_reply=user_reply_text)

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
    log_llm_result(logger, "check_plan_confirmation", "confirmation_parse", data)

    is_confirmed = bool(data.get("is_confirmed", False))
    revision_notes = (data.get("revision_notes") or "").strip()
    requires_plan_revision = bool(data.get("requires_plan_revision", False))

    if is_confirmed and not requires_plan_revision:
        log_branch(logger, "check_plan_confirmation", "confirmed_no_revision", is_confirmed=is_confirmed, requires_plan_revision=requires_plan_revision)
        return {
            "plan_confirmed": True,
            "plan_revision_notes": "",
        }

    if is_confirmed and requires_plan_revision:
        log_branch(logger, "check_plan_confirmation", "confirmed_but_revision_needed", revision_notes=revision_notes)
        return {
            "plan_confirmed": False,
            "plan_revision_notes": revision_notes or user_reply_text,
        }

    log_branch(logger, "check_plan_confirmation", "not_confirmed", revision_notes=revision_notes or user_reply_text)
    return {
        "plan_confirmed": False,
        "plan_revision_notes": revision_notes or user_reply_text,
    }


# ============================================================================
# Execution / reflection / finish
# ============================================================================

async def execute_step(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute the current plan step using the research tool loop."""
    log_node_enter(logger, "execute_step", thread_id=state.get("thread_id", ""), state=state)

    goal = state.get("goal", "")
    plan = state.get("plan", [])
    current_idx = state.get("current_step", 0)
    findings = state.get("findings", "")
    next_focus = state.get("next_focus", "")
    iteration = state.get("iteration", 0)

    if current_idx >= len(plan):
        log_branch(logger, "execute_step", "no_remaining_steps", current_idx=current_idx, total_steps=len(plan))
        return {"is_done": True}

    current_step_text = plan[current_idx]
    completed_steps = (
        "\n".join(f"✓ {plan[i]}" for i in range(current_idx))
        if current_idx > 0
        else "(none yet)"
    )

    log_branch(logger, "execute_step", "run_step", current_idx=current_idx, current_step=current_step_text, iteration=iteration)

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

    log_branch(logger, "execute_step", "step_complete", next_step_index=current_idx + 1, findings_length=len(new_findings))

    return {
        "findings": new_findings,
        "current_step": current_idx + 1,
        "iteration": iteration + 1,
        "messages": [AIMessage(content=f"Step {current_idx + 1} completed: {current_step_text}")],
    }


async def reflect(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Evaluate progress and decide whether more research is needed."""
    log_node_enter(logger, "reflect", thread_id=state.get("thread_id", ""), state=state)

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
    log_llm_result(logger, "reflect", "reflector_output", data)

    is_done = bool(data.get("is_done", False))
    next_focus = data.get("next_focus", "")
    reason = data.get("reason", "")

    log_branch(logger, "reflect", "reflection_result", is_done=is_done, next_focus=next_focus, reason=reason)

    status_msg = "Research complete." if is_done else f"Continuing research: {reason}"

    return {
        "is_done": is_done,
        "next_focus": next_focus,
        "messages": [AIMessage(content=status_msg)],
    }


async def finish(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Write and persist the final research answer."""
    log_node_enter(logger, "finish", thread_id=state.get("thread_id", ""), state=state)

    goal = state.get("goal", "")
    findings = state.get("findings", "")
    thread_id = _get_thread_id(state, config)

    log_branch(logger, "finish", "synthesise_final_answer", findings_length=len(findings), thread_id=thread_id)

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

    log_branch(logger, "finish", "final_answer_ready", answer_length=len(content))

    return {
        "final_answer": content,
        "messages": [AIMessage(content=content)],
    }


# ============================================================================
# Internal helpers
# ============================================================================

async def _run_with_tools(messages: List[Any], config: RunnableConfig) -> str:
    """Run a bounded LLM + tool loop for a single research step.

    The loop:
        1. Ask the LLM for the next action.
        2. Persist the assistant message and any tool calls.
        3. Execute tool calls through the shared ToolNode.
        4. Persist tool outputs.
        5. Repeat until the LLM returns a normal response with no tool calls.

    Returns:
        The final assistant text for the step.
    """
    current_messages = list(messages)
    thread_id = (
        config.get("configurable", {}).get("thread_id", "")
        or config.get("metadata", {}).get("thread_id", "")
    )

    for attempt in range(5):
        log_branch(logger, "_run_with_tools", "llm_turn", attempt=attempt + 1)

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
            log_branch(logger, "_run_with_tools", "llm_finished_without_tools", attempt=attempt + 1, content_length=len(content))
            return content

        log_branch(logger, "_run_with_tools", "execute_tools", attempt=attempt + 1, tool_call_count=len(tool_calls))

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
    log_branch(logger, "_run_with_tools", "max_attempts_reached", content_length=len(str(fallback_content)))
    return fallback_content


def _parse_json(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Parse JSON from LLM output, tolerating fenced markdown.

    Args:
        raw: Raw model output.
        fallback: Value returned if parsing fails.

    Returns:
        Parsed dict or the fallback.
    """
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