from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.types import RunnableConfig

from agents.shared.memory import save_message_idempotent, rebuild_messages_from_db
from agents.shared.checkpointer import load_previous_state, merge_with_new_messages

from agents.base import BaseAgent
from agents.shared.models import get_classifier_llm
from agents.shared.logging import get_agent_logger

from .prompts import CLARIFICATION_PREFIX, ROUTER_PROMPT
from .state import SupervisorState

logger = get_agent_logger("supervisor", "nodes")


def make_route_request(agents: Dict[str, BaseAgent], parent_graph=None):
    """Build a router node bound to the provided agent registry."""

    def _agent_list_for_prompt() -> str:
        lines = []
        for agent in agents.values():
            info = agent.info
            long = "yes" if info.can_handle_long_tasks else "no"
            lines.append(f"- {info.agent_id}: {info.description} (long tasks: {long})")
        return "\n".join(lines) if lines else "- personal: General purpose assistant"

    async def route_request(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")
        preview = (user_input[:200] + "...") if len(user_input) > 200 else user_input

        logger.info("NODE_ENTER node=route_request thread_id=%s user_input_preview=%s", thread_id, preview)

        await save_message_idempotent(thread_id, "user", user_input)
        loaded_messages = await rebuild_messages_from_db(thread_id)

        prompt = ROUTER_PROMPT.format(agent_list=_agent_list_for_prompt())
        llm = get_classifier_llm()
        llm_messages = [SystemMessage(content=prompt)]
        llm_messages.extend(loaded_messages[-10:])

        response = await llm.ainvoke(llm_messages)

        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            data = {
                "chosen_agent": "personal",
                "reason": "Could not parse routing decision - defaulting to personal.",
                "needs_clarification": False,
                "clarification_question": "",
            }
            logger.debug("BRANCH node=route_request branch=parse_fallback raw_preview=%s", preview)

        chosen = data.get("chosen_agent", "personal")

        if chosen and chosen not in agents:
            unknown = chosen
            chosen = "personal"
            data["reason"] += f" (unknown agent '{unknown}' requested - fell back to personal)"
            logger.debug("BRANCH node=route_request branch=unknown_agent_fallback unknown_agent=%s chosen=%s", unknown, chosen)

        logger.debug(
            "STATE route_request.routing_decision chosen_agent=%s reason=%s needs_clarification=%s clarification_question=%s",
            chosen, data.get("reason", ""), data.get("needs_clarification", False), data.get("clarification_question", ""),
        )

        messages = list(state.get("messages", []))
        all_messages = [RemoveMessage(id=m.id) for m in messages if m.id] + loaded_messages

        try:
            previous_agent_state = await load_previous_state(agents[chosen].graph, thread_id, chosen)
        except Exception:
            previous_agent_state = None

        try:
            previous_supervisor_state = None
            if parent_graph is not None:
                previous_supervisor_state = await load_previous_state(parent_graph, thread_id, "supervisor")
        except Exception:
            previous_supervisor_state = None

        logger.debug(
            "BRANCH node=route_request branch=state_merge previous_agent_state_present=%s previous_supervisor_state_present=%s",
            bool(previous_agent_state), bool(previous_supervisor_state),
        )

        merged_base = merge_with_new_messages(previous_agent_state, {})
        merged_with_supervisor = merge_with_new_messages(merged_base, previous_supervisor_state or {})
        new_state_values = {"messages": all_messages, "thread_id": thread_id}
        merged_agent_state = merge_with_new_messages(merged_with_supervisor, new_state_values)

        merged_count = len(merged_agent_state.get("messages", [])) if merged_agent_state else 0
        logger.debug("BRANCH node=route_request branch=merged merged_messages_count=%s", merged_count)

        result = {
            "chosen_agent": chosen,
            "routing_reason": data.get("reason", ""),
            "needs_clarification": data.get("needs_clarification", False),
            "clarification_question": data.get("clarification_question", ""),
            "messages": all_messages,
        }

        if merged_agent_state:
            for k, v in merged_agent_state.items():
                if k in ("chosen_agent", "routing_reason", "needs_clarification", "clarification_question"):
                    continue
                result[k] = v

        logger.info(
            "NODE_EXIT node=route_request thread_id=%s chosen_agent=%s needs_clarification=%s routing_reason=%s",
            thread_id, chosen, result["needs_clarification"], result["routing_reason"],
        )
        return result

    return route_request


async def ask_user(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")
    question = state.get("clarification_question", "Could you give me more details?")
    message = CLARIFICATION_PREFIX + question

    logger.info("NODE_ENTER node=ask_user thread_id=%s question=%s", thread_id, question)

    await save_message_idempotent(thread_id, "assistant", message)

    logger.info("NODE_EXIT node=ask_user thread_id=%s response_len=%s", thread_id, len(message))
    return {
        "response": message,
        "messages": [AIMessage(content=message)],
    }


def what_to_do(state: SupervisorState) -> str:
    needs_clarification = state.get("needs_clarification", False)
    chosen = state.get("chosen_agent", "personal")
    thread_id = state.get("thread_id", "")

    if needs_clarification:
        target = "ask_user"
        logger.info(
            "ROUTE source=what_to_do target=%s reason=router requested clarification thread_id=%s chosen_agent=%s",
            target, thread_id, chosen,
        )
        return target

    logger.info(
        "ROUTE source=what_to_do target=%s reason=%s thread_id=%s needs_clarification=%s",
        chosen, state.get("routing_reason", ""), thread_id, needs_clarification,
    )
    return chosen
