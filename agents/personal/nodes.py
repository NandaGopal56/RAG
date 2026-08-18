from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Sequence

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig

from agents.shared.memory import (
    save_tool_call,
    save_tool_result,
    save_message_idempotent
)
from agents.shared.models import get_classifier_llm, get_llm
from agents.shared.tools import personal_tools
from agents.shared.video_buffer import video_buffer
from agents.shared.logging import get_agent_logger

from .prompts import STEP_CLASSIFIER_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
from .state import PersonalState

logger = get_agent_logger("personal", "nodes")

_tool_node = ToolNode(tools=personal_tools)
_llm_with_tools = get_llm().bind_tools(personal_tools)
MAX_HISTORY_TURNS = 6


def _thread_id(config: RunnableConfig, state: PersonalState) -> str:
    return config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")


async def decide_steps(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    logger.info("NODE_ENTER node=decide_steps thread_id=%s", tid)

    messages      = list(state.get("messages", []))
    last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        logger.info("NODE_EXIT node=decide_steps thread_id=%s steps_needed=[]", tid)
        return {"steps_needed": []}

    user_text = last_user_msg.content if isinstance(last_user_msg.content, str) else str(last_user_msg.content)

    response = await get_classifier_llm().ainvoke([
        SystemMessage(content=STEP_CLASSIFIER_PROMPT),
        HumanMessage(content=user_text),
    ])

    try:
        steps = json.loads(response.content)
        steps = steps if isinstance(steps, list) else []
    except (json.JSONDecodeError, ValueError):
        steps = []

    logger.info("NODE_EXIT node=decide_steps thread_id=%s steps_needed=%s", tid, steps)
    return {"steps_needed": steps}


def pick_context_steps(state: PersonalState) -> List[str]:
    tid = state.get("thread_id", "")
    logger.info("NODE_ENTER node=pick_context_steps thread_id=%s", tid)

    step_map = {
        "video_capture":   "grab_video_frame",
        "web_search":      "fetch_web_context",
        "document_search": "fetch_doc_context",
    }
    steps_needed = state.get("steps_needed", [])
    selected     = [step_map[s] for s in steps_needed if s in step_map]
    chosen = selected if selected else ["call_llm"]

    logger.info(
        "ROUTE source=pick_context_steps target=%s reason=%s steps_needed=%s selected=%s",
        ",".join(chosen),
        "context steps selected by classifier" if selected else "no context needed",
        steps_needed, chosen,
    )
    return chosen


async def grab_video_frame(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    logger.info("NODE_ENTER node=grab_video_frame thread_id=%s", tid)

    frame = video_buffer.latest()
    if frame is None:
        logger.debug("BRANCH node=grab_video_frame branch=no_frame_available")
        logger.info("NODE_EXIT node=grab_video_frame thread_id=%s frame_available=False", tid)
        return {"messages": [HumanMessage(content=[{"type": "text", "text": "No camera frame available right now."}])]}

    b64      = base64.b64encode(frame).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    logger.info("NODE_EXIT node=grab_video_frame thread_id=%s frame_available=True", tid)
    return {"messages": [HumanMessage(content=[
        {"type": "text",      "text": "Here is the current camera view:"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ])]}


async def fetch_web_context(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    logger.info("NODE_ENTER node=fetch_web_context thread_id=%s", tid)
    logger.info("NODE_EXIT node=fetch_web_context thread_id=%s status=stub", tid)
    return {}


async def fetch_doc_context(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    logger.info("NODE_ENTER node=fetch_doc_context thread_id=%s", tid)
    logger.info("NODE_EXIT node=fetch_doc_context thread_id=%s status=stub", tid)
    return {}


def join_context(state: PersonalState) -> PersonalState:
    tid = state.get("thread_id", "")
    logger.info("NODE_ENTER node=join_context thread_id=%s", tid)
    logger.info("NODE_EXIT node=join_context thread_id=%s", tid)
    return state


async def call_llm(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = _thread_id(config, state)
    messages  = list(state.get("messages", []))
    summary   = state.get("summary", "")
    logger.info(
        "NODE_ENTER node=call_llm thread_id=%s summary_len=%s message_count=%s",
        thread_id, len(summary) if summary else 0, len(messages),
    )

    prompt: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        prompt.append(SystemMessage(content=f"Summary of earlier conversation:\n{summary}"))
    prompt.extend(_trim_to_recent_turns(messages, MAX_HISTORY_TURNS))

    response         = await _llm_with_tools.ainvoke(prompt, config=config)
    content          = response.content if isinstance(response.content, str) else ""
    assistant_msg_id = await save_message_idempotent(thread_id, "assistant", content)

    tool_calls = getattr(response, "tool_calls", []) or []
    logger.debug(
        "BRANCH node=call_llm branch=response_ready assistant_msg_id=%s tool_call_count=%s",
        assistant_msg_id, len(tool_calls),
    )

    for tc in tool_calls:
        await save_tool_call(message_id=assistant_msg_id, call_id=tc.get("id", ""), tool_input=tc)

    logger.info(
        "NODE_EXIT node=call_llm thread_id=%s response_len=%s tool_call_count=%s",
        thread_id, len(content), len(tool_calls),
    )
    return {
        "messages":                     [response],
        "current_assistant_message_id": assistant_msg_id,
    }


def what_next(state: PersonalState) -> str:
    tid = state.get("thread_id", "")
    messages = list(state.get("messages", []))
    if not messages:
        target = "done"
        logger.info("ROUTE source=what_next target=%s reason=no messages thread_id=%s", target, tid)
        return target

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        target = "run_tools"
        logger.info(
            "ROUTE source=what_next target=%s reason=assistant issued tool calls thread_id=%s tool_call_count=%s",
            target, tid, len(last_msg.tool_calls),
        )
        return target

    if len(messages) > 12:
        target = "compress_history"
        logger.info(
            "ROUTE source=what_next target=%s reason=message history exceeds threshold thread_id=%s message_count=%s",
            target, tid, len(messages),
        )
        return target

    logger.info("ROUTE source=what_next target=done reason=conversation turn complete thread_id=%s", tid)
    return "done"


async def run_tools(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    logger.info("NODE_ENTER node=run_tools thread_id=%s", tid)

    messages   = list(state.get("messages", []))
    last_msg   = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", None) if last_msg else None
    if not tool_calls:
        logger.info("NODE_EXIT node=run_tools thread_id=%s tool_messages=0", tid)
        return {}

    result        = await _tool_node.ainvoke({"messages": messages}, config)
    tool_messages = result.get("messages", [])

    assistant_msg_id = state.get("current_assistant_message_id")
    for tm in tool_messages:
        if assistant_msg_id:
            await save_tool_result(message_id=assistant_msg_id, call_id=tm.tool_call_id, output=tm.content)

    logger.info("NODE_EXIT node=run_tools thread_id=%s tool_messages=%s", tid, len(tool_messages))
    return {"messages": tool_messages}


async def compress_history(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    logger.info("NODE_ENTER node=compress_history thread_id=%s", tid)

    existing_summary = state.get("summary", "")
    messages         = list(state.get("messages", []))

    msg_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
        f"{m.content if isinstance(m.content, str) else str(m.content)}"
        for m in messages if isinstance(m, (HumanMessage, AIMessage))
    )

    prompt   = SUMMARY_PROMPT.format(existing_summary=existing_summary or "None yet.", new_messages=msg_text)
    response = await get_llm().ainvoke([HumanMessage(content=prompt)])
    summary_len = len(response.content) if response and getattr(response, "content", None) else 0
    logger.info("NODE_EXIT node=compress_history thread_id=%s summary_len=%s", tid, summary_len)
    return {"summary": response.content}


def _trim_to_recent_turns(messages: Sequence[Any], max_turns: int) -> List[Any]:
    if not messages:
        return []

    last_human_idx = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
        None,
    )
    if last_human_idx is None:
        return list(messages)

    kept, turns_seen = [], 0
    for msg in reversed(messages[:last_human_idx]):
        kept.append(msg)
        if isinstance(msg, HumanMessage):
            turns_seen += 1
            if turns_seen >= max_turns:
                break

    kept.reverse()
    kept.extend(messages[last_human_idx:])
    return kept
