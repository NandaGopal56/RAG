from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import (
    get_checkpointer,
    load_previous_state,
    merge_with_new_messages
)
from agents.personal.nodes import (
    call_llm,
    compress_history,
    decide_steps,
    fetch_doc_context,
    fetch_web_context,
    grab_video_frame,
    join_context,
    pick_context_steps,
    run_tools,
    what_next,
)
from agents.personal.state import PersonalState
from agents.shared.logging import get_agent_logger, ensure_invocation_session

logger = get_agent_logger("personal", "graph")


def build_personal_graph(checkpointer=None):
    g = StateGraph(PersonalState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("decide_steps",       decide_steps)

    # Parallel context-gathering branches
    g.add_node("grab_video_frame",   grab_video_frame)
    g.add_node("fetch_web_context",  fetch_web_context)
    g.add_node("fetch_doc_context",  fetch_doc_context)

    g.add_node("join_context",       join_context)
    g.add_node("call_llm",           call_llm)
    g.add_node("run_tools",          run_tools)
    g.add_node("compress_history",   compress_history)

    # -- Edges ----------------------------------------------------------------
    g.add_edge(START, "decide_steps")

    # Fan-out: decide_steps -> one or more context nodes (or straight to call_llm)
    g.add_conditional_edges(
        "decide_steps",
        pick_context_steps,
        path_map={
            "grab_video_frame":  "grab_video_frame",
            "fetch_web_context": "fetch_web_context",
            "fetch_doc_context": "fetch_doc_context",
            "call_llm":          "call_llm",
        },
    )

    # Fan-in: all context branches converge before calling the LLM
    g.add_edge("grab_video_frame",  "join_context")
    g.add_edge("fetch_web_context", "join_context")
    g.add_edge("fetch_doc_context", "join_context")
    g.add_edge("join_context",      "call_llm")

    # After the LLM responds: run tools, compress history, or finish
    g.add_conditional_edges(
        "call_llm",
        what_next,
        path_map={
            "run_tools":        "run_tools",
            "compress_history": "compress_history",
            "done":             END,
        },
    )

    # Tool loop: keep calling the LLM until it stops making tool calls
    g.add_edge("run_tools",       "call_llm")
    g.add_edge("compress_history", END)

    if checkpointer is not None:
        logger.info("GRAPH_BUILD component=personal checkpointer=%s", repr(checkpointer))
    graph = g.compile(checkpointer=checkpointer)
    logger.info("GRAPH_COMPILED component=personal")
    return graph


#---------------------------------------------------------------------------
# PersonalAgent — the BaseAgent wrapper the supervisor talks to
# ---------------------------------------------------------------------------

class PersonalAgent(BaseAgent):
    """
    Everyday chat assistant with memory, vision, and tool use.
    Handles single-turn and short multi-turn conversations.
    """

    def __init__(self):
        self._graph = None   # built lazily on first call

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="personal",
            name="Personal Assistant",
            description=(
                "Handles everyday chat, quick questions, task execution, "
                "web search, code running, and live camera vision. "
                "Best for short to medium conversations."
            ),
            can_handle_long_tasks=False,
        )

    @property
    def graph(self):
        if self._graph is None:
            # Standalone agent: compile with its own persistent checkpointer
            self._graph = build_personal_graph(get_checkpointer("personal"))
        return self._graph

    def get_compiled_graph(self, checkpointer=None):
        return build_personal_graph(checkpointer=checkpointer)

    async def invoke(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})
        async with ensure_invocation_session("personal", thread_id, mode="invoke"):
            logger.info("INVOKE_START agent=personal thread=%s mode=invoke task=%s", thread_id, (task[:200] + "...") if len(task) > 200 else task)

            from agents.shared.memory import save_message_idempotent
            await save_message_idempotent(thread_id, "user", task)

            previous_state = await load_previous_state(self.graph, thread_id, "personal")

            if previous_state is None:
                logger.info("STATE_INIT thread_id=%s source=new_session", thread_id)
                initial_state = PersonalState(
                    messages=[HumanMessage(content=task)],
                    thread_id=thread_id,
                )
                result = await self.graph.ainvoke(initial_state, config=cfg)
            else:
                logger.info("STATE_MERGE thread_id=%s source=checkpoint", thread_id)
                new_state_values = {
                    "messages": [HumanMessage(content=task)],
                    "thread_id": thread_id,
                }
                merged_state = merge_with_new_messages(previous_state, new_state_values)
                result = await self.graph.ainvoke(merged_state, config=cfg)

            messages = list(result.get("messages", []))

            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    response = msg.content if isinstance(msg.content, str) else str(msg.content)
                    logger.info("INVOKE_END agent=personal thread=%s mode=invoke response_length=%s", thread_id, len(response))
                    return response

            logger.info("INVOKE_END agent=personal thread=%s mode=invoke response_length=0", thread_id)
            return ""

    async def stream(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})
        logger.info("INVOKE_START agent=personal thread=%s mode=stream task=%s", thread_id, (task[:200] + "...") if len(task) > 200 else task)

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        previous_state = await load_previous_state(self.graph, thread_id, "personal")

        if previous_state is None:
            logger.info("STATE_INIT thread_id=%s source=new_session", thread_id)
            run_input = PersonalState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
            )
        else:
            logger.info("STATE_MERGE thread_id=%s source=checkpoint", thread_id)
            new_state_values = {
                "messages": [HumanMessage(content=task)],
                "thread_id": thread_id,
            }
            run_input = merge_with_new_messages(previous_state, new_state_values)

        async for message, metadata in self.graph.astream(
            run_input,
            config=cfg,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") == 'call_llm':
                yield message.content if isinstance(message, AIMessage) else str(message.content)
