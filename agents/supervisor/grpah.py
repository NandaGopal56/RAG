from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional
from agents.shared.logging import (
    get_agent_logger,
    ensure_invocation_session,
    log_event,
    log_invoke_end,
    log_invoke_start,
    log_node_state,
    log_stream_update,
)

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import get_checkpointer, load_previous_state, merge_with_new_messages

from .nodes import ask_user, make_route_request, what_to_do
from .state import SupervisorState

DEBUG_MODE = True
logger = get_agent_logger("supervisor", "graph")


def build_supervisor_graph(agents: Dict[str, BaseAgent]):
    """Build and compile the supervisor's LangGraph."""
    log_event(logger, "GRAPH_BUILD", component="supervisor", agents=list(agents.keys()))
    g = StateGraph(SupervisorState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("route_request", make_route_request(agents))
    g.add_node("ask_user",      ask_user)
    log_event(logger, "GRAPH_NODES_ADDED", component="supervisor", nodes=["route_request", "ask_user", *list(agents.keys())])

    for agent_id, agent in agents.items():
        try:
            subgraph = agent.get_compiled_graph(checkpointer=True)
            log_event(logger, "SUBGRAPH_COMPILED", agent_id=agent_id, source="standalone")
        except Exception:
            subgraph = agent.graph
            log_event(logger, "SUBGRAPH_ATTACHED", agent_id=agent_id, source="existing_graph")
        g.add_node(agent_id, subgraph)

    # -- Edges ----------------------------------------------------------------
    g.add_edge(START, "route_request")

    g.add_conditional_edges(
        "route_request",
        what_to_do,
        path_map={"ask_user": "ask_user", **{agent_id: agent_id for agent_id in agents}},
    )

    g.add_edge("ask_user", END)
    for agent_id in agents:
        g.add_edge(agent_id, END)

    checkpointer = get_checkpointer("supervisor")
    compiled = g.compile(checkpointer=checkpointer)
    log_event(logger, "GRAPH_COMPILED", component="supervisor", checkpointer="supervisor")
    return compiled


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor(BaseAgent):
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self._graph = build_supervisor_graph(agents)

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="supervisor",
            name="Supervisor",
            description="Routes requests across registered agents.",
            can_handle_long_tasks=True,
        )

    @property
    def graph(self):
        return self._graph

    async def _pending_interrupt(self, thread_id: str, cfg: RunnableConfig) -> bool:
        """
        Check whether this thread is paused at an interrupt ANYWHERE in the
        graph, including inside a subgraph node (e.g. deep_research's
        check_plan_confirmation).

        subgraphs=True is mandatory: without it, aget_state only reads the
        supervisor's own top-level state and never sees an interrupt() raised
        inside a child graph's checkpoint_ns.
        """
        if not thread_id:
            return False
        try:
            snapshot = await self._graph.aget_state(cfg, subgraphs=True)
            pending = bool(snapshot.next)
            log_event(logger, "INTERRUPT_CHECK", thread_id=thread_id, pending=pending, next_nodes=str(snapshot.next))
            return pending
        except Exception as e:
            log_event(logger, "INTERRUPT_CHECK_ERROR", level=30, thread_id=thread_id, error=str(e))
            return False

    async def _run(self, task: str, thread_id: str, config: Optional[RunnableConfig]):
        """
        Shared setup for invoke()/stream().

        If the thread is paused at an interrupt anywhere in the graph
        (including inside a subgraph), resume it with Command(resume=task)
        using the SAME config — no new state dict. This is the correct
        LangGraph resume pattern: LangGraph's own machinery delivers `task`
        to the waiting interrupt() call.

        Only when nothing is pending do we fall through to the normal
        route_request path, which rebuilds state and classifies the intent.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})
        log_invoke_start(logger, "supervisor", thread_id=thread_id, mode="invoke", task_preview=task)

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        if await self._pending_interrupt(thread_id, cfg):
            log_event(logger, "RESUME_INTERRUPT", thread_id=thread_id, reply_preview=(task[:200] + "...") if len(task) > 200 else task)
            return Command(resume=task), cfg

        previous_state = await load_previous_state(self.graph, thread_id, "supervisor")
        log_event(logger, "STATE_LOAD", thread_id=thread_id, found=bool(previous_state))

        if previous_state is None:
            state = SupervisorState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                user_input=task,
            )
            log_event(logger, "STATE_INIT", thread_id=thread_id, source="new_session")
        else:
            state = merge_with_new_messages(previous_state, {
                "messages":   [HumanMessage(content=task)],
                "thread_id":  thread_id,
                "user_input": task,
            })
            log_event(logger, "STATE_MERGE", thread_id=thread_id, source="checkpoint")
            log_node_state(logger, "supervisor", state, label="merged_state")

        return state, cfg

    async def _extract_response(self, result: Dict[str, Any], thread_id: str) -> str:
        from agents.shared.memory import load_thread
        db_history = await load_thread(thread_id)
        for msg in reversed(db_history):
            if msg.get("role") == "assistant":
                return msg.get("content") or ""
        messages = list(result.get("messages", []))
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return result.get("response", "")

    async def invoke(
        self,
        task: str,
        thread_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        async with ensure_invocation_session("supervisor", thread_id, mode="invoke"):
            run_input, cfg = await self._run(task, thread_id, config)
            result = await self._graph.ainvoke(run_input, config=cfg)

            pending = result.get("__interrupt__")
            if pending:
                payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
                log_event(logger, "INTERRUPT_RAISED", thread_id=thread_id, payload_type=type(payload).__name__)
                response = payload.get("message", str(payload)) if isinstance(payload, dict) else str(payload)
                log_invoke_end(logger, "supervisor", thread_id=thread_id, mode="invoke", interrupted=True, response_length=len(response))
                return response

            response = await self._extract_response(result, thread_id)
            log_invoke_end(logger, "supervisor", thread_id=thread_id, mode="invoke", response_length=len(response))
            return response

    async def stream(
        self,
        task: str,
        thread_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        async with ensure_invocation_session("supervisor", thread_id, mode="stream"):
            run_input, cfg = await self._run(task, thread_id, config)

            async for data in self.graph.astream(
                run_input,
                config=cfg,
                stream_mode="messages",
                subgraphs=True
            ):
                namespace: tuple = data[0]
                data = data[1]

                message = data[0]
                metadata = data[1]

                if namespace[0] if len(namespace) > 0 else namespace == 'personal':
                    if metadata.get("langgraph_node") == 'call_llm':
                        yield message.content if isinstance(message, AIMessage) else str(message.content)

            log_invoke_end(logger, "supervisor", thread_id=thread_id, mode="stream")