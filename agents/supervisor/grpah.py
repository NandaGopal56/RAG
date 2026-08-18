from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional
from agents.shared.logging import get_agent_logger, ensure_invocation_session

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
    logger.info("GRAPH_BUILD component=supervisor agents=%s", list(agents.keys()))
    g = StateGraph(SupervisorState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("route_request", make_route_request(agents))
    g.add_node("ask_user",      ask_user)
    logger.info("GRAPH_NODES_ADDED component=supervisor nodes=%s", ["route_request", "ask_user", *list(agents.keys())])

    for agent_id, agent in agents.items():
        try:
            subgraph = agent.get_compiled_graph(checkpointer=True)
            logger.info("SUBGRAPH_COMPILED agent_id=%s source=standalone", agent_id)
        except Exception:
            subgraph = agent.graph
            logger.info("SUBGRAPH_ATTACHED agent_id=%s source=existing_graph", agent_id)
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
    logger.info("GRAPH_COMPILED component=supervisor checkpointer=supervisor")
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
        if not thread_id:
            return False
        try:
            snapshot = await self._graph.aget_state(cfg, subgraphs=True)
            pending = bool(snapshot.next)
            logger.info("INTERRUPT_CHECK thread_id=%s pending=%s next_nodes=%s", thread_id, pending, str(snapshot.next))
            return pending
        except Exception as e:
            logger.warning("INTERRUPT_CHECK_ERROR thread_id=%s error=%s", thread_id, e)
            return False

    async def _run(self, task: str, thread_id: str, config: Optional[RunnableConfig]):
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})
        logger.info("INVOKE_START agent=supervisor thread=%s mode=invoke task=%s", thread_id, (task[:200] + "...") if len(task) > 200 else task)

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        if await self._pending_interrupt(thread_id, cfg):
            logger.info("RESUME_INTERRUPT thread_id=%s reply_preview=%s", thread_id, (task[:200] + "...") if len(task) > 200 else task)
            return Command(resume=task), cfg

        previous_state = await load_previous_state(self.graph, thread_id, "supervisor")
        logger.info("STATE_LOAD thread_id=%s found=%s", thread_id, bool(previous_state))

        if previous_state is None:
            state = SupervisorState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                user_input=task,
            )
            logger.info("STATE_INIT thread_id=%s source=new_session", thread_id)
        else:
            state = merge_with_new_messages(previous_state, {
                "messages":   [HumanMessage(content=task)],
                "thread_id":  thread_id,
                "user_input": task,
            })
            logger.info("STATE_MERGE thread_id=%s source=checkpoint", thread_id)
            logger.debug("STATE supervisor.merged_state %s", state)

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
                logger.info("INTERRUPT_RAISED thread_id=%s payload_type=%s", thread_id, type(payload).__name__)
                response = payload.get("message", str(payload)) if isinstance(payload, dict) else str(payload)
                logger.info("INVOKE_END agent=supervisor thread=%s mode=invoke interrupted=True response_length=%s", thread_id, len(response))
                return response

            response = await self._extract_response(result, thread_id)
            logger.info("INVOKE_END agent=supervisor thread=%s mode=invoke response_length=%s", thread_id, len(response))
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

            logger.info("INVOKE_END agent=supervisor thread=%s mode=stream", thread_id)
