from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import (
    get_checkpointer,
    load_previous_state,
    merge_with_new_messages,
)

from .nodes import (
    check_goal_confirmation,
    check_plan_confirmation,
    clarify_goal,
    create_plan,
    entry_router,
    execute_step,
    execution_router,
    finish,
    goal_confirmation_router,
    plan_confirmation_router,
    reflect,
)
from agents.shared.logging import get_agent_logger, ensure_invocation_session


from .state import ResearchState

logger = get_agent_logger("deep_research", "graph")


def build_research_graph(checkpointer=None):
    g = StateGraph(ResearchState)

    # Nodes
    g.add_node("clarify_goal", clarify_goal)
    g.add_node("check_goal_confirmation", check_goal_confirmation)
    g.add_node("create_plan", create_plan)
    g.add_node("check_plan_confirmation", check_plan_confirmation)
    g.add_node("execute_step", execute_step)
    g.add_node("reflect", reflect)
    g.add_node("finish", finish)

    # Entry routing
    g.add_conditional_edges(
        START,
        entry_router,
        path_map={
            "clarify_goal": "clarify_goal",
            "create_plan": "create_plan",
            "execute_step": "execute_step",
        },
    )

    # Goal loop
    g.add_edge("clarify_goal", "check_goal_confirmation")
    g.add_conditional_edges(
        "check_goal_confirmation",
        goal_confirmation_router,
        path_map={
            "clarify_goal": "clarify_goal",
            "create_plan": "create_plan",
        },
    )

    # Plan loop
    g.add_edge("create_plan", "check_plan_confirmation")
    g.add_conditional_edges(
        "check_plan_confirmation",
        plan_confirmation_router,
        path_map={
            "create_plan": "create_plan",
            "execute_step": "execute_step",
        },
    )

    # Execution loop
    g.add_edge("execute_step", "reflect")
    g.add_conditional_edges(
        "reflect",
        execution_router,
        path_map={
            "execute_step": "execute_step",
            "finish": "finish",
        },
    )

    g.add_edge("finish", END)

    logger.info("GRAPH_COMPILED component=deep_research")
    return g.compile(checkpointer=checkpointer)


class DeepResearchAgent(BaseAgent):
    def __init__(self):
        self._graph = None

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="deep_research",
            name="Deep Research Agent",
            description=(
                "Handles complex questions needing multi-step research, "
                "synthesis across many sources, or thorough investigation. "
                "Clarifies the goal, confirms the goal, creates a plan, "
                "confirms the plan, then executes the research."
            ),
            can_handle_long_tasks=True,
        )

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_research_graph(get_checkpointer("deep_research"))
        return self._graph

    def get_compiled_graph(self, checkpointer=None):
        return build_research_graph(checkpointer=checkpointer)

    async def _is_interrupted(self, cfg: RunnableConfig) -> bool:
        try:
            snapshot = await self.graph.aget_state(cfg)
            return bool(snapshot.next)
        except Exception:
            return False

    async def _run(
        self,
        task: str,
        thread_id: str,
        config: Optional[RunnableConfig] = None,
    ):
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        from agents.shared.memory import save_message_idempotent

        await save_message_idempotent(thread_id, "user", task)

        if await self._is_interrupted(cfg):
            logger.info("INVOKE_START agent=deep_research thread=%s mode=invoke task=%s resumed=True", thread_id, (task[:200] + "...") if len(task) > 200 else task)
            return Command(resume=task), cfg

        logger.info("INVOKE_START agent=deep_research thread=%s mode=invoke task=%s", thread_id, (task[:200] + "...") if len(task) > 200 else task)

        previous_state = await load_previous_state(self.graph, thread_id, "deep_research")

        if previous_state is None:
            logger.info("STATE_INIT thread_id=%s source=new_session", thread_id)
            state = ResearchState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                original_question=task,
                goal="",
                clarifying_questions=[],
                user_clarification="",
                goal_ready=False,
                goal_confirmed=False,
                goal_revision_notes="",
                plan=[],
                done_when="",
                plan_confirmed=False,
                plan_revision_notes="",
                current_step=0,
                findings="",
                next_focus="",
                is_done=False,
                iteration=0,
                final_answer="",
            )
        else:
            logger.info("STATE_MERGE thread_id=%s source=checkpoint", thread_id)
            state = merge_with_new_messages(
                previous_state,
                {
                    "messages": [HumanMessage(content=task)],
                    "thread_id": thread_id,
                },
            )

        return state, cfg

    async def invoke(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        async with ensure_invocation_session("deep_research", thread_id, mode="invoke"):
            run_input, cfg = await self._run(task, thread_id, config)
            result = await self.graph.ainvoke(run_input, config=cfg)

            pending = result.get("__interrupt__")
            if pending:
                payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
                logger.info("INTERRUPT_RAISED thread_id=%s payload_type=%s", thread_id, type(payload).__name__)
                if isinstance(payload, dict):
                    response = payload.get("message", str(payload))
                else:
                    response = str(payload)
                logger.info("INVOKE_END agent=deep_research thread=%s mode=invoke interrupted=True response_length=%s", thread_id, len(response))
                return response

            for msg in reversed(list(result.get("messages", []))):
                if isinstance(msg, AIMessage) and msg.content:
                    response = msg.content if isinstance(msg.content, str) else str(msg.content)
                    logger.info("INVOKE_END agent=deep_research thread=%s mode=invoke response_length=%s", thread_id, len(response))
                    return response

            final = result.get("final_answer", "")
            logger.info("INVOKE_END agent=deep_research thread=%s mode=invoke response_length=%s", thread_id, len(final))
            return final

    async def stream(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        async with ensure_invocation_session("deep_research", thread_id, mode="stream"):
            run_input, cfg = await self._run(task, thread_id, config)
            async for update in self.graph.astream(run_input, config=cfg, stream_mode="updates"):
                for node_name, node_update in update.items():
                    logger.debug(
                        "STREAM_UPDATE agent=deep_research thread=%s node=%s update_keys=%s",
                        thread_id, node_name, list(node_update.keys()) if isinstance(node_update, dict) else None,
                    )
                    yield {"node": node_name, "update": node_update}
            logger.info("INVOKE_END agent=deep_research thread=%s mode=stream", thread_id)
