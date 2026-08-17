from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

from langgraph.types import RunnableConfig

from agents.base import BaseAgent
from agents.deep_research.graph import DeepResearchAgent
from agents.personal.graph import PersonalAgent
from agents.supervisor.grpah import Supervisor
from agents.shared.logging import (
    get_agent_logger,
    log_error,
    log_event,
    log_invoke_end,
    log_invoke_start,
    log_node_state,
    log_stream_update,
)


logger = get_agent_logger("client", "gateway")


class AgentGateway:
    """
    Thin client-side gateway for selecting and invoking agents.
    Keeps one shared entrypoint for direct agent use and supervisor use.
    """

    def __init__(self) -> None:
        self._agents: Optional[Dict[str, BaseAgent]] = None
        self._supervisor: Optional[Supervisor] = None

    async def get_individual_agents(self) -> Dict[str, BaseAgent]:
        if self._agents is None:
            log_event(logger, "AGENTS_INIT", component="gateway")
            self._agents = {
                "personal": PersonalAgent(),
                "deep_research": DeepResearchAgent(),
            }
            log_event(logger, "AGENTS_READY", agents=list(self._agents.keys()))
        return self._agents

    async def get_supervisor(self) -> Supervisor:
        if self._supervisor is None:
            log_event(logger, "SUPERVISOR_INIT", component="gateway")
            self._supervisor = Supervisor(agents=await self.get_individual_agents())
            log_event(logger, "SUPERVISOR_READY", supervisor_type=type(self._supervisor).__name__)
        return self._supervisor

    async def get_agent(self, agent_name: str) -> BaseAgent:
        log_event(logger, "AGENT_SELECT", agent=agent_name, level=10)
        if agent_name == "supervisor":
            return await self.get_supervisor()

        agents = await self.get_individual_agents()
        if agent_name not in agents:
            log_error(logger, "AGENT_UNKNOWN", agent=agent_name)
            raise KeyError(f"Unknown agent '{agent_name}'")
        return agents[agent_name]

    async def invoke(
        self,
        agent_name: str,
        task: str,
        thread_id: str = "1",
        context: Optional[Dict[str, object]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        from agents.shared.logging import ensure_invocation_session

        async with ensure_invocation_session(agent_name, thread_id, mode="invoke"):
            agent_logger = get_agent_logger(agent_name)
            log_invoke_start(agent_logger, agent_name, thread_id=thread_id, mode="invoke", task_preview=task)
            if context:
                log_node_state(agent_logger, "invoke", context, label="context")
            try:
                result = await self.get_agent(agent_name).invoke(
                    task=task,
                    thread_id=thread_id,
                    context=context,
                    config=config,
                )
                log_invoke_end(
                    agent_logger,
                    agent_name,
                    thread_id=thread_id,
                    mode="invoke",
                    response_length=len(result) if result else 0,
                )
                return result
            except Exception as e:
                log_error(agent_logger, "INVOKE_ERROR", agent=agent_name, thread=thread_id, error=str(e))
                raise

    async def stream(
        self,
        agent_name: str,
        task: str,
        thread_id: str = "1",
        context: Optional[Dict[str, object]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, object]]:
        from agents.shared.logging import ensure_invocation_session

        async with ensure_invocation_session(agent_name, thread_id, mode="stream"):
            agent_logger = get_agent_logger(agent_name)
            log_invoke_start(agent_logger, agent_name, thread_id=thread_id, mode="stream", task_preview=task)
            if context:
                log_node_state(agent_logger, "stream", context, label="context")

            agent = await self.get_agent(agent_name)

            async for update in agent.stream(
                task=task,
                thread_id=thread_id,
                context=context,
                config=config,
            ):
                yield update

    def save_graphs(self) -> None:
        artifacts = Path(__file__).parent / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        personal = self.get_individual_agents()["personal"]
        research = self.get_individual_agents()["deep_research"]
        supervisor = self.get_supervisor()

        personal.graph.get_graph().draw_mermaid_png(
            output_file_path=str(artifacts / "personal_agent.png")
        )
        research.graph.get_graph().draw_mermaid_png(
            output_file_path=str(artifacts / "deep_research_agent.png")
        )
        supervisor.graph.get_graph().draw_mermaid_png(
            output_file_path=str(artifacts / "supervisor_shell.png")
        )
        supervisor.graph.get_graph(xray=True).draw_mermaid_png(
            output_file_path=str(artifacts / "supervisor_full_system.png")
        )

        log_event(logger, "GRAPHS_SAVED", path=str(artifacts))

    async def registered_agents(self) -> Dict[str, str]:
        agents = await self.get_individual_agents()
        supervisor = await self.get_supervisor()
        return {
            **{aid: agent.info.name for aid, agent in agents.items()},
            "supervisor": supervisor.info.name,
        }


gateway = AgentGateway()
