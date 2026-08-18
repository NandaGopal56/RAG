from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

from langgraph.types import RunnableConfig

from agents.base import BaseAgent
from agents.deep_research.graph import DeepResearchAgent
from agents.personal.graph import PersonalAgent
from agents.supervisor.grpah import Supervisor
from agents.shared.logging import get_agent_logger, ensure_invocation_session


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
            logger.info("AGENTS_INIT component=gateway")
            self._agents = {
                "personal": PersonalAgent(),
                "deep_research": DeepResearchAgent(),
            }
            logger.info("AGENTS_READY agents=%s", list(self._agents.keys()))
        return self._agents

    async def get_supervisor(self) -> Supervisor:
        if self._supervisor is None:
            logger.info("SUPERVISOR_INIT component=gateway")
            self._supervisor = Supervisor(agents=await self.get_individual_agents())
            logger.info("SUPERVISOR_READY supervisor_type=%s", type(self._supervisor).__name__)
        return self._supervisor

    async def get_agent(self, agent_name: str) -> BaseAgent:
        logger.log(10, "AGENT_SELECT agent=%s", agent_name)
        if agent_name == "supervisor":
            return await self.get_supervisor()

        agents = await self.get_individual_agents()
        if agent_name not in agents:
            logger.error("AGENT_UNKNOWN agent=%s", agent_name)
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
        async with ensure_invocation_session(agent_name, thread_id, mode="invoke"):
            agent_logger = get_agent_logger(agent_name)
            agent_logger.info("INVOKE_START agent=%s thread=%s mode=invoke task=%s", agent_name, thread_id, (task[:200] + "...") if len(task) > 200 else task)
            if context:
                agent_logger.debug("STATE invoke.context %s", context)
            try:
                agent = await self.get_agent(agent_name)

                result = await agent.invoke(
                    task=task,
                    thread_id=thread_id,
                    context=context,
                    config=config,
                )
                agent_logger.info(
                    "INVOKE_END agent=%s thread=%s mode=invoke response_length=%s",
                    agent_name, thread_id, len(result) if result else 0,
                )
                return result
            except Exception as e:
                agent_logger.error("INVOKE_ERROR agent=%s thread=%s error=%s", agent_name, thread_id, e)
                raise

    async def stream(
        self,
        agent_name: str,
        task: str,
        thread_id: str = "1",
        context: Optional[Dict[str, object]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, object]]:
        async with ensure_invocation_session(agent_name, thread_id, mode="stream"):
            agent_logger = get_agent_logger(agent_name)
            agent_logger.info("INVOKE_START agent=%s thread=%s mode=stream task=%s", agent_name, thread_id, (task[:200] + "...") if len(task) > 200 else task)
            if context:
                agent_logger.debug("STATE stream.context %s", context)

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

        logger.info("GRAPHS_SAVED path=%s", str(artifacts))

    async def registered_agents(self) -> Dict[str, str]:
        agents = await self.get_individual_agents()
        supervisor = await self.get_supervisor()
        return {
            **{aid: agent.info.name for aid, agent in agents.items()},
            "supervisor": supervisor.info.name,
        }


gateway = AgentGateway()
