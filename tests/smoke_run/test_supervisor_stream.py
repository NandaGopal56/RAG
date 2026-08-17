from __future__ import annotations

import pytest

from agents.supervisor.grpah import Supervisor
from agents.personal.graph import PersonalAgent
from agents.deep_research.graph import DeepResearchAgent


@pytest.mark.asyncio
async def test_supervisor_stream_yields_updates(thread_id: str):
    agents = {
        "personal": PersonalAgent(),
        "deep_research": DeepResearchAgent(),
    }
    supervisor = Supervisor(agents=agents)
    updates = []
    async for update in supervisor.stream(
        task="Say hello in one sentence.",
        thread_id=thread_id,
    ):
        updates.append(update)
        if len(updates) >= 5:
            break

    assert len(updates) > 0, "stream() yielded no updates"
    assert all(isinstance(u, dict) for u in updates), "stream() yielded non-dict updates"
