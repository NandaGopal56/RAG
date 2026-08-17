from __future__ import annotations

import pytest

from agents.supervisor.grpah import Supervisor
from agents.personal.graph import PersonalAgent
from agents.deep_research.graph import DeepResearchAgent


@pytest.mark.asyncio
async def test_supervisor_invoke_returns_response(thread_id: str):
    agents = {
        "personal": PersonalAgent(),
        "deep_research": DeepResearchAgent(),
    }
    supervisor = Supervisor(agents=agents)
    result = await supervisor.invoke(
        task="Say hello in one sentence.",
        thread_id=thread_id,
    )
    assert isinstance(result, str)
    assert result.strip(), "invoke() returned an empty response"
