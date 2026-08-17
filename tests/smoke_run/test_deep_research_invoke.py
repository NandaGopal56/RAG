from __future__ import annotations

import pytest

from agents.deep_research.graph import DeepResearchAgent


@pytest.mark.asyncio
async def test_deep_research_invoke_returns_response(thread_id: str):
    agent = DeepResearchAgent()
    result = await agent.invoke(
        task="What are the pros and cons of electric vehicles?",
        thread_id=thread_id,
    )
    assert isinstance(result, str)
    assert result.strip(), "invoke() returned an empty response"
