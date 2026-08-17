from __future__ import annotations

import pytest

from agents.deep_research.graph import DeepResearchAgent


@pytest.mark.asyncio
async def test_deep_research_stream_yields_updates(thread_id: str):
    agent = DeepResearchAgent()
    updates = []
    async for update in agent.stream(
        task="What are the pros and cons of electric vehicles?",
        thread_id=thread_id,
    ):
        updates.append(update)
        if len(updates) >= 5:
            break

    assert len(updates) > 0, "stream() yielded no updates"
    assert all(isinstance(u, dict) for u in updates), "stream() yielded non-dict updates"
