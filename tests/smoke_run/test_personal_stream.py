from __future__ import annotations

import pytest

from agents.personal.graph import PersonalAgent


@pytest.mark.asyncio
async def test_personal_stream_yields_updates(thread_id: str):
    agent = PersonalAgent()
    updates = []
    async for update in agent.stream(
        task="Say hello in one sentence.",
        thread_id=thread_id,
    ):
        updates.append(update)

    assert len(updates) > 0, "stream() yielded no updates"
    assert all(isinstance(u, dict) for u in updates), "stream() yielded non-dict updates"
