from __future__ import annotations

import pytest

from agents.personal.graph import PersonalAgent


@pytest.mark.asyncio
async def test_personal_invoke_returns_response(thread_id: str):
    agent = PersonalAgent()
    result = await agent.invoke(
        task="Say hello in one sentence.",
        thread_id=thread_id,
    )
    assert isinstance(result, str)
    assert result.strip(), "invoke() returned an empty response"
