from __future__ import annotations

import uuid

import pytest


@pytest.fixture
def thread_id():
    """Unique thread ID per test to avoid state collisions."""
    return f"test-{uuid.uuid4().hex[:8]}"
