"""Shared fixtures for tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory


@pytest.fixture
def working_memory():
    return WorkingMemory()


@pytest.fixture
def short_term_memory():
    mem = ShortTermMemory(redis_url="redis://localhost:6379/0")
    mem._redis = None  # force fallback store
    return mem


@pytest.fixture
def mock_llm():
    """A mock ChatOpenAI that returns controllable responses."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm
