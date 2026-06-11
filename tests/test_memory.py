"""Tests for memory modules — WorkingMemory and ShortTermMemory."""

from __future__ import annotations

import pytest

from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory


class TestWorkingMemory:
    """Tests for in-process WorkingMemory."""

    def test_update_and_get_context(self, working_memory):
        working_memory.update("s1", {"intent": "knowledge_rag"})
        ctx = working_memory.get_context("s1")
        assert ctx["intent"] == "knowledge_rag"

    def test_update_merges_data(self, working_memory):
        working_memory.update("s1", {"a": 1})
        working_memory.update("s1", {"b": 2})
        ctx = working_memory.get_context("s1")
        assert ctx == {"a": 1, "b": 2}

    def test_update_overwrites_same_key(self, working_memory):
        working_memory.update("s1", {"intent": "old"})
        working_memory.update("s1", {"intent": "new"})
        assert working_memory.get_context("s1")["intent"] == "new"

    def test_sessions_are_isolated(self, working_memory):
        working_memory.update("s1", {"x": 1})
        working_memory.update("s2", {"x": 2})
        assert working_memory.get_context("s1")["x"] == 1
        assert working_memory.get_context("s2")["x"] == 2

    def test_get_history(self, working_memory):
        for i in range(5):
            working_memory.update("s1", {"step": i})
        history = working_memory.get_history("s1", last_n=3)
        assert len(history) == 3
        assert history[-1]["data"]["step"] == 4

    def test_clear(self, working_memory):
        working_memory.update("s1", {"x": 1})
        working_memory.clear("s1")
        assert working_memory.get_context("s1") == {}
        assert working_memory.get_history("s1") == []

    def test_max_entries_enforced(self):
        mem = WorkingMemory(max_entries_per_session=3)
        for i in range(10):
            mem.update("s1", {"step": i})
        history = mem.get_history("s1", last_n=100)
        assert len(history) == 3

    def test_export_for_persistence(self, working_memory):
        working_memory.update("s1", {"intent": "test"})
        export = working_memory.export_for_persistence("s1")
        assert export["session_id"] == "s1"
        assert export["context"]["intent"] == "test"
        assert "exported_at" in export


class TestShortTermMemory:
    """Tests for ShortTermMemory with fallback (no Redis)."""

    @pytest.mark.asyncio
    async def test_add_and_get_history(self, short_term_memory):
        await short_term_memory.add_message("s1", "user", "你好")
        await short_term_memory.add_message("s1", "assistant", "您好！有什么可以帮您？")
        history = await short_term_memory.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "您好！有什么可以帮您？"

    @pytest.mark.asyncio
    async def test_get_history_last_n(self, short_term_memory):
        for i in range(10):
            await short_term_memory.add_message("s1", "user", f"msg{i}")
        history = await short_term_memory.get_history("s1", last_n=3)
        assert len(history) == 3
        assert history[-1]["content"] == "msg9"

    @pytest.mark.asyncio
    async def test_max_turns_enforced(self):
        mem = ShortTermMemory(max_turns=5)
        mem._redis = None
        for i in range(20):
            await mem.add_message("s1", "user", f"msg{i}")
        history = await mem.get_history("s1")
        assert len(history) == 5
        assert history[0]["content"] == "msg15"

    @pytest.mark.asyncio
    async def test_clear(self, short_term_memory):
        await short_term_memory.add_message("s1", "user", "hello")
        await short_term_memory.clear("s1")
        history = await short_term_memory.get_history("s1")
        assert history == []

    @pytest.mark.asyncio
    async def test_sessions_isolated(self, short_term_memory):
        await short_term_memory.add_message("s1", "user", "a")
        await short_term_memory.add_message("s2", "user", "b")
        h1 = await short_term_memory.get_history("s1")
        h2 = await short_term_memory.get_history("s2")
        assert len(h1) == 1
        assert h1[0]["content"] == "a"
        assert h2[0]["content"] == "b"

    @pytest.mark.asyncio
    async def test_get_context_window(self, short_term_memory):
        await short_term_memory.add_message("s1", "user", "问题1")
        await short_term_memory.add_message("s1", "assistant", "回答1")
        window = await short_term_memory.get_context_window("s1")
        assert "user: 问题1" in window
        assert "assistant: 回答1" in window
