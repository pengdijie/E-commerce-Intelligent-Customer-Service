"""Tests for TicketHandlerAgent — ticket creation, querying, and process node."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from agents.ticket_handler import TicketHandlerAgent, TicketStore


@pytest.fixture
def ticket_store():
    return TicketStore()


@pytest.fixture
def ticket_agent(mock_llm, ticket_store):
    return TicketHandlerAgent(mock_llm, ticket_store)


class TestTicketStore:
    """Tests for the in-memory TicketStore."""

    def test_create_ticket(self, ticket_store):
        ticket = ticket_store.create(
            ticket_type="refund",
            priority="high",
            summary="退款请求",
            details="用户要求退款100元",
            user_id="user_001",
        )
        assert ticket["ticket_id"].startswith("TK-")
        assert ticket["type"] == "refund"
        assert ticket["priority"] == "high"
        assert ticket["status"] == "created"
        assert ticket["user_id"] == "user_001"

    def test_query_ticket(self, ticket_store):
        ticket = ticket_store.create("refund", "medium", "test", "test detail", "u1")
        found = ticket_store.query(ticket["ticket_id"])
        assert found is not None
        assert found["ticket_id"] == ticket["ticket_id"]

    def test_query_nonexistent_returns_none(self, ticket_store):
        assert ticket_store.query("TK-FAKE-000000") is None

    def test_query_by_user(self, ticket_store):
        ticket_store.create("refund", "medium", "t1", "d1", "user_a")
        ticket_store.create("claim", "high", "t2", "d2", "user_a")
        ticket_store.create("general", "low", "t3", "d3", "user_b")
        results = ticket_store.query_by_user("user_a")
        assert len(results) == 2

    def test_update_status(self, ticket_store):
        ticket = ticket_store.create("refund", "medium", "test", "d", "u1")
        updated = ticket_store.update_status(ticket["ticket_id"], "processing")
        assert updated["status"] == "processing"

    def test_update_nonexistent_returns_none(self, ticket_store):
        assert ticket_store.update_status("TK-FAKE-000", "closed") is None


class TestTicketHandlerAgent:
    """Tests for TicketHandlerAgent methods."""

    @pytest.mark.asyncio
    async def test_create_ticket_returns_formatted_string(self, ticket_agent):
        result = await ticket_agent.create_ticket(
            {"ticket_type": "refund", "priority": "high", "summary": "退款", "details": "详情"},
            user_id="u1",
        )
        assert "工单已创建成功" in result
        assert "TK-" in result
        assert "退款" in result

    @pytest.mark.asyncio
    async def test_query_ticket_found(self, ticket_agent, ticket_store):
        ticket = ticket_store.create("refund", "medium", "test", "d", "u1")
        result = await ticket_agent.query_ticket(ticket["ticket_id"])
        assert "工单查询结果" in result
        assert ticket["ticket_id"] in result

    @pytest.mark.asyncio
    async def test_query_ticket_not_found(self, ticket_agent):
        result = await ticket_agent.query_ticket("TK-FAKE-999999")
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_process_node_creates_ticket(self, ticket_agent, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content='{"action": "create", "ticket_type": "refund", "priority": "high", "summary": "退款100元", "details": "买错了"}'
        ))
        state = {
            "messages": [HumanMessage(content="我要退款")],
            "user_id": "user_001",
            "sub_results": {},
        }
        result = await ticket_agent.process(state)
        assert "ticket_handler" in result["sub_results"]
        assert "工单已创建成功" in result["sub_results"]["ticket_handler"]

    @pytest.mark.asyncio
    async def test_process_node_queries_ticket(self, ticket_agent, mock_llm, ticket_store):
        ticket = ticket_store.create("refund", "medium", "test", "d", "u1")
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=f'{{"action": "query", "ticket_id": "{ticket["ticket_id"]}"}}'
        ))
        state = {
            "messages": [HumanMessage(content=f"查询工单 {ticket['ticket_id']}")],
            "user_id": "user_001",
            "sub_results": {},
        }
        result = await ticket_agent.process(state)
        assert "工单查询结果" in result["sub_results"]["ticket_handler"]
