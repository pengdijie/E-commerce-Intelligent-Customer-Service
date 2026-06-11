"""Tests for the Supervisor routing logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from agents.supervisor import SupervisorNode, route_to_agent, AgentState
from memory.working_memory import WorkingMemory


@pytest.fixture
def supervisor(mock_llm):
    return SupervisorNode(mock_llm, WorkingMemory())


class TestRouteToAgent:
    """Tests for the static route_to_agent function."""

    def test_knowledge_rag_intent(self):
        state: AgentState = {"intent": "knowledge_rag", "messages": [], "user_id": "", "session_id": "",
                             "sub_results": {}, "compliance_passed": True, "final_response": "",
                             "current_agent": "", "retry_count": 0}
        assert route_to_agent(state) == "knowledge_rag"

    def test_ticket_handler_intent(self):
        state: AgentState = {"intent": "ticket_handler", "messages": [], "user_id": "", "session_id": "",
                             "sub_results": {}, "compliance_passed": True, "final_response": "",
                             "current_agent": "", "retry_count": 0}
        assert route_to_agent(state) == "ticket_handler"

    def test_compliance_checker_intent(self):
        state: AgentState = {"intent": "compliance_checker", "messages": [], "user_id": "", "session_id": "",
                             "sub_results": {}, "compliance_passed": True, "final_response": "",
                             "current_agent": "", "retry_count": 0}
        assert route_to_agent(state) == "compliance_check"

    def test_unknown_intent_defaults_to_knowledge_rag(self):
        state: AgentState = {"intent": "some_random_thing", "messages": [], "user_id": "", "session_id": "",
                             "sub_results": {}, "compliance_passed": True, "final_response": "",
                             "current_agent": "", "retry_count": 0}
        assert route_to_agent(state) == "knowledge_rag"

    def test_empty_intent_defaults_to_knowledge_rag(self):
        state: AgentState = {"intent": "", "messages": [], "user_id": "", "session_id": "",
                             "sub_results": {}, "compliance_passed": True, "final_response": "",
                             "current_agent": "", "retry_count": 0}
        assert route_to_agent(state) == "knowledge_rag"


class TestSupervisorRouteDecision:
    """Tests for the Supervisor LLM routing decision."""

    @pytest.mark.asyncio
    async def test_routes_to_knowledge_rag(self, supervisor, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="knowledge_rag"))
        state = {
            "messages": [HumanMessage(content="退货政策是什么？")],
            "session_id": "test-session",
            "intent": "",
            "sub_results": {},
            "compliance_passed": True,
            "final_response": "",
            "current_agent": "",
            "retry_count": 0,
            "user_id": "u1",
        }
        result = await supervisor.route_decision(state)
        assert result["intent"] == "knowledge_rag"

    @pytest.mark.asyncio
    async def test_routes_to_ticket_handler(self, supervisor, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="ticket_handler"))
        state = {
            "messages": [HumanMessage(content="我要退款")],
            "session_id": "test-session",
            "intent": "",
            "sub_results": {},
            "compliance_passed": True,
            "final_response": "",
            "current_agent": "",
            "retry_count": 0,
            "user_id": "u1",
        }
        result = await supervisor.route_decision(state)
        assert result["intent"] == "ticket_handler"

    @pytest.mark.asyncio
    async def test_invalid_llm_response_defaults_to_knowledge_rag(self, supervisor, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="I think we should use knowledge_rag agent here"))
        state = {
            "messages": [HumanMessage(content="你好")],
            "session_id": "test-session",
            "intent": "",
            "sub_results": {},
            "compliance_passed": True,
            "final_response": "",
            "current_agent": "",
            "retry_count": 0,
            "user_id": "u1",
        }
        result = await supervisor.route_decision(state)
        assert result["intent"] == "knowledge_rag"


class TestSupervisorSynthesize:
    """Tests for result synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_with_results(self, supervisor):
        state = {
            "messages": [],
            "sub_results": {"knowledge_rag": "退货政策是7天无理由退货。"},
            "compliance_passed": True,
            "intent": "knowledge_rag",
            "final_response": "",
            "current_agent": "",
            "retry_count": 0,
            "user_id": "u1",
            "session_id": "s1",
        }
        result = await supervisor.synthesize_response(state)
        assert "退货政策" in result["final_response"]

    @pytest.mark.asyncio
    async def test_synthesize_compliance_failed(self, supervisor):
        state = {
            "messages": [],
            "sub_results": {"knowledge_rag": "一些内容"},
            "compliance_passed": False,
            "intent": "knowledge_rag",
            "final_response": "",
            "current_agent": "",
            "retry_count": 0,
            "user_id": "u1",
            "session_id": "s1",
        }
        result = await supervisor.synthesize_response(state)
        assert "敏感内容" in result["final_response"]

    @pytest.mark.asyncio
    async def test_synthesize_empty_results(self, supervisor):
        state = {
            "messages": [],
            "sub_results": {},
            "compliance_passed": True,
            "intent": "knowledge_rag",
            "final_response": "",
            "current_agent": "",
            "retry_count": 0,
            "user_id": "u1",
            "session_id": "s1",
        }
        result = await supervisor.synthesize_response(state)
        assert "暂时无法处理" in result["final_response"]
