"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("api.main.create_supervisor_graph") as mock_graph_factory, \
         patch("api.main.init_tracer"), \
         patch("api.main.WikiknowledgeBase"):

        mock_compiled = MagicMock()
        mock_compiled.ainvoke = AsyncMock(return_value={
            "messages": [],
            "intent": "knowledge_rag",
            "sub_results": {"knowledge_rag": "测试回答"},
            "compliance_passed": True,
            "final_response": "测试回答",
        })
        mock_graph_factory.return_value = mock_compiled

        from api.main import app, lifespan
        import api.main as api_module
        api_module.graph = mock_compiled

        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestChatEndpoint:
    def test_chat_returns_response(self, client):
        resp = client.post("/api/chat", json={
            "message": "退货政策是什么？",
            "user_id": "test_user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "session_id" in data

    def test_chat_with_session_id(self, client):
        resp = client.post("/api/chat", json={
            "message": "你好",
            "user_id": "test_user",
            "session_id": "my-session-123",
        })
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "my-session-123"

    def test_chat_empty_message(self, client):
        resp = client.post("/api/chat", json={
            "message": "",
            "user_id": "test_user",
        })
        # FastAPI might still process it — depends on validation
        # At minimum it shouldn't crash
        assert resp.status_code in (200, 422)


class TestHistoryEndpoint:
    def test_get_history_empty_session(self, client):
        resp = client.get("/api/history/nonexistent-session")
        assert resp.status_code == 200
        data = resp.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)


class TestMetricsEndpoint:
    def test_metrics_returns_data(self, client):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_metrics" in data
        assert "tool_call_log" in data
