"""Tests for ComplianceCheckerAgent — rule-based and LLM checks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.compliance_checker import ComplianceCheckerAgent, ComplianceResult


@pytest.fixture
def compliance_agent(mock_llm):
    return ComplianceCheckerAgent(mock_llm)


class TestRuleBasedCheck:
    """Tests for the regex/rule-based compliance checks (no LLM)."""

    @pytest.fixture(autouse=True)
    def setup(self, compliance_agent):
        self.agent = compliance_agent

    @pytest.mark.asyncio
    async def test_clean_content_passes(self):
        result = await self.agent.rule_check("您的订单已发货，预计3天内到达。")
        assert result.passed is True
        assert result.risk_level == "low"
        assert result.violations == []

    @pytest.mark.asyncio
    async def test_forbidden_financial_term_detected(self):
        result = await self.agent.rule_check("我们保证收益率不低于10%。")
        assert result.passed is False
        assert any("保证收益" in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_multiple_forbidden_terms(self):
        result = await self.agent.rule_check("这款产品零风险，稳赚不赔！")
        assert result.passed is False
        assert len(result.violations) >= 2

    @pytest.mark.asyncio
    async def test_phone_number_pii_detected(self):
        result = await self.agent.rule_check("您的手机号是13812345678")
        assert result.passed is False
        assert any("PII" in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_id_card_pii_detected(self):
        result = await self.agent.rule_check("身份证号：110101199001011234")
        assert result.passed is False
        assert any("身份证号" in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_bank_card_pii_detected(self):
        result = await self.agent.rule_check("卡号6222021234567890123")
        assert result.passed is False
        assert any("银行卡号" in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_email_pii_detected(self):
        result = await self.agent.rule_check("请发送到 user@example.com")
        assert result.passed is False
        assert any("邮箱" in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_pii_plus_forbidden_is_critical(self):
        result = await self.agent.rule_check("保证收益，请拨打13812345678")
        assert result.passed is False
        assert result.risk_level == "critical"


class TestPIIMasking:
    """Tests for _mask_pii."""

    @pytest.fixture(autouse=True)
    def setup(self, compliance_agent):
        self.agent = compliance_agent

    def test_phone_masked(self):
        masked = self.agent._mask_pii("手机号13812345678请保存")
        assert "13812345678" not in masked
        assert "138" in masked  # keeps first 3
        assert "678" in masked  # keeps last 3

    def test_no_pii_unchanged(self):
        text = "您好，请问有什么可以帮您？"
        assert self.agent._mask_pii(text) == text


class TestLLMCheck:
    """Tests for the LLM-based compliance check."""

    @pytest.mark.asyncio
    async def test_llm_check_passes_when_json_valid(self, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content='{"passed": true, "risk_level": "low", "violations": [], "suggestions": []}'
        ))
        agent = ComplianceCheckerAgent(mock_llm)
        result = await agent.llm_check("正常回复内容")
        assert result.passed is True
        assert result.risk_level == "low"

    @pytest.mark.asyncio
    async def test_llm_check_fails_when_violations_found(self, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content='{"passed": false, "risk_level": "high", "violations": ["越权承诺退款"], "suggestions": ["删除承诺"]}'
        ))
        agent = ComplianceCheckerAgent(mock_llm)
        result = await agent.llm_check("我保证给你退款100万")
        assert result.passed is False
        assert "越权承诺退款" in result.violations

    @pytest.mark.asyncio
    async def test_llm_check_graceful_on_invalid_json(self, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="not json at all"))
        agent = ComplianceCheckerAgent(mock_llm)
        result = await agent.llm_check("测试内容")
        assert result.passed is True  # defaults to pass on parse failure


class TestProcessNode:
    """Tests for the graph-node process() interface."""

    @pytest.mark.asyncio
    async def test_empty_sub_results_passes(self, mock_llm):
        agent = ComplianceCheckerAgent(mock_llm)
        state = {"sub_results": {}, "messages": []}
        result = await agent.process(state)
        assert result["compliance_passed"] is True

    @pytest.mark.asyncio
    async def test_process_checks_sub_results(self, mock_llm):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content='{"passed": true, "risk_level": "low", "violations": [], "suggestions": []}'
        ))
        agent = ComplianceCheckerAgent(mock_llm)
        state = {
            "sub_results": {"knowledge_rag": "退货政策是7天内无理由退货。"},
            "messages": [],
        }
        result = await agent.process(state)
        assert result["compliance_passed"] is True
