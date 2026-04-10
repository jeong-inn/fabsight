"""
Test suite for agents module
Tests for DetectionAgent, DiagnosisAgent, ActionAgent, and ReportAgent
"""
import pytest
import numpy as np
import pandas as pd
from src.agents.pipeline import DetectionAgent, DiagnosisAgent, ActionAgent


class TestDetectionAgent:
    """Test DetectionAgent functionality"""

    @pytest.fixture
    def agent(self):
        return DetectionAgent()

    def test_detection_agent_initialization(self, agent):
        """Test DetectionAgent can be initialized"""
        assert agent is not None

    def test_detection_agent_run_basic(self, agent):
        """Test DetectionAgent.run() with basic input"""
        anomaly_scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9])
        risk_scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9])

        result = agent.run(anomaly_scores, risk_scores, threshold=0.5)

        assert "anomaly_count" in result
        assert "total_count" in result
        assert "anomaly_rate" in result
        assert "high_risk_count" in result
        assert result["total_count"] == 5

    def test_detection_agent_anomaly_count(self, agent):
        """Test anomaly count calculation"""
        anomaly_scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9])
        risk_scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9])

        result = agent.run(anomaly_scores, risk_scores, threshold=0.5)

        # With threshold 0.5, anomalies are where anomaly_scores < 0.5
        # That's indices 0, 1, 2 (3 anomalies)
        assert result["anomaly_count"] == 3

    def test_detection_agent_high_risk_count(self, agent):
        """Test high risk count calculation (>= 0.7)"""
        anomaly_scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9])
        risk_scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9])

        result = agent.run(anomaly_scores, risk_scores)

        # High risk count is where risk_scores >= 0.7
        # That's indices 3, 4 (2 high risk)
        assert result["high_risk_count"] == 2


class TestDiagnosisAgent:
    """Test DiagnosisAgent functionality"""

    @pytest.fixture
    def agent(self):
        return DiagnosisAgent()

    @pytest.fixture
    def sample_top5_df(self):
        """Create sample top 5 sensors dataframe"""
        return pd.DataFrame({
            'sensor': [31, 487, 545, 59, 419],
            'shap_score': [0.025, 0.020, 0.015, 0.012, 0.010]
        })

    def test_diagnosis_agent_initialization(self, agent):
        """Test DiagnosisAgent can be initialized"""
        assert agent is not None

    def test_diagnosis_agent_run(self, agent, sample_top5_df):
        """Test DiagnosisAgent.run() with sample data"""
        result = agent.run(sample_top5_df)

        assert "root_causes" in result
        assert "primary_process" in result
        assert "affected_stages" in result
        assert len(result["root_causes"]) == 5

    def test_diagnosis_agent_primary_process(self, agent, sample_top5_df):
        """Test primary process is identified correctly"""
        result = agent.run(sample_top5_df)

        # First sensor (31) is CVD
        assert result["primary_process"] == "CVD"

    def test_diagnosis_agent_affected_stages(self, agent, sample_top5_df):
        """Test affected stages are identified"""
        result = agent.run(sample_top5_df)

        assert "affected_stages" in result
        assert len(result["affected_stages"]) > 0


class TestActionAgent:
    """Test ActionAgent functionality"""

    @pytest.fixture
    def agent(self):
        return ActionAgent()

    def test_action_agent_initialization(self, agent):
        """Test ActionAgent can be initialized"""
        assert agent is not None

    def test_action_agent_run(self, agent):
        """Test ActionAgent.run() with sample diagnosis"""
        diagnosis = {
            "primary_process": "CVD",
            "root_causes": [{"shap_score": 0.025}]
        }

        result = agent.run(diagnosis)

        assert "primary_process" in result
        assert "recommended_actions" in result
        assert "priority" in result
        assert result["primary_process"] == "CVD"

    def test_action_agent_priority_critical(self, agent):
        """Test priority is critical when shap_score > 0.02"""
        diagnosis = {
            "primary_process": "CVD",
            "root_causes": [{"shap_score": 0.025}]
        }

        result = agent.run(diagnosis)
        assert result["priority"] == "즉시 조치"

    def test_action_agent_priority_monitoring(self, agent):
        """Test priority is monitoring when shap_score <= 0.02"""
        diagnosis = {
            "primary_process": "CVD",
            "root_causes": [{"shap_score": 0.015}]
        }

        result = agent.run(diagnosis)
        assert result["priority"] == "모니터링"

    def test_action_agent_unknown_process(self, agent):
        """Test ActionAgent handles unknown process gracefully"""
        diagnosis = {
            "primary_process": "UNKNOWN_PROCESS",
            "root_causes": [{"shap_score": 0.025}]
        }

        result = agent.run(diagnosis)

        # Should return default actions
        assert "recommended_actions" in result
        assert len(result["recommended_actions"]) > 0
