"""
Test suite for model modules
Tests for model training, prediction, and evaluation
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.prediction.risk_scorer import PreFailureRiskScorer


class TestPreFailureRiskScorer:
    """Test PreFailureRiskScorer functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, 10),
            columns=[f"feature_{i}" for i in range(10)]
        )
        # Create labels: 1=normal, -1=anomaly
        y = pd.Series([1] * 80 + [-1] * 20)
        return X, y

    @pytest.fixture
    def scorer(self):
        return PreFailureRiskScorer()

    def test_scorer_initialization(self, scorer):
        """Test PreFailureRiskScorer can be initialized"""
        assert scorer is not None
        assert scorer.is_trained is False

    def test_scorer_training(self, scorer, sample_data):
        """Test model training"""
        X, y = sample_data
        metrics = scorer.train(X, y)

        assert scorer.is_trained is True
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_scorer_predict_risk(self, scorer, sample_data):
        """Test risk prediction after training"""
        X, y = sample_data
        scorer.train(X, y)

        risk_scores = scorer.predict_risk(X)

        assert len(risk_scores) == len(X)
        assert np.all((risk_scores >= 0) & (risk_scores <= 1))

    def test_scorer_predict_without_training(self, scorer, sample_data):
        """Test that predict raises error if not trained"""
        X, _ = sample_data

        with pytest.raises(ValueError):
            scorer.predict_risk(X)

    def test_scorer_get_risk_level_high(self, scorer):
        """Test risk level classification for high risk"""
        level = scorer.get_risk_level(0.75)
        assert "HIGH" in level

    def test_scorer_get_risk_level_medium(self, scorer):
        """Test risk level classification for medium risk"""
        level = scorer.get_risk_level(0.55)
        assert "MEDIUM" in level

    def test_scorer_get_risk_level_low(self, scorer):
        """Test risk level classification for low risk"""
        level = scorer.get_risk_level(0.25)
        assert "LOW" in level

    def test_scorer_metrics_values(self, scorer, sample_data):
        """Test that metrics are reasonable values"""
        X, y = sample_data
        metrics = scorer.train(X, y)

        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1


class TestIsolationForestBasic:
    """Test basic Isolation Forest functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for anomaly detection"""
        np.random.seed(42)
        # Normal data
        normal = np.random.randn(80, 5)
        # Anomalies
        anomalies = np.random.uniform(5, 10, (20, 5))
        X = np.vstack([normal, anomalies])
        return X

    def test_isolation_forest_initialization(self):
        """Test IsolationForest can be initialized"""
        model = IsolationForest(n_estimators=100, random_state=42)
        assert model is not None

    def test_isolation_forest_training(self, sample_data):
        """Test IsolationForest training"""
        model = IsolationForest(n_estimators=100, random_state=42)
        model.fit(sample_data)

        assert model is not None

    def test_isolation_forest_prediction(self, sample_data):
        """Test IsolationForest prediction"""
        model = IsolationForest(n_estimators=100, random_state=42)
        model.fit(sample_data)

        predictions = model.predict(sample_data)

        assert len(predictions) == len(sample_data)
        assert set(predictions) == {-1, 1}

    def test_isolation_forest_anomaly_detection(self, sample_data):
        """Test that IsolationForest detects anomalies correctly"""
        model = IsolationForest(
            n_estimators=100,
            contamination=0.2,  # 20% anomalies
            random_state=42
        )
        model.fit(sample_data)
        predictions = model.predict(sample_data)

        # Should detect roughly 20% as anomalies
        anomaly_ratio = (predictions == -1).sum() / len(predictions)
        assert 0.15 < anomaly_ratio < 0.25  # Allow ±5% tolerance
