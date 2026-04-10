"""
Test suite for configuration module
Tests for config constants and environment setup
"""
import pytest
import os
from src import config


class TestConfigValues:
    """Test configuration values are correctly set"""

    def test_api_model_is_set(self):
        """Test OpenAI model configuration is set"""
        assert config.OPENAI_MODEL_MAIN == "gpt-4o-mini"
        assert config.OPENAI_MODEL_REPORT == "gpt-4o"

    def test_api_token_limits(self):
        """Test token limits are reasonable"""
        assert config.OPENAI_MAX_TOKENS > 0
        assert config.OPENAI_MAX_TOKENS <= 4000

    def test_api_temperature(self):
        """Test temperature is in valid range"""
        assert 0 <= config.OPENAI_TEMPERATURE <= 1

    def test_retry_configuration(self):
        """Test retry configuration values"""
        assert config.OPENAI_RETRY_ATTEMPTS >= 1
        assert config.OPENAI_RETRY_DELAY >= 0
        assert config.OPENAI_RETRY_MAX_DELAY > config.OPENAI_RETRY_DELAY

    def test_data_paths_exist(self):
        """Test that data directory is configured"""
        assert config.DATA_DIR == "data/raw"
        assert "X_processed.csv" in config.DATA_X_PATH
        assert "y.csv" in config.DATA_Y_PATH

    def test_model_parameters(self):
        """Test model parameter values"""
        assert config.ISOLATION_FOREST_N_ESTIMATORS > 0
        assert 0 < config.ISOLATION_FOREST_CONTAMINATION < 1
        assert config.PCA_N_COMPONENTS > 0

    def test_process_thresholds(self):
        """Test process thresholds are configured"""
        assert "CVD" in config.PROCESS_THRESHOLDS
        assert "ETCH" in config.PROCESS_THRESHOLDS
        assert "CMP" in config.PROCESS_THRESHOLDS
        assert "LITHO" in config.PROCESS_THRESHOLDS

        for process, thresholds in config.PROCESS_THRESHOLDS.items():
            assert "warning" in thresholds
            assert "critical" in thresholds
            # Critical should be higher than warning
            assert thresholds["critical"] > thresholds["warning"]

    def test_process_order(self):
        """Test process order is defined"""
        assert len(config.PROCESS_ORDER) == 4
        assert "CVD" in config.PROCESS_ORDER
        assert "ETCH" in config.PROCESS_ORDER

    def test_colors_defined(self):
        """Test UI colors are defined"""
        assert len(config.PROCESS_COLORS) == 4
        assert config.ALERT_COLOR_CRITICAL.startswith("#")
        assert config.ALERT_COLOR_WARNING.startswith("#")
        assert config.ALERT_COLOR_NORMAL.startswith("#")

    def test_anomaly_thresholds(self):
        """Test anomaly risk thresholds"""
        assert config.ANOMALY_HIGH_RISK_THRESHOLD > config.ANOMALY_MEDIUM_RISK_THRESHOLD
        assert 0 < config.ANOMALY_MEDIUM_RISK_THRESHOLD < 1
        assert 0 < config.ANOMALY_HIGH_RISK_THRESHOLD <= 1

    def test_log_directory(self):
        """Test log directory exists"""
        assert os.path.exists(config.LOG_DIR)
        assert os.path.isdir(config.LOG_DIR)


class TestConfigConsistency:
    """Test configuration consistency across modules"""

    def test_max_api_calls_is_positive(self):
        """Test max API calls is positive"""
        assert config.STREAMLIT_MAX_API_CALLS > 0

    def test_all_required_configs_exist(self):
        """Test all required configuration keys exist"""
        required_attrs = [
            "OPENAI_API_KEY",
            "OPENAI_MODEL_MAIN",
            "OPENAI_RETRY_ATTEMPTS",
            "DATA_DIR",
            "PROCESS_THRESHOLDS",
            "PROCESS_COLORS",
            "LOG_LEVEL",
        ]

        for attr in required_attrs:
            assert hasattr(config, attr), f"Missing config: {attr}"
