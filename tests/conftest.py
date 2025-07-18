"""
Shared pytest fixtures and configuration for the test suite.
"""

import pytest
import tempfile
import os
import sys

# Add the src directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists for the test session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_dir():
    """Create a temporary directory for individual tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "name": "mvmae",
            "model_params": {
                "hidden_size": 768,
                "num_hidden_layers": 12
            }
        },
        "training": {
            "num_epochs": 100,
            "train_batch_size": 32
        },
        "optimizer": {
            "lr": 1e-4,
            "wd": 0.01,
            "warmup_pct": 0.1
        }
    }


@pytest.fixture
def sample_training_info():
    """Sample training information for testing."""
    return {
        "epochs": 100,
        "total_steps": 1000,
        "learning_rate": 1e-4,
        "global_batch_size": 64,
        "local_batch_size": 32,
        "world_size": 2,
        "dataset_size": 10000,
        "steps_per_epoch": 10,
        "available_views": ["top", "bottom"],
        "num_views": 2,
        "weight_decay": 0.01,
        "warmup_percentage": 0.1,
        "scheduler_type": "OneCycleLR",
        "optimizer_type": "AdamW",
        "model_name": "mvmae",
        "seed": 42,
        "experiment_name": "test_experiment"
    }


@pytest.fixture
def mock_args():
    """Mock command line arguments for testing."""
    class MockArgs:
        def __init__(self):
            self.config = "configs/test.yaml"
            self.seed = 42
            self.resume = None
            self.resume_from_best = False
    
    return MockArgs() 