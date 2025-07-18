# Test Suite

This directory contains the test suite for the multi-view project.

## Structure

- `conftest.py` - Shared pytest fixtures and configuration
- `test_utils.py` - Tests for utility functions in `src.utils.utils`
- `__init__.py` - Makes tests a Python package

## Running Tests

### Prerequisites

Install pytest and required dependencies:

```bash
pip install pytest pytest-cov
```

### Running All Tests

From the project root directory:

```bash
pytest
```

### Running Specific Test Files

```bash
# Run only utils tests
pytest tests/test_utils.py

# Run with verbose output
pytest -v tests/test_utils.py
```

### Running Specific Test Classes

```bash
# Run only config saving tests
pytest tests/test_utils.py::TestConfigSaving
```

### Running Specific Test Methods

```bash
# Run only the save_training_config test
pytest tests/test_utils.py::TestConfigSaving::test_save_training_config
```

### Running Tests with Coverage

```bash
# Run tests with coverage report
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
```

## Test Organization

### Fixtures

Shared fixtures are defined in `conftest.py`:

- `temp_dir` - Creates a temporary directory for each test
- `sample_config` - Sample configuration for testing
- `sample_training_info` - Sample training information for testing
- `mock_args` - Mock command line arguments for testing

### Test Classes

- `TestConfigSaving` - Tests for configuration saving functions

### Test Methods

Each test method focuses on testing a specific function or behavior:

- `test_save_training_config` - Tests the `save_training_config` function
- `test_save_training_config_summary` - Tests the `save_training_config_summary` function
- `test_save_environment_info` - Tests the `save_environment_info` function
- `test_save_all_training_info` - Tests the `save_all_training_info` function
- `test_saved_files_are_readable` - Tests that saved files are readable
- `test_file_paths_are_unique` - Tests that file paths are unique
- `test_files_are_in_temp_directory` - Tests that files are saved in the correct directory

## Adding New Tests

1. Create a new test file following the naming convention `test_*.py`
2. Import the functions you want to test from the appropriate module
3. Create test classes that inherit from `object` (pytest doesn't require unittest.TestCase)
4. Use the shared fixtures from `conftest.py` when possible
5. Write descriptive test method names that start with `test_`

## Best Practices

- Use descriptive test names that explain what is being tested
- Use fixtures for common setup and teardown
- Test both success and failure cases
- Use assertions to verify expected behavior
- Clean up any temporary files or resources
- Keep tests independent of each other 