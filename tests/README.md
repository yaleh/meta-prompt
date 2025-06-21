# Test Organization

This directory contains all tests for the meta-prompt project, organized by test type and scope.

## Directory Structure

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── core/               # Core meta-prompt functionality
│   │   ├── test_meta_prompt_graph_*.py
│   │   ├── test_task_description_generator.py
│   │   └── test_think_tag_remover.py
│   ├── app/                # Application layer tests
│   │   ├── test_app_config.py
│   │   ├── test_gradio_meta_prompt_utils.py
│   │   └── test_gradio_utils_extended.py
│   └── utils/              # Utility function tests
│       ├── test_config_utils.py
│       └── test_meta_prompt_utils.py
├── integration/            # Integration tests (components working together)
│   └── test_integration_workflows.py
└── live/                   # Live tests with real LLM APIs
    ├── test_live_meta_prompt.py
    ├── test_simple_demo.py
    └── test_iterative_improvement.py
```

## Running Tests

### All Tests
```bash
poetry run pytest tests/ -v
```

### Unit Tests Only
```bash
poetry run pytest tests/unit/ -v
```

### Integration Tests
```bash
poetry run pytest tests/integration/ -v
```

### Live Tests (requires API keys in config.yml)
```bash
poetry run pytest tests/live/ -v
```

### Specific Test Categories
```bash
# Core functionality tests
poetry run pytest tests/unit/core/ -v

# Application layer tests
poetry run pytest tests/unit/app/ -v

# Utility function tests
poetry run pytest tests/unit/utils/ -v
```

## Test Types

### Unit Tests (`tests/unit/`)
- **Fast**: Complete in seconds
- **Isolated**: Test individual functions/classes
- **Mocked**: Use mocked LLM interactions
- **Always Run**: Part of standard test suite

### Integration Tests (`tests/integration/`)
- **Medium Speed**: May take longer due to workflow complexity
- **Component Integration**: Test how components work together
- **Partially Mocked**: Some real interactions, some mocked
- **Standard Suite**: Run with regular testing

### Live Tests (`tests/live/`)
- **Slow**: Require real API calls
- **Real APIs**: Use actual LLM services
- **Manual Execution**: Not part of automated CI
- **Demonstration**: Show real-world functionality

## Coverage

Unit and integration tests aim for high coverage of the codebase. Live tests provide validation that the system works with real APIs but are not included in coverage metrics.

Current coverage targets:
- `meta_prompt/` module: >95%
- `app/` module: >80%
- Overall project: >85%