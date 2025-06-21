# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meta-Prompt is a system for automatically generating and optimizing language model prompts using LangGraph workflows. It uses LLMs to create, test, and iteratively improve system messages through a graph-based optimization process.

## Development Commands

### Setup and Installation
```bash
poetry install --with=dev
cp example_config.yml config.yml  # Edit with your API keys
```

### Running Applications
```bash
# Primary Gradio web interface
poetry run python app/gradio_meta_prompt.py

# Alternative Streamlit interface  
poetry run streamlit run app/streamlit_tab_app.py

# Docker deployment
docker run -p 7860:7860 yaleh/meta-prompt
```

### Testing and Quality Assurance
```bash
# Run all tests (uses LLM configurations from config.yml)
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_meta_prompt_graph_workflow.py

# Run tests quietly without verbose output
poetry run pytest tests/ -q

# Lint and format code
poetry run ruff check .
poetry run ruff format .
```

**Testing Configuration:**
- Tests automatically use LLM settings from `config.yml`
- Test utility functions in `tests/test_config_utils.py` handle configuration loading
- Tests requiring LLM API access use the `@skip_if_no_api_key` decorator
- All tests use mocked LLM interactions or actual API calls depending on configuration availability

### Development Workflow
```bash
# Install new dependencies
poetry add package_name
poetry add --group dev dev_package_name

# Launch Jupyter for notebook development
poetry run jupyter lab

# Activate poetry shell for interactive development
poetry shell
```

## Core Architecture

### Meta-Prompt Graph System (`meta_prompt/`)
- **MetaPromptGraph**: Central LangGraph-based workflow orchestrator with multiple specialized nodes:
  - `prompt_initial_developer`: Creates initial system messages
  - `prompt_developer`: Iteratively improves prompts
  - `prompt_executor`: Tests prompts with user inputs
  - `output_history_analyzer`: Compares outputs to find optimal results
  - `prompt_analyzer`: Evaluates if outputs meet acceptance criteria
- **AgentState**: TypedDict-based state management with annotated merge strategies
- **TaskDescriptionGenerator**: Generates task descriptions from input/output examples
- **ThinkTagRemover**: Strips reasoning tokens from thinking models

### Application Layer (`app/`)
- **Gradio Interface**: Primary web UI with Scope Tab (task generation) and Prompt Tab (optimization)
- **Streamlit Interface**: Alternative dashboard-style UI
- **Utilities**: Helper functions for UI operations and config handling

### Configuration System
- **config.yml**: Multi-provider LLM configurations with node-specific settings
- **Prompt Templates**: Multiple template groups for different LLM families (GPT, Sonnet, Merged)
- **Workflow Parameters**: Recursion limits, output aging, acceptance criteria

## Key Development Patterns

### LangGraph Workflow Design
- State-driven execution with conditional edges based on acceptance criteria
- Node-specific model and temperature configuration
- Retry logic and fallback mechanisms for robust operation

### Multi-Model Strategy
- Different models for different workflow stages (initial creation vs. refinement)
- Support for thinking models with automatic reasoning token removal
- OpenAI-compatible API interface supporting various providers

### Testing Architecture
- Comprehensive pytest suite with mocked LLM interactions
- Parameterized tests for different workflow scenarios
- Separate tests for state management, node execution, and error handling

## Important File Locations

### Core Logic
- `meta_prompt/meta_prompt.py`: Main MetaPromptGraph implementation
- `meta_prompt/sample_generator.py`: Task description generation
- `meta_prompt/think_tag_remover.py`: Reasoning token cleanup

### UI Entry Points
- `app/gradio_meta_prompt.py`: Primary web interface
- `app/streamlit_tab_app.py`: Alternative interface
- `demo/`: Jupyter notebooks for experimentation

### Configuration
- `config.yml`: Main configuration file (copy from example_config.yml)
- `pyproject.toml`: Poetry dependencies and project metadata

### Testing
- `tests/test_meta_prompt_graph_*.py`: Core workflow tests
- `tests/test_task_description_generator.py`: Task generation tests
- `tests/test_think_tag_remover.py`: Token cleanup tests