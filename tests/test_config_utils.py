"""Test configuration utilities for loading LLM settings from config.yml."""

import os
import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional

def load_test_config() -> Dict[str, Any]:
    """Load configuration from config.yml file."""
    config_path = Path(__file__).parent.parent / "config.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_test_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    """Get a configured LLM instance for testing.
    
    Args:
        model_name: Optional specific model name. If not provided, uses first available model.
        
    Returns:
        Configured ChatOpenAI instance for testing.
    """
    config = load_test_config()
    llms_config = config.get('llms', {})
    
    if not llms_config:
        raise ValueError("No LLM configurations found in config.yml")
    
    # Use specified model or first available model
    if model_name and model_name in llms_config:
        llm_config = llms_config[model_name]
    else:
        # Get first available model
        llm_config = next(iter(llms_config.values()))
    
    # Create ChatOpenAI instance with config settings
    return ChatOpenAI(
        model_name=llm_config.get('model_name'),
        openai_api_key=llm_config.get('openai_api_key'),
        openai_api_base=llm_config.get('openai_api_base'),
        temperature=llm_config.get('temperature', 0.1),
        max_tokens=llm_config.get('max_tokens', 8192),
        verbose=llm_config.get('verbose', False),
        max_retries=3
    )

def get_test_llms_dict() -> Dict[str, ChatOpenAI]:
    """Get a dictionary of all configured LLMs for testing.
    
    Returns:
        Dictionary mapping node names to configured ChatOpenAI instances.
    """
    from meta_prompt.consts import META_PROMPT_NODES
    
    # Use a single LLM instance for all nodes in tests
    test_llm = get_test_llm()
    
    return {node: test_llm for node in META_PROMPT_NODES}

def skip_if_no_api_key(test_func):
    """Decorator to skip tests if no valid API key is available."""
    import unittest
    import functools
    
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        try:
            get_test_llm()
        except (FileNotFoundError, ValueError, Exception):
            raise unittest.SkipTest("Skipping test: No valid LLM configuration available")
        return test_func(*args, **kwargs)
    return wrapper