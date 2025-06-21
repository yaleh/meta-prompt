"""Unit tests for app configuration module."""

import unittest
import tempfile
import os
import yaml
from unittest.mock import patch, mock_open

from app.config import MetaPromptConfig, LLMConfig, RoleMessage, PromptGroup


class TestRoleMessage(unittest.TestCase):
    """Test RoleMessage model."""
    
    def test_role_message_creation(self):
        """Test creating a RoleMessage instance"""
        message = RoleMessage(role="user", message="Hello world")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.message, "Hello world")
    
    def test_role_message_validation(self):
        """Test RoleMessage validation"""
        with self.assertRaises(ValueError):
            RoleMessage()  # Missing required fields


class TestLLMConfig(unittest.TestCase):
    """Test LLMConfig model."""
    
    def test_llm_config_creation(self):
        """Test creating an LLMConfig instance"""
        config = LLMConfig(type="openai")
        self.assertEqual(config.type, "openai")
    
    def test_llm_config_extra_fields(self):
        """Test LLMConfig allows extra fields"""
        config = LLMConfig(
            type="openai",
            model_name="gpt-4",
            temperature=0.7,
            api_key="test-key"
        )
        self.assertEqual(config.type, "openai")
        self.assertEqual(config.model_name, "gpt-4")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.api_key, "test-key")
    
    def test_llm_config_missing_type(self):
        """Test LLMConfig requires type field"""
        with self.assertRaises(ValueError):
            LLMConfig(model_name="gpt-4")


class TestPromptGroup(unittest.TestCase):
    """Test PromptGroup model."""
    
    def test_prompt_group_extra_fields(self):
        """Test PromptGroup allows extra fields"""
        group = PromptGroup(
            name="test_group",
            templates=["template1", "template2"],
            description="Test prompt group"
        )
        self.assertEqual(group.name, "test_group")
        self.assertEqual(group.templates, ["template1", "template2"])
        self.assertEqual(group.description, "Test prompt group")


class TestMetaPromptConfig(unittest.TestCase):
    """Test MetaPromptConfig configuration."""
    
    def test_meta_prompt_config_defaults(self):
        """Test MetaPromptConfig with default values"""
        config = MetaPromptConfig(
            llms={"test": LLMConfig(type="openai")},
            examples_path="/test/path"
        )
        self.assertEqual(config.default_llm_temperature, 0.1)
        self.assertFalse(config.aggressive_exploration)
        self.assertIsNone(config.server_name)
        self.assertIsNone(config.server_port)
        self.assertEqual(config.recursion_limit, 25)
        self.assertEqual(config.recursion_limit_max, 50)
        self.assertFalse(config.allow_flagging)
        self.assertFalse(config.verbose)
        self.assertEqual(config.max_output_age, 3)
        self.assertEqual(config.max_output_age_max, 8)
    
    def test_meta_prompt_config_with_values(self):
        """Test MetaPromptConfig with custom values"""
        llm_config = {"test_llm": LLMConfig(type="openai", model_name="gpt-4")}
        
        config = MetaPromptConfig(
            llms=llm_config,
            examples_path="/test/examples",
            default_llm_temperature=0.5,
            aggressive_exploration=True,
            server_name="localhost",
            server_port=8080,
            recursion_limit=30,
            allow_flagging=True,
            verbose=True,
            max_output_age=5
        )
        
        self.assertEqual(config.llms, llm_config)
        self.assertEqual(config.default_llm_temperature, 0.5)
        self.assertTrue(config.aggressive_exploration)
        self.assertEqual(config.server_name, "localhost")
        self.assertEqual(config.server_port, 8080)
        self.assertEqual(config.recursion_limit, 30)
        self.assertTrue(config.allow_flagging)
        self.assertTrue(config.verbose)
        self.assertEqual(config.max_output_age, 5)
    
    def test_meta_prompt_config_with_prompt_templates(self):
        """Test MetaPromptConfig with prompt templates"""
        templates = {
            "group1": {
                "template1": [
                    RoleMessage(role="system", message="You are a helpful assistant"),
                    RoleMessage(role="user", message="Hello")
                ]
            }
        }
        
        config = MetaPromptConfig(
            llms={"test": LLMConfig(type="openai")},
            examples_path="/test",
            prompt_templates=templates
        )
        self.assertEqual(config.prompt_templates, templates)
    
    def test_meta_prompt_config_extra_fields(self):
        """Test MetaPromptConfig allows extra fields"""
        config = MetaPromptConfig(
            llms={"test": LLMConfig(type="openai")},
            examples_path="/test",
            custom_field="custom_value",
            another_field=123
        )
        self.assertEqual(config.custom_field, "custom_value")
        self.assertEqual(config.another_field, 123)


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading from files."""
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "llms": {
                "openai": {
                    "type": "openai",
                    "model_name": "gpt-4",
                    "temperature": 0.7
                }
            },
            "examples_path": "/test/examples",
            "default_llm_temperature": 0.3,
            "recursion_limit": 20,
            "verbose": True
        }
        
        config = MetaPromptConfig(**config_dict)
        self.assertIsNotNone(config.llms)
        self.assertEqual(config.default_llm_temperature, 0.3)
        self.assertEqual(config.recursion_limit, 20)
        self.assertTrue(config.verbose)
    
    def test_config_with_nested_llm_configs(self):
        """Test config with properly structured LLM configurations"""
        config_dict = {
            "llms": {
                "openai_gpt4": {
                    "type": "openai",
                    "model_name": "gpt-4",
                    "openai_api_key": "sk-test",
                    "temperature": 0.7,
                    "max_tokens": 4096
                },
                "anthropic_claude": {
                    "type": "anthropic", 
                    "model_name": "claude-3-sonnet",
                    "anthropic_api_key": "test-key",
                    "temperature": 0.5
                }
            },
            "examples_path": "/test/examples"
        }
        
        config = MetaPromptConfig(**config_dict)
        self.assertEqual(len(config.llms), 2)
        self.assertIn("openai_gpt4", config.llms)
        self.assertIn("anthropic_claude", config.llms)
        
        openai_config = config.llms["openai_gpt4"]
        self.assertEqual(openai_config.type, "openai")
        self.assertEqual(openai_config.model_name, "gpt-4")
        
    def test_config_validation_errors(self):
        """Test config validation for invalid configurations"""
        # Test invalid LLM config (missing type)
        with self.assertRaises(ValueError):
            MetaPromptConfig(
                llms={
                    "invalid_llm": {
                        "model_name": "gpt-4"  # Missing required 'type' field
                    }
                },
                examples_path="/test"
            )
    
    def test_config_with_role_messages(self):
        """Test config with RoleMessage objects in prompt templates"""
        config_dict = {
            "llms": {"test": {"type": "openai"}},
            "examples_path": "/test",
            "prompt_templates": {
                "default": {
                    "system_prompt": [
                        {"role": "system", "message": "You are helpful"},
                        {"role": "user", "message": "Hello"}
                    ]
                }
            }
        }
        
        config = MetaPromptConfig(**config_dict)
        self.assertIsNotNone(config.prompt_templates)
        self.assertIn("default", config.prompt_templates)


class TestConfigFileOperations(unittest.TestCase):
    """Test file-based configuration operations."""
    
    def test_config_yaml_structure_validation(self):
        """Test that we can validate expected YAML config structure"""
        yaml_content = """
llms:
  openai_gpt4:
    type: openai
    model_name: gpt-4
    openai_api_key: sk-test
    temperature: 0.7

examples_path: /test/examples
default_llm_temperature: 0.1
recursion_limit: 25
verbose: false
max_output_age: 3
"""
        
        config_data = yaml.safe_load(yaml_content)
        config = MetaPromptConfig(**config_data)
        
        self.assertIsNotNone(config.llms)
        self.assertEqual(config.default_llm_temperature, 0.1)
        self.assertEqual(config.recursion_limit, 25)
        self.assertFalse(config.verbose)
    
    def test_partial_config_loading(self):
        """Test loading config with only some fields specified"""
        partial_config = {
            "llms": {"test": {"type": "openai"}},
            "examples_path": "/test",
            "recursion_limit": 15,
            "verbose": True
        }
        
        config = MetaPromptConfig(**partial_config)
        # Should use defaults for unspecified fields
        self.assertEqual(config.recursion_limit, 15)
        self.assertTrue(config.verbose)
        self.assertEqual(config.default_llm_temperature, 0.1)  # default


if __name__ == '__main__':
    unittest.main()