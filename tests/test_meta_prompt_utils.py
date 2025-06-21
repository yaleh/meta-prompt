"""Unit tests for meta_prompt_utils module."""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.meta_prompt_utils import (
    prompt_templates_confz2langchain,
    LLMModelFactory,
    chat_log_2_chatbot_list,
    get_current_model
)
from app.config import MetaPromptConfig, LLMConfig, RoleMessage


class TestPromptTemplateConversion(unittest.TestCase):
    """Test prompt template conversion utilities."""
    
    def test_prompt_templates_confz2langchain_basic(self):
        """Test basic prompt template conversion"""
        confz_templates = {
            "system_node": [
                RoleMessage(role="system", message="You are a helpful assistant"),
                RoleMessage(role="user", message="Hello")
            ],
            "user_node": [
                RoleMessage(role="user", message="What is AI?"),
                RoleMessage(role="assistant", message="AI stands for...")
            ]
        }
        
        result = prompt_templates_confz2langchain(confz_templates)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn("system_node", result)
        self.assertIn("user_node", result)
        
        # Check that each result is a ChatPromptTemplate
        for template in result.values():
            self.assertIsInstance(template, ChatPromptTemplate)
    
    def test_prompt_templates_confz2langchain_empty(self):
        """Test conversion with empty templates"""
        confz_templates = {}
        result = prompt_templates_confz2langchain(confz_templates)
        self.assertEqual(result, {})
    
    def test_prompt_templates_confz2langchain_single_message(self):
        """Test conversion with single message templates"""
        confz_templates = {
            "single_node": [
                RoleMessage(role="system", message="Single message")
            ]
        }
        
        result = prompt_templates_confz2langchain(confz_templates)
        
        self.assertEqual(len(result), 1)
        self.assertIn("single_node", result)
        template = result["single_node"]
        self.assertIsInstance(template, ChatPromptTemplate)


class TestLLMModelFactory(unittest.TestCase):
    """Test LLM model factory singleton pattern and creation."""
    
    def test_singleton_pattern(self):
        """Test that LLMModelFactory follows singleton pattern"""
        factory1 = LLMModelFactory()
        factory2 = LLMModelFactory()
        self.assertIs(factory1, factory2)
    
    @patch('app.meta_prompt_utils.globals')
    def test_create_model_success(self, mock_globals):
        """Test successful model creation"""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_globals.return_value = {"ChatOpenAI": mock_model_class}
        
        factory = LLMModelFactory()
        
        with patch.object(factory, 'create') as mock_create:
            mock_create.return_value = mock_model_instance
            result = factory.create("ChatOpenAI", temperature=0.7, model="gpt-4")
            mock_create.assert_called_once_with("ChatOpenAI", temperature=0.7, model="gpt-4")
            self.assertEqual(result, mock_model_instance)


class TestChatLogConversion(unittest.TestCase):
    """Test chat log to chatbot list conversion."""
    
    def test_chat_log_2_chatbot_list_basic(self):
        """Test basic chat log conversion"""
        chat_log = '''{"action": "invoke", "message": "Hello"}
{"action": "response", "message": "Hi there!"}
{"action": "invoke", "message": "How are you?"}
{"action": "response", "message": "I'm fine"}'''
        
        result = chat_log_2_chatbot_list(chat_log)
        
        expected = [
            ["Hello", None],
            [None, "Hi there!"],
            ["How are you?", None],
            [None, "I'm fine"]
        ]
        self.assertEqual(result, expected)
    
    def test_chat_log_2_chatbot_list_empty(self):
        """Test conversion with empty chat log"""
        result = chat_log_2_chatbot_list("")
        self.assertEqual(result, [])
        
        result = chat_log_2_chatbot_list(None)
        self.assertEqual(result, [])
    
    def test_chat_log_2_chatbot_list_invalid_json(self):
        """Test conversion with invalid JSON lines"""
        chat_log = '''{"action": "invoke", "message": "Hello"}
invalid json line
{"action": "response", "message": "Hi"}'''
        
        with patch('builtins.print') as mock_print:
            result = chat_log_2_chatbot_list(chat_log)
            
            expected = [
                ["Hello", None],
                [None, "Hi"]
            ]
            self.assertEqual(result, expected)
            # Check that error was printed
            mock_print.assert_called()
    
    def test_chat_log_2_chatbot_list_missing_keys(self):
        """Test conversion with missing required keys"""
        chat_log = '''{"action": "invoke", "message": "Hello"}
{"no_action": "response", "message": "Hi"}
{"action": "response", "message": "Working"}'''
        
        with patch('builtins.print') as mock_print:
            result = chat_log_2_chatbot_list(chat_log)
            
            expected = [
                ["Hello", None],
                [None, "Working"]
            ]
            self.assertEqual(result, expected)
            # Check that error was printed for missing key
            mock_print.assert_called()
    
    def test_chat_log_2_chatbot_list_unknown_action(self):
        """Test conversion with unknown action types"""
        chat_log = '''{"action": "invoke", "message": "Hello"}
{"action": "unknown", "message": "Should be ignored"}
{"action": "response", "message": "Hi"}'''
        
        result = chat_log_2_chatbot_list(chat_log)
        
        expected = [
            ["Hello", None],
            [None, "Hi"]
        ]
        self.assertEqual(result, expected)


class TestGetCurrentModel(unittest.TestCase):
    """Test model retrieval functionality."""
    
    def setUp(self):
        """Set up test configuration"""
        self.llm_configs = {
            "simple_model": LLMConfig(
                type="ChatOpenAI",
                model_name="gpt-3.5-turbo",
                temperature=0.1
            ),
            "advanced_model": LLMConfig(
                type="ChatOpenAI", 
                model_name="gpt-4",
                temperature=0.3
            ),
            "expert_model": LLMConfig(
                type="ChatOpenAI",
                model_name="gpt-4-turbo",
                temperature=0.5
            )
        }
        
        self.config = MetaPromptConfig(llms=self.llm_configs)
    
    @patch('app.meta_prompt_utils.LLMModelFactory')
    def test_get_current_model_simple(self, mock_factory_class):
        """Test getting simple model"""
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        mock_factory_class.return_value = mock_factory
        
        result = get_current_model(
            "simple_model", "advanced_model", "expert_model",
            config=self.config, active_model_tab="Simple"
        )
        
        mock_factory.create.assert_called_once()
        args, kwargs = mock_factory.create.call_args
        self.assertEqual(args[0], "ChatOpenAI")
        self.assertIn("model_name", kwargs)
        self.assertEqual(kwargs["model_name"], "gpt-3.5-turbo")
    
    @patch('app.meta_prompt_utils.LLMModelFactory')
    def test_get_current_model_advanced(self, mock_factory_class):
        """Test getting advanced model"""
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        mock_factory_class.return_value = mock_factory
        
        result = get_current_model(
            "simple_model", "advanced_model", "expert_model",
            config=self.config, active_model_tab="Advanced"
        )
        
        mock_factory.create.assert_called_once()
        args, kwargs = mock_factory.create.call_args
        self.assertEqual(args[0], "ChatOpenAI")
        self.assertEqual(kwargs["model_name"], "gpt-4")
    
    @patch('app.meta_prompt_utils.LLMModelFactory')
    def test_get_current_model_expert_with_config(self, mock_factory_class):
        """Test getting expert model with additional config"""
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        mock_factory_class.return_value = mock_factory
        
        expert_config = {"temperature": 0.8, "max_tokens": 2048}
        
        result = get_current_model(
            "simple_model", "advanced_model", "expert_model",
            expert_model_config=expert_config,
            config=self.config, 
            active_model_tab="Expert"
        )
        
        mock_factory.create.assert_called_once()
        args, kwargs = mock_factory.create.call_args
        self.assertEqual(args[0], "ChatOpenAI")
        self.assertEqual(kwargs["temperature"], 0.8)  # Should override
        self.assertEqual(kwargs["max_tokens"], 2048)   # Should be added
    
    def test_get_current_model_invalid_model_name(self):
        """Test error handling for invalid model name"""
        with self.assertRaises(ValueError) as context:
            get_current_model(
                "nonexistent_model", "advanced_model", "expert_model",
                config=self.config, active_model_tab="Simple"
            )
        
        self.assertIn("Invalid model name", str(context.exception))
    
    def test_get_current_model_invalid_tab(self):
        """Test fallback to simple model for invalid tab"""
        with patch('app.meta_prompt_utils.LLMModelFactory') as mock_factory_class:
            mock_factory = Mock()
            mock_model = Mock()
            mock_factory.create.return_value = mock_model
            mock_factory_class.return_value = mock_factory
            
            result = get_current_model(
                "simple_model", "advanced_model", "expert_model",
                config=self.config, active_model_tab="InvalidTab"
            )
            
            # Should fall back to simple model
            mock_factory.create.assert_called_once()
            args, kwargs = mock_factory.create.call_args
            self.assertEqual(kwargs["model_name"], "gpt-3.5-turbo")


if __name__ == '__main__':
    unittest.main()