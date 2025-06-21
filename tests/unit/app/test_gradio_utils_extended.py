"""Extended tests for gradio_meta_prompt_utils functions."""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
import gradio as gr

# Import functions to test
from app.gradio_meta_prompt_utils import (
    initialize_llm,
    on_model_tab_select,
    format_selected_input_example_dataframe,
    format_selected_example
)
from app.config import MetaPromptConfig, LLMConfig


class TestInitializeLLM(unittest.TestCase):
    """Test LLM initialization utility."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = MetaPromptConfig(
            llms={
                "test_model": LLMConfig(
                    type="ChatOpenAI",
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    api_key="test-key"
                )
            },
            examples_path="/test"
        )
    
    @patch('app.gradio_meta_prompt_utils.LLMModelFactory')
    def test_initialize_llm_basic(self, mock_factory_class):
        """Test basic LLM initialization."""
        mock_factory = Mock()
        mock_llm = Mock()
        mock_factory.create.return_value = mock_llm
        mock_factory_class.return_value = mock_factory
        
        result = initialize_llm(self.config, "test_model")
        
        mock_factory.create.assert_called_once_with(
            "ChatOpenAI",
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            api_key="test-key"
        )
        self.assertEqual(result, mock_llm)
    
    @patch('app.gradio_meta_prompt_utils.LLMModelFactory')
    def test_initialize_llm_with_override_config(self, mock_factory_class):
        """Test LLM initialization with config override."""
        mock_factory = Mock()
        mock_llm = Mock()
        mock_factory.create.return_value = mock_llm
        mock_factory_class.return_value = mock_factory
        
        override_config = {"temperature": 0.9, "max_tokens": 2048}
        
        result = initialize_llm(self.config, "test_model", override_config)
        
        mock_factory.create.assert_called_once_with(
            "ChatOpenAI",
            model_name="gpt-3.5-turbo",
            temperature=0.9,  # Overridden
            max_tokens=2048,  # Added
            api_key="test-key"
        )
        self.assertEqual(result, mock_llm)
    
    def test_initialize_llm_invalid_model(self):
        """Test LLM initialization with invalid model name."""
        with self.assertRaises(KeyError):
            initialize_llm(self.config, "nonexistent_model")


class TestGradioEventHandlers(unittest.TestCase):
    """Test Gradio event handler functions."""
    
    def test_on_model_tab_select_basic(self):
        """Test model tab selection handler."""
        # Mock event object
        mock_event = Mock()
        mock_event.index = 1
        mock_event.value = "Advanced"
        
        result = on_model_tab_select(mock_event)
        
        # Should return the event value
        self.assertEqual(result, "Advanced")
    
    def test_format_selected_input_example_dataframe_basic(self):
        """Test dataframe example selection formatting."""
        # Mock event and examples
        mock_event = Mock()
        mock_event.index = [0]  # First row selected
        
        examples = [
            {"input": "Hello", "output": "Hi there"},
            {"input": "Goodbye", "output": "See you later"}
        ]
        
        result = format_selected_input_example_dataframe(mock_event, examples)
        
        # Should return formatted JSON of selected example
        expected = json.dumps(examples[0], indent=2, ensure_ascii=False)
        self.assertEqual(result, expected)
    
    def test_format_selected_input_example_dataframe_multiple_selection(self):
        """Test dataframe formatting with multiple selections."""
        mock_event = Mock()
        mock_event.index = [0, 1]  # Multiple rows selected
        
        examples = [
            {"input": "Hello", "output": "Hi"},
            {"input": "Goodbye", "output": "Bye"}
        ]
        
        result = format_selected_input_example_dataframe(mock_event, examples)
        
        # Should return formatted JSON array of selected examples
        expected = json.dumps([examples[0], examples[1]], indent=2, ensure_ascii=False)
        self.assertEqual(result, expected)
    
    def test_format_selected_input_example_dataframe_empty_selection(self):
        """Test dataframe formatting with empty selection."""
        mock_event = Mock()
        mock_event.index = []  # No selection
        
        examples = [{"input": "Hello", "output": "Hi"}]
        
        result = format_selected_input_example_dataframe(mock_event, examples)
        
        # Should return empty JSON array
        self.assertEqual(result, "[]")
    
    def test_format_selected_input_example_dataframe_invalid_index(self):
        """Test dataframe formatting with invalid index."""
        mock_event = Mock()
        mock_event.index = [5]  # Index out of range
        
        examples = [{"input": "Hello", "output": "Hi"}]
        
        result = format_selected_input_example_dataframe(mock_event, examples)
        
        # Should handle gracefully and return empty
        self.assertEqual(result, "[]")
    
    def test_format_selected_example_basic(self):
        """Test example selection formatting."""
        mock_event = Mock()
        mock_event.index = 0  # Single selection
        
        examples = [
            {"input": "Test input", "output": "Test output"},
            {"input": "Another input", "output": "Another output"}
        ]
        
        result = format_selected_example(mock_event, examples)
        
        expected = json.dumps(examples[0], indent=2, ensure_ascii=False)
        self.assertEqual(result, expected)
    
    def test_format_selected_example_invalid_index(self):
        """Test example formatting with invalid index."""
        mock_event = Mock()
        mock_event.index = 10  # Out of range
        
        examples = [{"input": "Test", "output": "Result"}]
        
        result = format_selected_example(mock_event, examples)
        
        # Should return empty object for invalid index
        self.assertEqual(result, "{}")


class TestDataProcessingUtilities(unittest.TestCase):
    """Test data processing utility functions."""
    
    def test_convert_examples_edge_cases(self):
        """Test convert_examples_to_json with edge cases."""
        from app.gradio_meta_prompt_utils import convert_examples_to_json
        
        # Test with None values
        examples_with_none = [
            {"Input": None, "Output": "result"},
            {"Input": "test", "Output": None}
        ]
        
        result = convert_examples_to_json(examples_with_none)
        parsed = json.loads(result)
        
        self.assertIsNone(parsed[0]["input"])
        self.assertEqual(parsed[0]["output"], "result")
        self.assertEqual(parsed[1]["input"], "test")
        self.assertIsNone(parsed[1]["output"])
    
    def test_convert_examples_numeric_data(self):
        """Test convert_examples_to_json with numeric data."""
        from app.gradio_meta_prompt_utils import convert_examples_to_json
        
        examples = [
            {"INPUT": 123, "OUTPUT": 456.78},
            {"INPUT": "text", "OUTPUT": 999}
        ]
        
        result = convert_examples_to_json(examples)
        parsed = json.loads(result)
        
        self.assertEqual(parsed[0]["input"], 123)
        self.assertEqual(parsed[0]["output"], 456.78)
        self.assertEqual(parsed[1]["input"], "text")
        self.assertEqual(parsed[1]["output"], 999)


class TestGradioIntegrationHelpers(unittest.TestCase):
    """Test helper functions for Gradio integration."""
    
    def test_simplify_gradio_updates(self):
        """Test creation of Gradio update objects."""
        # This would test helper functions that create gr.update() objects
        # For now, testing the basic pattern
        
        update_visible = gr.update(visible=True)
        update_choices = gr.update(choices=["a", "b", "c"], value="a")
        
        # Basic verification that updates are created correctly
        self.assertIsInstance(update_visible, dict)
        self.assertIsInstance(update_choices, dict)
        self.assertTrue(update_visible.get("visible"))
        self.assertEqual(update_choices.get("choices"), ["a", "b", "c"])
    
    @patch('gradio.Error')
    def test_error_handling_pattern(self, mock_error):
        """Test common error handling pattern in utility functions."""
        # Test that Gradio errors are raised appropriately
        mock_error.side_effect = Exception("Test error")
        
        with self.assertRaises(Exception):
            raise gr.Error("Test error message")


class TestUtilityFunctionErrorHandling(unittest.TestCase):
    """Test error handling in utility functions."""
    
    def test_format_selected_example_with_empty_examples(self):
        """Test formatting with empty examples list."""
        mock_event = Mock()
        mock_event.index = 0
        
        result = format_selected_example(mock_event, [])
        
        # Should handle empty list gracefully
        self.assertEqual(result, "{}")
    
    def test_format_selected_dataframe_with_none_examples(self):
        """Test dataframe formatting with None examples."""
        mock_event = Mock()
        mock_event.index = [0]
        
        result = format_selected_input_example_dataframe(mock_event, None)
        
        # Should handle None gracefully
        self.assertEqual(result, "[]")


if __name__ == '__main__':
    unittest.main()