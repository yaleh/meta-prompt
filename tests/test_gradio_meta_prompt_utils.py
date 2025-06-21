"""Unit tests for gradio_meta_prompt_utils module."""

import unittest
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
from difflib import Differ

from app.gradio_meta_prompt_utils import (
    convert_examples_to_json,
    compare_outputs,
    prompt_templates_confz2langchain,
    SimplifiedCSVLogger
)
from app.config import RoleMessage


class TestConvertExamplesToJson(unittest.TestCase):
    """Test examples to JSON conversion utilities."""
    
    def test_convert_examples_to_json_basic(self):
        """Test basic conversion of examples to JSON"""
        examples = [
            {"Input": "Hello", "Output": "Hi there"},
            {"Input": "Goodbye", "Output": "See you later"}
        ]
        
        result = convert_examples_to_json(examples)
        parsed_result = json.loads(result)
        
        self.assertIsInstance(parsed_result, list)
        self.assertEqual(len(parsed_result), 2)
        
        # Check that columns are lowercase
        self.assertIn("input", parsed_result[0])
        self.assertIn("output", parsed_result[0])
        self.assertEqual(parsed_result[0]["input"], "Hello")
        self.assertEqual(parsed_result[0]["output"], "Hi there")
    
    def test_convert_examples_to_json_empty(self):
        """Test conversion with empty examples"""
        examples = []
        result = convert_examples_to_json(examples)
        
        # Empty DataFrame to_json returns "[]"
        self.assertEqual(result, "[]")
    
    def test_convert_examples_to_json_mixed_case(self):
        """Test conversion with mixed case column names"""
        examples = [
            {"INPUT": "Test", "OUTPUT": "Result", "ExtrA": "data"},
            {"INPUT": "Test2", "OUTPUT": "Result2", "ExtrA": "data2"}
        ]
        
        result = convert_examples_to_json(examples)
        parsed_result = json.loads(result)
        
        # All column names should be lowercase
        self.assertIn("input", parsed_result[0])
        self.assertIn("output", parsed_result[0])
        self.assertIn("extra", parsed_result[0])
        self.assertEqual(parsed_result[0]["input"], "Test")
        self.assertEqual(parsed_result[0]["extra"], "data")
    
    def test_convert_examples_to_json_special_characters(self):
        """Test conversion with special characters and unicode"""
        examples = [
            {"Input": "Hello 🌍", "Output": "Hi there! 😊"},
            {"Input": "Test \"quotes\"", "Output": "Result with 'quotes'"}
        ]
        
        result = convert_examples_to_json(examples)
        parsed_result = json.loads(result)
        
        self.assertEqual(parsed_result[0]["input"], "Hello 🌍")
        self.assertEqual(parsed_result[0]["output"], "Hi there! 😊")
        self.assertEqual(parsed_result[1]["input"], "Test \"quotes\"")
        self.assertEqual(parsed_result[1]["output"], "Result with 'quotes'")


class TestCompareOutputs(unittest.TestCase):
    """Test output comparison utilities."""
    
    def test_compare_outputs_identical(self):
        """Test comparison of identical outputs"""
        expected = "Hello world test"
        actual = "Hello world test"
        
        result = compare_outputs(expected, actual)
        
        # All tokens should have no change marker (None)
        for token, marker in result:
            self.assertIsNone(marker)
        
        # Should have all the words
        tokens = [token for token, marker in result]
        self.assertIn("Hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)
    
    def test_compare_outputs_different(self):
        """Test comparison of different outputs"""
        expected = "Hello world"
        actual = "Hello universe"
        
        result = compare_outputs(expected, actual)
        
        # Convert to dict for easier testing
        token_markers = {token: marker for token, marker in result}
        
        # "Hello" should be unchanged
        self.assertIn("Hello", token_markers)
        self.assertIsNone(token_markers["Hello"])
        
        # "world" should be removed (-), "universe" should be added (+)
        self.assertIn("world", token_markers)
        self.assertEqual(token_markers["world"], "-")
        self.assertIn("universe", token_markers)
        self.assertEqual(token_markers["universe"], "+")
    
    def test_compare_outputs_addition(self):
        """Test comparison with additions"""
        expected = "Hello"
        actual = "Hello world"
        
        result = compare_outputs(expected, actual)
        token_markers = {token: marker for token, marker in result}
        
        self.assertIsNone(token_markers["Hello"])  # unchanged
        self.assertEqual(token_markers["world"], "+")  # added
    
    def test_compare_outputs_deletion(self):
        """Test comparison with deletions"""
        expected = "Hello world"
        actual = "Hello"
        
        result = compare_outputs(expected, actual)
        token_markers = {token: marker for token, marker in result}
        
        self.assertIsNone(token_markers["Hello"])  # unchanged
        self.assertEqual(token_markers["world"], "-")  # removed
    
    def test_compare_outputs_empty_strings(self):
        """Test comparison with empty strings"""
        result1 = compare_outputs("", "")
        self.assertEqual(result1, [])
        
        result2 = compare_outputs("", "Hello")
        self.assertEqual(len(result2), 1)
        self.assertEqual(result2[0], ("Hello", "+"))
        
        result3 = compare_outputs("Hello", "")
        self.assertEqual(len(result3), 1)
        self.assertEqual(result3[0], ("Hello", "-"))
    
    def test_compare_outputs_complex_difference(self):
        """Test comparison with complex differences"""
        expected = "The quick brown fox jumps"
        actual = "The slow brown dog walks"
        
        result = compare_outputs(expected, actual)
        token_markers = {token: marker for token, marker in result}
        
        # Unchanged words
        self.assertIsNone(token_markers["The"])
        self.assertIsNone(token_markers["brown"])
        
        # Changed words
        self.assertEqual(token_markers["quick"], "-")
        self.assertEqual(token_markers["slow"], "+")
        self.assertEqual(token_markers["fox"], "-")
        self.assertEqual(token_markers["dog"], "+")
        self.assertEqual(token_markers["jumps"], "-")
        self.assertEqual(token_markers["walks"], "+")


class TestPromptTemplateConversionGradio(unittest.TestCase):
    """Test prompt template conversion in gradio utils (duplicate testing for completeness)."""
    
    def test_prompt_templates_confz2langchain_single_template(self):
        """Test conversion of single prompt template"""
        confz_templates = {
            "test_node": [
                RoleMessage(role="system", message="You are helpful"),
                RoleMessage(role="user", message="Hello")
            ]
        }
        
        result = prompt_templates_confz2langchain(confz_templates)
        
        self.assertEqual(len(result), 1)
        self.assertIn("test_node", result)
        # Don't test internal structure as it's a LangChain object
    
    def test_prompt_templates_confz2langchain_multiple_templates(self):
        """Test conversion of multiple prompt templates"""
        confz_templates = {
            "system_node": [
                RoleMessage(role="system", message="System prompt")
            ],
            "user_node": [
                RoleMessage(role="user", message="User prompt"),
                RoleMessage(role="assistant", message="Assistant response")
            ]
        }
        
        result = prompt_templates_confz2langchain(confz_templates)
        
        self.assertEqual(len(result), 2)
        self.assertIn("system_node", result)
        self.assertIn("user_node", result)


class TestSimplifiedCSVLogger(unittest.TestCase):
    """Test SimplifiedCSVLogger functionality."""
    
    def setUp(self):
        """Set up test components and logger"""
        self.mock_components = [
            Mock(label="input_text"),
            Mock(label="output_text"),
            Mock(label=None)  # Component without label
        ]
        
        # Mock the flag method for components
        for component in self.mock_components:
            component.flag.return_value = "mocked_data"
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('csv.writer')
    @patch('app.gradio_meta_prompt_utils.Path')
    def test_simplified_csv_logger_flag_new_file(self, mock_path_class, mock_writer_class, mock_file, mock_exists):
        """Test CSV logger with new file creation"""
        mock_exists.return_value = False  # File doesn't exist
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        
        # Create logger
        logger = SimplifiedCSVLogger(
            components=self.mock_components,
            flagging_dir="/test/dir"
        )
        
        flag_data = ["test_input", "test_output", "test_component"]
        
        # Call flag method
        result = logger.flag(flag_data)
        
        # Verify writer was called to write headers and data
        self.assertEqual(mock_writer.writerow.call_count, 2)  # Headers + data
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('csv.writer')
    @patch('app.gradio_meta_prompt_utils.Path')
    def test_simplified_csv_logger_flag_existing_file(self, mock_path_class, mock_writer_class, mock_file, mock_exists):
        """Test CSV logger with existing file"""
        mock_exists.return_value = True  # File exists
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        
        logger = SimplifiedCSVLogger(
            components=self.mock_components,
            flagging_dir="/test/dir"
        )
        
        flag_data = ["test_input", "test_output", "test_component"]
        
        result = logger.flag(flag_data)
        
        # Should only write data, not headers
        self.assertEqual(mock_writer.writerow.call_count, 1)  # Only data


if __name__ == '__main__':
    unittest.main()