"""Unit tests for the task description generator module."""

import json
import unittest
from unittest.mock import Mock, patch
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from meta_prompt.sample_generator import TaskDescriptionGenerator
from tests.test_config_utils import get_test_llm, skip_if_no_api_key

class TestTaskDescriptionGeneratorBasic(unittest.TestCase):
    """Basic test cases for TaskDescriptionGenerator."""

    def setUp(self):
        self.model = get_test_llm()
        self.generator = TaskDescriptionGenerator(self.model)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_description(self, mock_invoke):
        """Test the generate_description method with mocked ChatOpenAI invoke."""
        mock_invoke.return_value = '{"description": "Task Description: Describe a cat."}'
        input_json = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = self.generator.generate_description(input_json)
        self.assertEqual(
            description,
            {
                "description": "Task Description: Describe a cat.",
                "suggestions": []
            }
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_analyze_input(self, mock_invoke):
        mock_invoke.return_value = "Input Analysis: The input is an animal."
        description = "Task Description: Describe a cat."
        input_analysis = self.generator.analyze_input(description)
        self.assertEqual(
            input_analysis,
            "Input Analysis: The input is an animal."
        )


class TestTaskDescriptionGeneratorExamples(unittest.TestCase):
    """Test cases for TaskDescriptionGenerator focusing on examples."""

    def setUp(self):
        self.model = get_test_llm()
        self.generator = TaskDescriptionGenerator(self.model)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_briefs(self, mock_invoke):
        mock_invoke.return_value = (
            '{"new_example_briefs": [{"example_brief": "Brief 1"}, '
            '{"example_brief": "Brief 2"}]}'
        )
        description = "Task Description: Describe a cat."
        input_analysis = "Input Analysis: The input is an animal."
        generating_batch_size = 2
        briefs = self.generator.generate_briefs(
            description, input_analysis, generating_batch_size
        )
        self.assertEqual(
            briefs,
            [
                {"example_brief": "Brief 1"},
                {"example_brief": "Brief 2"}
            ]
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_examples_from_briefs(self, mock_invoke):
        mock_invoke.return_value = (
            '{"examples": [{"input": "Input 1", "output": "Output 1"}, '
            '{"input": "Input 2", "output": "Output 2"}]}'
        )
        description = "Task Description: Describe a cat."
        new_example_briefs = {
            "new_example_briefs": [
                {"example_brief": "Brief 1"},
                {"example_brief": "Brief 2"}
            ]
        }
        raw_example = json.dumps({"input": "A cat", "output": "A furry animal"})
        generating_batch_size = 2
        examples = self.generator.generate_examples_from_briefs(
            description,
            new_example_briefs,
            raw_example,
            generating_batch_size
        )
        self.assertEqual(
            examples,
            {
                "examples": [
                    {"input": "Input 1", "output": "Output 1"},
                    {"input": "Input 2", "output": "Output 2"}
                ]
            }
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_examples(self, mock_invoke):
        mock_invoke.return_value = (
            '{"examples": [{"input": "Input 1", "output": "Output 1"}, '
            '{"input": "Input 2", "output": "Output 2"}]}'
        )
        description = "Task Description: Describe a cat."
        raw_example = json.dumps({"input": "A cat", "output": "A furry animal"})
        generating_batch_size = 2
        examples = self.generator.generate_examples_directly(
            description, raw_example, generating_batch_size
        )
        self.assertEqual(
            examples,
            {
                "examples": [
                    {"input": "Input 1", "output": "Output 1"},
                    {"input": "Input 2", "output": "Output 2"}
                ]
            }
        )

class TestTaskDescriptionGeneratorSuggestions(unittest.TestCase):

    def setUp(self):
        self.model = get_test_llm()
        self.generator = TaskDescriptionGenerator(self.model)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_basic(self, mock_invoke):
        mock_invoke.side_effect = [
            '{"suggestions": [{"suggestion": "Specify cat breed"}, '
            '{"suggestion": "Include cat age"}]}',
            '{"suggestions": [{"suggestion": "Expand to all pets"}, '
            '{"suggestion": "Include habitat description"}]}'
        ]
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        self.assertEqual(len(result['suggestions']), 4)
        self.assertTrue(all(isinstance(s, str) for s in result['suggestions']))

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_empty_input(self, mock_invoke):
        mock_invoke.side_effect = [
            '{"suggestions": []}',
            '{"suggestions": []}'
        ]
        input_str = json.dumps({})
        description = ""
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        self.assertEqual(len(result['suggestions']), 0)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_long_input(self, mock_invoke):
        mock_invoke.side_effect = [
            '{"suggestions": [{"suggestion": "Summarize key points"}, '
            '{"suggestion": "Extract main themes"}]}',
            '{"suggestions": [{"suggestion": "Expand analysis scope"}, '
            '{"suggestion": "Include cross-references"}]}'
        ]
        input_str = json.dumps({"input": "A" * 1000, "output": "B" * 1000})
        description = "Task Description: Analyze a long text."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn("suggestions", result)
        self.assertEqual(len(result["suggestions"]), 4)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_complex_task(self, mock_invoke):
        mock_invoke.side_effect = [
            '{"suggestions": [{"suggestion": "Break down into subtasks"}, '
            '{"suggestion": "Specify input formats for each step"}]}',
            '{"suggestions": [{"suggestion": "Generalize to similar problem domains"}, '
            '{"suggestion": "Include error handling procedures"}]}'
        ]
        input_str = json.dumps({
            "input": "Complex task input",
            "output": "Complex task output"
        })
        description = "Task Description: Perform a complex multi-step analysis."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        self.assertEqual(len(result['suggestions']), 4)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_error_handling(self, mock_invoke):
        mock_invoke.side_effect = [
            Exception("API Error"),
            '{"suggestions": [{"suggestion": "Handle network errors"}, '
            '{"suggestion": "Implement retry logic"}]}'
        ]
        input_str = json.dumps({
            "input": "Error prone task",
            "output": "Error handling result"
        })
        description = "Task Description: Test error handling in a system."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        # Only generalization suggestions due to simulated error
        self.assertEqual(len(result['suggestions']), 2)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_format_validation(self, mock_invoke):
        mock_invoke.side_effect = [
            '{"suggestions": [{"suggestion": "Validate input format"}, '
            '{"suggestion": "Enforce output structure"}]}',
            '{"suggestions": [{"suggestion": "Allow flexible input formats"}, '
            '{"suggestion": "Generate multiple output formats"}]}'
        ]
        input_str = json.dumps({
            "input": "Unstructured data",
            "output": "Structured result"
        })
        description = "Task Description: Convert unstructured data to structured format."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        self.assertEqual(len(result['suggestions']), 4)
        self.assertEqual(
            sorted(result['suggestions']),
            sorted([
                "Validate input format",
                "Enforce output structure",
                "Allow flexible input formats",
                "Generate multiple output formats"
            ])
        )

class TestLoadAndValidateInput(unittest.TestCase):

    def setUp(self):
        self.model = get_test_llm()
        self.generator = TaskDescriptionGenerator(self.model)

    def test_valid_json_single_example(self):
        input_dict = {
            "input_str": '{"input": "A cat", "output": "A furry animal"}',
            "generating_batch_size": 3
        }
        result = self.generator.load_and_validate_input(input_dict)
        self.assertEqual(result, {
            "example": {"input": "A cat", "output": "A furry animal"},
            "generating_batch_size": 3
        })

    def test_valid_json_multiple_examples(self):
        input_dict = {
            "input_str": '[{"input": "A cat", "output": "A furry animal"}, '
                         '{"input": "A dog", "output": "A loyal pet"}]',
            "generating_batch_size": 3
        }
        result = self.generator.load_and_validate_input(input_dict)
        self.assertEqual(result, {
            "example": [
                {"input": "A cat", "output": "A furry animal"},
                {"input": "A dog", "output": "A loyal pet"}
            ],
            "generating_batch_size": 3
        })

    def test_valid_yaml_single_example(self):
        input_dict = {
            "input_str": "input: A cat\noutput: A furry animal",
            "generating_batch_size": 3
        }
        result = self.generator.load_and_validate_input(input_dict)
        self.assertEqual(result, {
            "example": {"input": "A cat", "output": "A furry animal"},
            "generating_batch_size": 3
        })

    def test_valid_yaml_multiple_examples(self):
        input_dict = {
            "input_str": "- input: A cat\n  output: A furry animal\n"
                         "- input: A dog\n  output: A loyal pet",
            "generating_batch_size": 3
        }
        result = self.generator.load_and_validate_input(input_dict)
        self.assertEqual(result, {
            "example": [
                {"input": "A cat", "output": "A furry animal"},
                {"input": "A dog", "output": "A loyal pet"}
            ],
            "generating_batch_size": 3
        })

    def test_invalid_json_format(self):
        input_dict = {
            "input_str": '{"input": "A cat", "output": "A furry animal"',
            "generating_batch_size": 3
        }
        with self.assertRaises(RuntimeError):
            self.generator.load_and_validate_input(input_dict)

    def test_invalid_yaml_format(self):
        input_dict = {
            "input_str": "input: A cat\noutput: A furry animal:",
            "generating_batch_size": 3
        }
        with self.assertRaises(RuntimeError):
            self.generator.load_and_validate_input(input_dict)

    def test_empty_input_string(self):
        input_dict = {
            "input_str": "",
            "generating_batch_size": 3
        }
        with self.assertRaises(RuntimeError):
            self.generator.load_and_validate_input(input_dict)

    def test_missing_input_field(self):
        input_dict = {
            "input_str": '{"output": "A furry animal"}',
            "generating_batch_size": 3
        }
        with self.assertRaises(RuntimeError):
            self.generator.load_and_validate_input(input_dict)

    def test_missing_output_field(self):
        input_dict = {
            "input_str": '{"input": "A cat"}',
            "generating_batch_size": 3
        }
        with self.assertRaises(RuntimeError):
            self.generator.load_and_validate_input(input_dict)

    def test_extra_fields(self):
        input_dict = {
            "input_str": '{"input": "A cat", "output": "A furry animal", "extra": "field"}',
            "generating_batch_size": 3
        }
        result = self.generator.load_and_validate_input(input_dict)
        self.assertEqual(
            result,
            {
                "example": {
                    "input": "A cat",
                    "output": "A furry animal",
                    "extra": "field"
                },
                "generating_batch_size": 3
            }
        )

    def test_nested_structures(self):
        input_dict = {
            "input_str": '{"input": {"animal": "cat", "age": 5}, "output": {"description": "A furry animal", "sound": "meow"}}',
            "generating_batch_size": 3
        }
        result = self.generator.load_and_validate_input(input_dict)
        self.assertEqual(
            result,
            {
                "example": {
                    "input": {"animal": "cat", "age": 5},
                    "output": {"description": "A furry animal", "sound": "meow"}
                },
                "generating_batch_size": 3
            }
        )

    def test_unicode_characters(self):
        input_dict = {
            "input_str": '{"input": "Un chat 🐱", "output": "Un animal pelucheux 🇫🇷"}',
            "generating_batch_size": 3
        }
        result = self.generator.load_and_validate_input(input_dict)
        self.assertEqual(
            result,
            {
                "example": {
                    "input": "Un chat 🐱",
                    "output": "Un animal pelucheux 🇫🇷"
                },
                "generating_batch_size": 3
            }
        )

    def test_without_generating_batch_size(self):
        input_dict = {
            "input_str": '{"input": "A cat", "output": "A furry animal"}'
        }
        result = self.generator.load_and_validate_input(input_dict)
        self.assertEqual(result, {"example": {"input": "A cat", "output": "A furry animal"}})


class TestTaskDescriptionGeneratorUpdateDescription(unittest.TestCase):
    """Test cases for the update_description method of TaskDescriptionGenerator."""

    def setUp(self):
        self.model = get_test_llm()
        self.generator = TaskDescriptionGenerator(self.model)

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_valid_inputs(self, mock_invoke):
        """Test update_description with valid description and suggestions."""
        mock_invoke.return_value = '{"description": "Updated Task Description: Describe a domestic cat."}'
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = ["Specify cat breed", "Include cat age"]
        updated_description = self.generator.update_description(
            input_str, description, suggestions
        )
        self.assertEqual(
            updated_description,
            {
                "description": "Updated Task Description: Describe a domestic cat.",
                "suggestions": [],
            },
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_multiple_suggestions(self, mock_invoke):
        """Test update_description with multiple suggestions across different dimensions."""
        mock_invoke.return_value = '{"description": "Updated Task Description: Provide a detailed description of a cat, including breed and age."}'
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = ["Specify cat breed", "Include cat age"]
        updated_description = self.generator.update_description(
            input_str, description, suggestions
        )
        self.assertEqual(
            updated_description,
            {
                "description": "Updated Task Description: Provide a detailed description of a cat, including breed and age.",
                "suggestions": [],
            },
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_empty_suggestions(self, mock_invoke):
        """Test update_description with valid description but empty suggestions."""
        mock_invoke.return_value = '{"description": "Task Description: Describe a cat."}'
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = []
        updated_description = self.generator.update_description(input_str, description, suggestions)
        self.assertEqual(
            updated_description,
            {"description": "Task Description: Describe a cat.", "suggestions": []}
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_single_suggestion(self, mock_invoke):
        """Test update_description with a single suggestion."""
        mock_invoke.return_value = '{"description": "Task Description: Describe a cat with its breed."}'
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = ["Specify cat breed"]
        updated_description = self.generator.update_description(input_str, description, suggestions)
        self.assertEqual(
            updated_description,
            {"description": "Task Description: Describe a cat with its breed.", "suggestions": []}
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_max_suggestions(self, mock_invoke):
        """Test update_description with maximum allowed suggestions."""
        mock_invoke.return_value = (
            '{"description": "Updated Task Description: Provide a comprehensive '
            'description of a cat, including breed, age, color, and health status."}'
        )
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = [
            "Specify cat breed",
            "Include cat age",
            "Add cat color",
            "Detail health status",
            "Mention temperament"
        ]
        updated_description = self.generator.update_description(
            input_str, description, suggestions
        )
        self.assertEqual(
            updated_description,
            {
                "description": "Updated Task Description: Provide a comprehensive "
                               "description of a cat, including breed, age, color, "
                               "and health status.",
                "suggestions": []
            }
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_invalid_suggestions_format(self, mock_invoke):
        """Test update_description with suggestions in an incorrect format."""
        mock_invoke.side_effect = ValueError("Invalid suggestions format.")
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = {"suggestion": "Specify cat breed"}  # Incorrect format, should be a list
        updated_description = self.generator.update_description(input_str, description, suggestions)
        # No exception raised, but the suggestions are not applied
        # TODO: Consider raising an exception here
        self.assertEqual(
            updated_description,
            {"description": "", "suggestions": []}
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_non_json_suggestions(self, mock_invoke):
        """Test update_description with suggestions as a non-JSON string."""
        mock_invoke.side_effect = ValueError("Suggestions must be a JSON array.")
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = "Specify cat breed"  # Should be a list
        updated_description = self.generator.update_description(input_str, description, suggestions)
        # No exception raised, but the suggestions are not applied
        # TODO: Consider raising an exception here
        self.assertEqual(
            updated_description,
            {"description": "", "suggestions": []}
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_malformed_json_suggestions(self, mock_invoke):
        """Test update_description with malformed JSON in suggestions."""
        mock_invoke.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = ["Specify cat breed", "Include cat age"]
        updated_description = self.generator.update_description(input_str, description, suggestions)
        # No exception raised, but the suggestions are not applied
        # TODO: Consider raising an exception here
        self.assertEqual(
            updated_description,
            {"description": "", "suggestions": []}
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_missing_description(self, mock_invoke):
        """Test update_description with missing description parameter."""
        mock_invoke.side_effect = ValueError("Description is required.")
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = ""  # Missing description
        suggestions = ["Specify cat breed"]
        updated_description = self.generator.update_description(input_str, description, suggestions)
        # No exception raised, but the suggestions are not applied
        # TODO: Consider raising an exception here
        self.assertEqual(
            updated_description,
            {"description": "", "suggestions": []}
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_api_failure(self, mock_invoke):
        """Simulate an API failure during description update."""
        mock_response = Mock()
        mock_response.request = Mock()
        mock_invoke.side_effect = BadRequestError(
            "API Error", response=mock_response, body=None
        )
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = ["Specify cat breed", "Include cat age"]
        updated_description = self.generator.update_description(input_str, description, suggestions)
        self.assertEqual(
            updated_description,
            {"description": "", "suggestions": []}  # Fallback description
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_partial_failure_suggestions(self, mock_invoke):
        """Simulate partial failure in applying suggestions."""
        # First suggestion succeeds, second fails
        mock_response = Mock()
        mock_response.request = Mock()
        mock_invoke.side_effect = [
            '{"description": "Updated Task Description: Describe a domestic cat."}',
            BadRequestError("API Error", response=mock_response, body=None)
        ]
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat."
        suggestions = ["Describe habitat", "Include dietary habits"]
        updated_description = self.generator.update_description(input_str, description, suggestions)
        self.assertEqual(
            updated_description,
            {"description": "Updated Task Description: Describe a domestic cat.", "suggestions": []}
        )

    @patch.object(ChatOpenAI, "invoke")
    def test_update_description_large_input(self, mock_invoke):
        """Test update_description with a large description and extensive suggestions."""
        mock_invoke.return_value = (
            '{"description": "Updated Task Description: Provide an in-depth description of a cat, '
            'covering breed, age, color, health status, behavior, and habitat."}'
        )
        input_str = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = "Task Description: Describe a cat in detail."
        suggestions = [
            "Specify cat breed",
            "Include cat age",
            "Add cat color",
            "Detail health status",
            "Describe behavior",
            "Mention habitat",
            "Include dietary habits",
            "Add grooming requirements",
            "Specify activity level",
            "Describe common health issues"
        ]
        updated_description = self.generator.update_description(input_str, description, suggestions)
        self.assertEqual(
            updated_description,
            {
                "description": "Updated Task Description: Provide an in-depth description of a cat, "
                               "covering breed, age, color, health status, behavior, and habitat.",
                "suggestions": []
            }
        )


class TestTaskDescriptionGeneratorProcess(unittest.TestCase):
    """Test the direct process method of TaskDescriptionGenerator."""
    
    def setUp(self):
        self.model = get_test_llm()
        self.generator = TaskDescriptionGenerator(self.model)
    
    def test_process_method_direct_call(self):
        """Test the process method calls chain.invoke correctly"""
        with patch.object(self.generator, 'chain') as mock_chain:
            mock_chain.invoke.return_value = {
                "description": "Generated task description",
                "suggestions": ["suggestion1", "suggestion2"]
            }
            
            input_str = '{"input": "test input", "output": "test output"}'
            generating_batch_size = 5
            
            result = self.generator.process(input_str, generating_batch_size)
            
            expected_input = {
                "input_str": input_str,
                "generating_batch_size": generating_batch_size
            }
            mock_chain.invoke.assert_called_once_with(expected_input)
            self.assertEqual(result["description"], "Generated task description")
            self.assertEqual(result["suggestions"], ["suggestion1", "suggestion2"])
    
    def test_process_method_with_default_batch_size(self):
        """Test process method with default generating_batch_size"""
        with patch.object(self.generator, 'chain') as mock_chain:
            mock_chain.invoke.return_value = {"description": "test", "suggestions": []}
            
            input_str = '{"input": "test", "output": "result"}'
            
            result = self.generator.process(input_str)
            
            expected_input = {
                "input_str": input_str,
                "generating_batch_size": 3  # default value
            }
            mock_chain.invoke.assert_called_once_with(expected_input)


if __name__ == '__main__':
    unittest.main()