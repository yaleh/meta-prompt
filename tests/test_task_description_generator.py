import json
import unittest
from unittest.mock import MagicMock, patch
from langchain_openai import ChatOpenAI
from meta_prompt.sample_generator import TaskDescriptionGenerator

class TestTaskDescriptionGeneratorBasic(unittest.TestCase):

    def setUp(self):
        self.model = ChatOpenAI(model="llama3-70b-8192", temperature=1.0, max_retries=3)
        self.generator = TaskDescriptionGenerator(self.model)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_description(self, mock_invoke):
        mock_invoke.return_value = '{"description": "Task Description: Describe a cat."}'
        input_json = json.dumps({"input": "A cat", "output": "A furry animal"})
        description = self.generator.generate_description(input_json)
        self.assertEqual(description, {'description': 'Task Description: Describe a cat.', 'suggestions': []})

    @patch.object(ChatOpenAI, "invoke")
    def test_analyze_input(self, mock_invoke):
        mock_invoke.return_value = "Input Analysis: The input is an animal."
        description = "Task Description: Describe a cat."
        input_analysis = self.generator.analyze_input(description)
        self.assertEqual(input_analysis, "Input Analysis: The input is an animal.")

class TestTaskDescriptionGeneratorExamples(unittest.TestCase):

    def setUp(self):
        self.model = ChatOpenAI(model="llama3-70b-8192", temperature=1.0, max_retries=3)
        self.generator = TaskDescriptionGenerator(self.model)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_briefs(self, mock_invoke):
        mock_invoke.return_value = '{"new_example_briefs": [{"example_brief": "Brief 1"}, {"example_brief": "Brief 2"}]}'
        description = "Task Description: Describe a cat."
        input_analysis = "Input Analysis: The input is an animal."
        generating_batch_size = 2
        briefs = self.generator.generate_briefs(description, input_analysis, generating_batch_size)
        self.assertEqual(briefs, [{"example_brief": "Brief 1"}, {"example_brief": "Brief 2"}])

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_examples_from_briefs(self, mock_invoke):
        mock_invoke.return_value = '{"examples": [{"input": "Input 1", "output": "Output 1"}, {"input": "Input 2", "output": "Output 2"}]}'
        description = "Task Description: Describe a cat."
        new_example_briefs = {"new_example_briefs": [{"example_brief": "Brief 1"}, {"example_brief": "Brief 2"}]}
        raw_example = json.dumps({"input": "A cat", "output": "A furry animal"})
        generating_batch_size = 2
        examples = self.generator.generate_examples_from_briefs(description, new_example_briefs, raw_example, generating_batch_size)
        self.assertEqual(examples, {"examples": [{"input": "Input 1", "output": "Output 1"}, {"input": "Input 2", "output": "Output 2"}]})

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_examples(self, mock_invoke):
        mock_invoke.return_value = '{"examples": [{"input": "Input 1", "output": "Output 1"}, {"input": "Input 2", "output": "Output 2"}]}'
        description = "Task Description: Describe a cat."
        raw_example = json.dumps({"input": "A cat", "output": "A furry animal"})
        generating_batch_size = 2
        examples = self.generator.generate_examples_directly(description, raw_example, generating_batch_size)
        self.assertEqual(examples, {"examples": [{"input": "Input 1", "output": "Output 1"}, {"input": "Input 2", "output": "Output 2"}]})

class TestTaskDescriptionGeneratorSuggestions(unittest.TestCase):

    def setUp(self):
        self.model = ChatOpenAI(model="llama3-70b-8192", temperature=1.0, max_retries=3)
        self.generator = TaskDescriptionGenerator(self.model)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_basic(self, mock_invoke):
        mock_invoke.side_effect = [
            '{"suggestions": [{"suggestion": "Specify cat breed"}, {"suggestion": "Include cat age"}]}',
            '{"suggestions": [{"suggestion": "Expand to all pets"}, {"suggestion": "Include habitat description"}]}'
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
            '{"suggestions": [{"suggestion": "Summarize key points"}, {"suggestion": "Extract main themes"}]}',
            '{"suggestions": [{"suggestion": "Expand analysis scope"}, {"suggestion": "Include cross-references"}]}'
        ]
        input_str = json.dumps({"input": "A" * 1000, "output": "B" * 1000})
        description = "Task Description: Analyze a long text."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        self.assertEqual(len(result['suggestions']), 4)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_complex_task(self, mock_invoke):
        mock_invoke.side_effect = [
            '{"suggestions": [{"suggestion": "Break down into subtasks"}, {"suggestion": "Specify input formats for each step"}]}',
            '{"suggestions": [{"suggestion": "Generalize to similar problem domains"}, {"suggestion": "Include error handling procedures"}]}'
        ]
        input_str = json.dumps({"input": "Complex task input", "output": "Complex task output"})
        description = "Task Description: Perform a complex multi-step analysis."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        self.assertEqual(len(result['suggestions']), 4)

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_error_handling(self, mock_invoke):
        mock_invoke.side_effect = [
            Exception("API Error"),
            '{"suggestions": [{"suggestion": "Handle network errors"}, {"suggestion": "Implement retry logic"}]}'
        ]
        input_str = json.dumps({"input": "Error prone task", "output": "Error handling result"})
        description = "Task Description: Test error handling in a system."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        self.assertEqual(len(result['suggestions']), 2)  # Only generalization suggestions due to simulated error

    @patch.object(ChatOpenAI, "invoke")
    def test_generate_suggestions_format_validation(self, mock_invoke):
        mock_invoke.side_effect = [
            '{"suggestions": [{"suggestion": "Validate input format"}, {"suggestion": "Enforce output structure"}]}',
            '{"suggestions": [{"suggestion": "Allow flexible input formats"}, {"suggestion": "Generate multiple output formats"}]}'
        ]
        input_str = json.dumps({"input": "Unstructured data", "output": "Structured result"})
        description = "Task Description: Convert unstructured data to structured format."
        result = self.generator.generate_suggestions(input_str, description)
        self.assertIn('suggestions', result)
        self.assertEqual(len(result['suggestions']), 4)
        self.assertEqual(sorted(result['suggestions']), sorted(["Validate input format", "Enforce output structure", "Allow flexible input formats", "Generate multiple output formats"]))

if __name__ == '__main__':
    unittest.main()