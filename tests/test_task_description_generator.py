import json
import unittest
from unittest.mock import MagicMock, patch
from langchain_openai import ChatOpenAI
from meta_prompt.sample_generator import TaskDescriptionGenerator

class TestTaskDescriptionGenerator(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()