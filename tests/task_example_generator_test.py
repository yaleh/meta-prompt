import unittest
from unittest.mock import MagicMock
import json
from meta_prompt.sample_generator import TaskDescriptionGenerator

class TestTaskDescriptionGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.generator = TaskDescriptionGenerator(self.mock_model)

    def test_load_and_validate_input_json(self):
        valid_json = '{"input": "test input", "output": "test output"}'
        result = self.generator.load_and_validate_input({"input_str": valid_json, "generating_batch_size": 3})
        self.assertEqual(result, {"example": {"input": "test input", "output": "test output"}, "generating_batch_size": 3})

    def test_load_and_validate_input_yaml(self):
        valid_yaml = 'input: test input\noutput: test output'
        result = self.generator.load_and_validate_input({"input_str": valid_yaml, "generating_batch_size": 3})
        self.assertEqual(result, {"example": {"input": "test input", "output": "test output"}, "generating_batch_size": 3})

    def test_process(self):
        self.mock_model.return_value = "test description"
        result = self.generator.process('{"input": "test input", "output": "test output"}', 3)
        self.assertIn("description", result)

    def test_generate_description(self):
        self.mock_model.return_value = "test description"
        result = self.generator.generate_description('{"input": "test input", "output": "test output"}')
        self.assertEqual(result, "test description")

    def test_analyze_input(self):
        self.mock_model.return_value = "test analysis"
        result = self.generator.analyze_input("test description")
        self.assertEqual(result, "test analysis")

    def test_generate_briefs(self):
        self.mock_model.return_value = "test briefs"
        result = self.generator.generate_briefs("test description", "test analysis", 3)
        self.assertEqual(result, "test briefs")

    def test_generate_examples_from_briefs(self):
        self.mock_model.return_value = {"examples": [{"input": "test input", "output": "test output"}]}
        result = self.generator.generate_examples_from_briefs("test description", "test briefs", '{"input": "test input", "output": "test output"}', 3)
        self.assertEqual(result, {"examples": [{"input": "test input", "output": "test output"}]})

    def test_generate_examples_directly(self):
        self.mock_model.return_value = {"examples": [{"input": "test input", "output": "test output"}]}
        result = self.generator.generate_examples_