"""Integration tests for CLI Meta Prompt."""

import json
import tempfile
import unittest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from app.cli_meta_prompt import MetaPromptCLI
from tests.unit.utils.test_config_utils import skip_if_no_api_key, get_test_llm


class TestCLIIntegrationWithMocks(unittest.TestCase):
    """Integration tests using mocked LLM responses."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config
        self.config_file = Path(self.temp_dir) / "test_config.yml"
        test_config = {
            'llms': {
                'test_model': {
                    'type': 'ChatOpenAI',
                    'model_name': 'gpt-3.5-turbo',
                    'temperature': 0.1,
                    'openai_api_key': 'test-key',
                    'openai_api_base': 'https://api.openai.com/v1'
                }
            },
            'examples_path': str(self.temp_dir),
            'max_output_age': 2,
            'recursion_limit': 10
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def create_examples_file(self, examples_data, filename="examples.json"):
        """Helper to create examples file."""
        examples_file = Path(self.temp_dir) / filename
        
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(examples_file, 'w') as f:
                yaml.dump(examples_data, f)
        else:
            with open(examples_file, 'w') as f:
                json.dump(examples_data, f)
        
        return str(examples_file)
    
    @patch('app.cli_meta_prompt.MetaPromptGraph')
    @patch('app.gradio_meta_prompt_utils.initialize_llm')
    def test_simple_generation_workflow(self, mock_init_llm, mock_graph_class):
        """Test simple prompt generation workflow."""
        # Setup mocks
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm
        
        mock_graph = Mock()
        mock_result = {
            'accepted': True,
            'best_system_message': 'You are a helpful assistant that provides clear and concise answers.',
            'best_output': 'The capital of France is Paris.',
            'best_output_age': 0,
            'analysis': '{"Accept": "Yes", "Acceptable Differences": [], "Unacceptable Differences": []}'
        }
        mock_graph.run_meta_prompt_graph.return_value = mock_result
        mock_graph_class.return_value = mock_graph
        
        # Create test examples
        examples_data = [
            {"user_message": "What is the capital of France?", "expected_output": "The capital of France is Paris."},
            {"user_message": "What is the capital of Germany?", "expected_output": "The capital of Germany is Berlin."}
        ]
        examples_file = self.create_examples_file(examples_data)
        
        # Test CLI
        cli = MetaPromptCLI(config_file=str(self.config_file))
        examples = cli._load_examples_from_file(examples_file)
        result = cli.generate_prompt(examples, acceptance_criteria="Should provide accurate geographical information")
        
        # Verify
        self.assertTrue(result['accepted'])
        self.assertIsNotNone(result['best_system_message'])
        self.assertIsNotNone(result['best_output'])
        
        # Verify graph was called with correct parameters
        mock_graph.run_meta_prompt_graph.assert_called_once()
        call_args = mock_graph.run_meta_prompt_graph.call_args[0][0]
        self.assertEqual(len(call_args['examples']), 2)
        self.assertIn("geographical information", call_args['acceptance_criteria'])
    
    @patch('app.cli_meta_prompt.MetaPromptGraph')
    @patch('app.gradio_meta_prompt_utils.initialize_llm')
    def test_yaml_examples_workflow(self, mock_init_llm, mock_graph_class):
        """Test workflow with YAML examples file."""
        # Setup mocks
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm
        
        mock_graph = Mock()
        mock_result = {
            'accepted': True,
            'best_system_message': 'You are a programming tutor.',
            'best_output': 'A loop is a control structure that repeats code.',
            'best_output_age': 1
        }
        mock_graph.run_meta_prompt_graph.return_value = mock_result
        mock_graph_class.return_value = mock_graph
        
        # Create YAML examples
        examples_data = {
            'examples': [
                {"user_message": "What is a loop?", "expected_output": "A loop is a control structure that repeats code."},
                {"user_message": "What is a variable?", "expected_output": "A variable is a storage location with a name."}
            ]
        }
        examples_file = self.create_examples_file(examples_data, "examples.yaml")
        
        # Test CLI
        cli = MetaPromptCLI(config_file=str(self.config_file))
        examples = cli._load_examples_from_file(examples_file)
        result = cli.generate_prompt(examples, max_output_age=3)
        
        # Verify
        self.assertTrue(result['accepted'])
        self.assertEqual(len(examples), 2)
        
        # Verify state was created correctly
        call_args = mock_graph.run_meta_prompt_graph.call_args[0][0]
        self.assertEqual(call_args['max_output_age'], 3)
    
    @patch('app.cli_meta_prompt.MetaPromptGraph')
    @patch('app.gradio_meta_prompt_utils.initialize_llm')
    def test_multiple_iterations_workflow(self, mock_init_llm, mock_graph_class):
        """Test workflow that requires multiple iterations."""
        # Setup mocks
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm
        
        mock_graph = Mock()
        mock_result = {
            'accepted': False,  # Not accepted initially
            'best_system_message': 'Initial system message',
            'best_output': 'Initial output',
            'best_output_age': 2,  # Went through multiple iterations
            'analysis': '{"Accept": "No", "Unacceptable Differences": ["Format not quite right"]}'
        }
        mock_graph.run_meta_prompt_graph.return_value = mock_result
        mock_graph_class.return_value = mock_graph
        
        # Create examples
        examples_data = [
            {"user_message": "Explain JSON format", "expected_output": '{"key": "value", "array": [1, 2, 3]}'}
        ]
        examples_file = self.create_examples_file(examples_data)
        
        # Test CLI
        cli = MetaPromptCLI(config_file=str(self.config_file))
        examples = cli._load_examples_from_file(examples_file)
        result = cli.generate_prompt(
            examples, 
            acceptance_criteria="Response must include valid JSON syntax",
            max_output_age=5,
            recursion_limit=15
        )
        
        # Verify
        self.assertFalse(result['accepted'])  # Didn't reach acceptance
        self.assertEqual(result['best_output_age'], 2)  # But tried multiple iterations
        
        # Verify parameters were passed correctly
        call_args = mock_graph.run_meta_prompt_graph.call_args
        state = call_args[0][0]
        recursion_limit = call_args[1]['recursion_limit']
        
        self.assertEqual(state['max_output_age'], 5)
        self.assertEqual(recursion_limit, 15)
        self.assertIn("JSON syntax", state['acceptance_criteria'])
    
    def test_invalid_examples_file_handling(self):
        """Test handling of invalid examples files."""
        # Test non-existent file
        cli = MetaPromptCLI(config_file=str(self.config_file))
        
        with self.assertRaises(FileNotFoundError):
            cli._load_examples_from_file("nonexistent.json")
        
        # Test invalid JSON
        invalid_file = Path(self.temp_dir) / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content {")
        
        with self.assertRaises(json.JSONDecodeError):
            cli._load_examples_from_file(str(invalid_file))
        
        # Test invalid structure
        invalid_data = {"wrong_key": "wrong_value"}
        invalid_structure_file = self.create_examples_file(invalid_data)
        
        with self.assertRaises(ValueError):
            cli._load_examples_from_file(invalid_structure_file)
    
    def test_model_selection_workflow(self):
        """Test model selection and validation."""
        # Add another model to config
        test_config = {
            'llms': {
                'model1': {'type': 'ChatOpenAI', 'model_name': 'gpt-3.5-turbo'},
                'model2': {'type': 'ChatOpenAI', 'model_name': 'gpt-4'}
            },
            'examples_path': str(self.temp_dir)
        }
        
        config_file = Path(self.temp_dir) / "multi_model_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        cli = MetaPromptCLI(config_file=str(config_file))
        
        # Test invalid model selection
        with self.assertRaises(ValueError) as context:
            cli._create_llm_graph("nonexistent_model")
        
        self.assertIn("Model 'nonexistent_model' not found", str(context.exception))
        self.assertIn("model1", str(context.exception))
        self.assertIn("model2", str(context.exception))
    
    def test_configuration_edge_cases(self):
        """Test edge cases in configuration handling."""
        # Test config with empty LLMs dict
        empty_llms_config = {"examples_path": str(self.temp_dir), "llms": {}}
        empty_llms_file = Path(self.temp_dir) / "empty_llms_config.yml"
        with open(empty_llms_file, 'w') as f:
            yaml.dump(empty_llms_config, f)
        
        cli = MetaPromptCLI(config_file=str(empty_llms_file))
        
        with self.assertRaises(ValueError) as context:
            cli._create_llm_graph()
        
        self.assertIn("No LLM configurations found", str(context.exception))


@skip_if_no_api_key
class TestCLIWithRealLLM(unittest.TestCase):
    """Integration tests with real LLM (requires API key)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_real_llm_simple_workflow(self):
        """Test CLI with actual LLM API call."""
        # Create config using test LLM
        test_config = {
            'llms': {
                'test_model': {
                    'type': 'ChatOpenAI',
                    'model_name': 'gpt-3.5-turbo',
                    'temperature': 0.1,
                    'openai_api_key': 'test-key',
                    'openai_api_base': 'https://api.openai.com/v1'
                }
            },
            'max_output_age': 1,  # Keep it short for testing
            'recursion_limit': 5
        }
        
        config_file = Path(self.temp_dir) / "real_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create simple examples
        examples_data = [
            {"user_message": "What is 2+2?", "expected_output": "2+2 equals 4."}
        ]
        
        examples_file = Path(self.temp_dir) / "examples.json"
        with open(examples_file, 'w') as f:
            json.dump(examples_data, f)
        
        # Patch to use test LLM
        with patch('app.gradio_meta_prompt_utils.initialize_llm') as mock_init:
            mock_init.return_value = get_test_llm()
            
            cli = MetaPromptCLI(config_file=str(config_file))
            examples = cli._load_examples_from_file(str(examples_file))
            result = cli.generate_prompt(
                examples, 
                acceptance_criteria="Response should provide a clear arithmetic answer"
            )
            
            # Basic verification
            self.assertIsInstance(result, dict)
            self.assertIn('best_system_message', result)
            self.assertIn('accepted', result)
            
            # The system message should be generated
            self.assertIsNotNone(result.get('best_system_message'))
            self.assertTrue(len(result.get('best_system_message', '')) > 0)


if __name__ == '__main__':
    unittest.main()