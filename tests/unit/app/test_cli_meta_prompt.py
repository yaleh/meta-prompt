"""Unit tests for CLI Meta Prompt interface."""

import json
import tempfile
import unittest
import yaml
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from argparse import Namespace

from app.cli_meta_prompt import MetaPromptCLI, create_parser, main
from app.config import MetaPromptConfig
from meta_prompt import Example


class TestMetaPromptCLI(unittest.TestCase):
    """Test MetaPromptCLI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yml"
        
        # Create test config
        test_config = {
            'llms': {
                'test_model': {
                    'type': 'ChatOpenAI',
                    'model_name': 'gpt-3.5-turbo',
                    'temperature': 0.1,
                    'api_key': 'test-key'
                }
            },
            'examples_path': str(self.temp_dir),
            'max_output_age': 2,
            'recursion_limit': 10
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def test_init_with_config_file(self):
        """Test CLI initialization with config file."""
        cli = MetaPromptCLI(config_file=str(self.config_file))
        self.assertIsInstance(cli.config, MetaPromptConfig)
        self.assertEqual(cli.config_file, str(self.config_file))
    
    def test_load_examples_from_json(self):
        """Test loading examples from JSON file."""
        examples_data = [
            {"user_message": "Hello", "expected_output": "Hi there"},
            {"input": "Goodbye", "output": "See you later"}
        ]
        
        examples_file = Path(self.temp_dir) / "examples.json"
        with open(examples_file, 'w') as f:
            json.dump(examples_data, f)
        
        cli = MetaPromptCLI(config_file=str(self.config_file))
        examples = cli._load_examples_from_file(str(examples_file))
        
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]['user_message'], "Hello")
        self.assertEqual(examples[0]['expected_output'], "Hi there")
        self.assertEqual(examples[1]['user_message'], "Goodbye")
        self.assertEqual(examples[1]['expected_output'], "See you later")
    
    def test_load_examples_from_yaml(self):
        """Test loading examples from YAML file."""
        examples_data = {
            'examples': [
                {"user_message": "Test question", "expected_output": "Test answer"}
            ]
        }
        
        examples_file = Path(self.temp_dir) / "examples.yaml"
        with open(examples_file, 'w') as f:
            yaml.dump(examples_data, f)
        
        cli = MetaPromptCLI(config_file=str(self.config_file))
        examples = cli._load_examples_from_file(str(examples_file))
        
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]['user_message'], "Test question")
        self.assertEqual(examples[0]['expected_output'], "Test answer")
    
    def test_load_examples_file_not_found(self):
        """Test loading examples with non-existent file."""
        cli = MetaPromptCLI(config_file=str(self.config_file))
        
        with self.assertRaises(FileNotFoundError):
            cli._load_examples_from_file("nonexistent.json")
    
    def test_load_examples_invalid_format(self):
        """Test loading examples with invalid format."""
        invalid_data = {"not_examples": "invalid"}
        
        examples_file = Path(self.temp_dir) / "invalid.json"
        with open(examples_file, 'w') as f:
            json.dump(invalid_data, f)
        
        cli = MetaPromptCLI(config_file=str(self.config_file))
        
        with self.assertRaises(ValueError):
            cli._load_examples_from_file(str(examples_file))
    
    @patch('app.gradio_meta_prompt_utils.initialize_llm')
    @patch('app.cli_meta_prompt.MetaPromptGraph')
    def test_create_llm_graph(self, mock_graph_class, mock_init_llm):
        """Test creating LLM graph."""
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm
        mock_graph = Mock()
        mock_graph_class.return_value = mock_graph
        
        cli = MetaPromptCLI(config_file=str(self.config_file))
        result = cli._create_llm_graph("test_model")
        
        mock_init_llm.assert_called_once()
        mock_graph_class.assert_called_once()
        self.assertEqual(result, mock_graph)
    
    def test_create_llm_graph_no_models(self):
        """Test creating LLM graph with no models configured."""
        # Create config with empty LLMs dict
        empty_llms_config_file = Path(self.temp_dir) / "empty_llms_config.yml"
        with open(empty_llms_config_file, 'w') as f:
            yaml.dump({'examples_path': str(self.temp_dir), 'llms': {}}, f)
        
        cli = MetaPromptCLI(config_file=str(empty_llms_config_file))
        
        with self.assertRaises(ValueError) as context:
            cli._create_llm_graph()
        
        self.assertIn("No LLM configurations found", str(context.exception))
    
    def test_create_llm_graph_invalid_model(self):
        """Test creating LLM graph with invalid model name."""
        cli = MetaPromptCLI(config_file=str(self.config_file))
        
        with self.assertRaises(ValueError) as context:
            cli._create_llm_graph("nonexistent_model")
        
        self.assertIn("Model 'nonexistent_model' not found", str(context.exception))
    
    @patch('app.cli_meta_prompt.MetaPromptCLI._create_llm_graph')
    def test_generate_prompt(self, mock_create_graph):
        """Test prompt generation."""
        # Mock the graph
        mock_graph = Mock()
        mock_result = {
            'accepted': True,
            'best_system_message': 'Test system message',
            'best_output': 'Test output',
            'best_output_age': 0
        }
        mock_graph.run_meta_prompt_graph.return_value = mock_result
        mock_create_graph.return_value = mock_graph
        
        cli = MetaPromptCLI(config_file=str(self.config_file))
        
        examples = [Example(user_message="Test", expected_output="Result")]
        result = cli.generate_prompt(examples, acceptance_criteria="Test criteria")
        
        self.assertEqual(result, mock_result)
        mock_graph.run_meta_prompt_graph.assert_called_once()
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_print_results_basic(self, mock_stdout):
        """Test printing results with basic output."""
        result = {
            'accepted': True,
            'best_system_message': 'Test system message',
            'best_output': 'Test output',
            'best_output_age': 1
        }
        
        cli = MetaPromptCLI(config_file=str(self.config_file))
        cli.print_results(result)
        
        output = mock_stdout.getvalue()
        self.assertIn("✅ Status: Accepted", output)
        self.assertIn("🔄 Output Age: 1", output)
        self.assertIn("📊 Iterations: 2", output)
        self.assertIn("Test system message", output)
        self.assertIn("Test output", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_print_results_verbose(self, mock_stdout):
        """Test printing results with verbose output."""
        result = {
            'accepted': False,
            'best_system_message': 'Test system message',
            'analysis': '{"Accept": "No", "Reasons": ["Test reason"]}'
        }
        
        cli = MetaPromptCLI(config_file=str(self.config_file))
        cli.print_results(result, verbose=True)
        
        output = mock_stdout.getvalue()
        self.assertIn("⚠️ Status: Not Accepted", output)
        self.assertIn("📈 Analysis:", output)
        self.assertIn("Test reason", output)


class TestCreateParser(unittest.TestCase):
    """Test argument parser creation."""
    
    def test_create_parser_basic(self):
        """Test basic parser creation."""
        parser = create_parser()
        self.assertIsNotNone(parser)
        
        # Test with minimal arguments
        args = parser.parse_args(['test.json'])
        self.assertEqual(args.examples_file, 'test.json')
        self.assertEqual(args.config, 'config.yml')
        self.assertFalse(args.verbose)
        self.assertFalse(args.quiet)
    
    def test_create_parser_all_args(self):
        """Test parser with all arguments."""
        parser = create_parser()
        
        args = parser.parse_args([
            'examples.json',
            '--config', 'custom.yml',
            '--model', 'test-model',
            '--criteria', 'Test criteria',
            '--max-output-age', '5',
            '--recursion-limit', '20',
            '--output', 'result.json',
            '--verbose'
        ])
        
        self.assertEqual(args.examples_file, 'examples.json')
        self.assertEqual(args.config, 'custom.yml')
        self.assertEqual(args.model, 'test-model')
        self.assertEqual(args.criteria, 'Test criteria')
        self.assertEqual(args.max_output_age, 5)
        self.assertEqual(args.recursion_limit, 20)
        self.assertEqual(args.output, 'result.json')
        self.assertTrue(args.verbose)
    
    def test_create_parser_list_models(self):
        """Test parser with list models flag."""
        parser = create_parser()
        
        args = parser.parse_args(['--list-models'])
        self.assertTrue(args.list_models)
        self.assertIsNone(args.examples_file)


class TestMainFunction(unittest.TestCase):
    """Test main CLI function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yml"
        
        # Create test config
        test_config = {
            'llms': {
                'test_model': {
                    'type': 'ChatOpenAI',
                    'model_name': 'gpt-3.5-turbo',
                    'temperature': 0.1,
                    'api_key': 'test-key'
                }
            },
            'examples_path': str(self.temp_dir)
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create test examples file
        self.examples_file = Path(self.temp_dir) / "examples.json"
        examples_data = [
            {"user_message": "Test", "expected_output": "Result"}
        ]
        with open(self.examples_file, 'w') as f:
            json.dump(examples_data, f)
    
    @patch('sys.argv')
    @patch('sys.stdout', new_callable=StringIO)
    def test_main_list_models(self, mock_stdout, mock_argv):
        """Test main function with list models."""
        mock_argv.__getitem__.side_effect = [
            'cli_meta_prompt.py',
            '--list-models',
            '--config', str(self.config_file)
        ]
        mock_argv.__len__.return_value = 4
        
        with patch('app.cli_meta_prompt.create_parser') as mock_parser:
            parser = create_parser()
            mock_parser.return_value = parser
            
            # Mock parse_args to return our desired arguments
            with patch.object(parser, 'parse_args') as mock_parse:
                mock_parse.return_value = Namespace(
                    list_models=True,
                    config=str(self.config_file),
                    examples_file=None,
                    verbose=False
                )
                
                result = main()
                
                self.assertEqual(result, 0)
                output = mock_stdout.getvalue()
                self.assertIn("Available models:", output)
                self.assertIn("test_model", output)
    
    @patch('sys.argv')
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_missing_examples_file(self, mock_stderr, mock_argv):
        """Test main function with missing examples file."""
        mock_argv.__getitem__.side_effect = [
            'cli_meta_prompt.py',
            '--config', str(self.config_file)
        ]
        mock_argv.__len__.return_value = 3
        
        with patch('app.cli_meta_prompt.create_parser') as mock_parser:
            parser = create_parser()
            mock_parser.return_value = parser
            
            with patch.object(parser, 'parse_args') as mock_parse:
                mock_parse.return_value = Namespace(
                    list_models=False,
                    config=str(self.config_file),
                    examples_file=None,
                    verbose=False
                )
                
                with patch.object(parser, 'print_help'):
                    result = main()
                    
                    self.assertEqual(result, 1)
                    output = mock_stderr.getvalue()
                    self.assertIn("examples_file is required", output)
    
    @patch('sys.argv')
    @patch('app.cli_meta_prompt.MetaPromptCLI')
    def test_main_successful_generation(self, mock_cli_class, mock_argv):
        """Test main function with successful prompt generation."""
        mock_argv.__getitem__.side_effect = [
            'cli_meta_prompt.py',
            str(self.examples_file),
            '--config', str(self.config_file),
            '--quiet'
        ]
        mock_argv.__len__.return_value = 5
        
        # Mock CLI instance
        mock_cli = Mock()
        mock_cli._load_examples_from_file.return_value = [
            Example(user_message="Test", expected_output="Result")
        ]
        mock_cli.generate_prompt.return_value = {
            'accepted': True,
            'best_system_message': 'Test message'
        }
        mock_cli_class.return_value = mock_cli
        
        with patch('app.cli_meta_prompt.create_parser') as mock_parser:
            parser = create_parser()
            mock_parser.return_value = parser
            
            with patch.object(parser, 'parse_args') as mock_parse:
                mock_parse.return_value = Namespace(
                    list_models=False,
                    config=str(self.config_file),
                    examples_file=str(self.examples_file),
                    model=None,
                    criteria=None,
                    max_output_age=None,
                    recursion_limit=None,
                    output=None,
                    verbose=False,
                    quiet=True
                )
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_cli._load_examples_from_file.assert_called_once()
                mock_cli.generate_prompt.assert_called_once()
    
    @patch('sys.argv')
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_exception_handling(self, mock_stderr, mock_argv):
        """Test main function exception handling."""
        mock_argv.__getitem__.side_effect = [
            'cli_meta_prompt.py',
            'nonexistent.json'
        ]
        mock_argv.__len__.return_value = 2
        
        with patch('app.cli_meta_prompt.create_parser') as mock_parser:
            parser = create_parser()
            mock_parser.return_value = parser
            
            with patch.object(parser, 'parse_args') as mock_parse:
                mock_parse.return_value = Namespace(
                    list_models=False,
                    config='config.yml',
                    examples_file='nonexistent.json',
                    verbose=False
                )
                
                result = main()
                
                self.assertEqual(result, 1)
                output = mock_stderr.getvalue()
                self.assertIn("Error:", output)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI functionality."""
    
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
                    'api_key': 'test-key'
                }
            },
            'examples_path': str(self.temp_dir),
            'max_output_age': 1,
            'recursion_limit': 5
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create test examples
        self.examples_file = Path(self.temp_dir) / "examples.json"
        examples_data = [
            {"user_message": "Hello", "expected_output": "Hi there"}
        ]
        with open(self.examples_file, 'w') as f:
            json.dump(examples_data, f)
    
    @patch('app.cli_meta_prompt.MetaPromptGraph')
    @patch('app.gradio_meta_prompt_utils.initialize_llm')
    def test_end_to_end_workflow(self, mock_init_llm, mock_graph_class):
        """Test complete CLI workflow."""
        # Mock LLM and graph
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm
        
        mock_graph = Mock()
        mock_result = {
            'accepted': True,
            'best_system_message': 'Generated system message',
            'best_output': 'Generated output',
            'best_output_age': 0
        }
        mock_graph.run_meta_prompt_graph.return_value = mock_result
        mock_graph_class.return_value = mock_graph
        
        # Test CLI workflow
        cli = MetaPromptCLI(config_file=str(self.config_file))
        examples = cli._load_examples_from_file(str(self.examples_file))
        result = cli.generate_prompt(examples)
        
        # Verify results
        self.assertTrue(result['accepted'])
        self.assertIn('best_system_message', result)
        self.assertIn('best_output', result)
        
        # Verify LLM was initialized and graph was called
        mock_init_llm.assert_called_once()
        mock_graph.run_meta_prompt_graph.assert_called_once()


if __name__ == '__main__':
    unittest.main()