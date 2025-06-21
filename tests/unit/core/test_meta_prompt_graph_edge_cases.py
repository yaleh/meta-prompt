"""Unit tests for edge cases and error paths in MetaPromptGraph."""

import unittest
from unittest.mock import Mock, patch
from pydantic import BaseModel
from langchain_core.language_models import BaseLanguageModel
from langgraph.errors import GraphRecursionError

from meta_prompt import MetaPromptGraph, AgentState, Example
from meta_prompt.meta_prompt import first_non_empty, last_non_empty
from tests.unit.utils.test_config_utils import get_test_llm, skip_if_no_api_key


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions in meta_prompt.py"""
    
    def test_first_non_empty_with_values(self):
        """Test first_non_empty returns first non-empty value"""
        result = first_non_empty("first", "second")
        self.assertEqual(result, "first")
        
    def test_first_non_empty_with_first_empty(self):
        """Test first_non_empty returns second when first is empty"""
        result = first_non_empty(None, "second")
        self.assertEqual(result, "second")
        
    def test_first_non_empty_with_both_empty(self):
        """Test first_non_empty returns None when both are empty"""
        result = first_non_empty(None, None)
        self.assertIsNone(result)
        
    def test_first_non_empty_with_empty_strings(self):
        """Test first_non_empty with empty strings"""
        result = first_non_empty("", "second")
        self.assertEqual(result, "second")
        
    def test_last_non_empty_with_values(self):
        """Test last_non_empty returns last non-empty value"""
        result = last_non_empty("first", "second")
        self.assertEqual(result, "second")
        
    def test_last_non_empty_with_second_empty(self):
        """Test last_non_empty returns first when second is empty"""
        result = last_non_empty("first", None)
        self.assertEqual(result, "first")
        
    def test_last_non_empty_with_both_empty(self):
        """Test last_non_empty returns None when both are empty"""
        result = last_non_empty(None, None)
        self.assertIsNone(result)


class TestStateConversion(unittest.TestCase):
    """Test state conversion and validation in MetaPromptGraph."""
    
    def setUp(self):
        mock_llm = Mock(spec=BaseLanguageModel)
        mock_llm.config_specs = []
        self.graph = MetaPromptGraph(llms=mock_llm)
    
    def test_convert_state_with_dict(self):
        """Test state conversion with dictionary input"""
        state = {
            "examples": [{"user_message": "test", "expected_output": "result"}],
            "current_example_index": 0,
            "max_output_age": 2
        }
        result = AgentState.to_dict(state)
        self.assertIsInstance(result, dict)
        # Examples are processed and removed, current example data is extracted
        self.assertNotIn("examples", result)
        self.assertNotIn("current_example_index", result)
        self.assertEqual(result["user_message"], "test")
        self.assertEqual(result["expected_output"], "result")
        self.assertEqual(result["max_output_age"], 2)
        
    def test_convert_state_with_basemodel(self):
        """Test state conversion with BaseModel input"""
        class TestModel(BaseModel):
            examples: list
            max_output_age: int
            best_output_age: int
            other_field: str
            current_example_index: int = 0
            
        model_state = TestModel(
            examples=[{"user_message": "test", "expected_output": "result"}],
            max_output_age=2,
            best_output_age=1,
            other_field="test",
            current_example_index=0
        )
        
        result = AgentState.to_dict(model_state)
        self.assertIsInstance(result, dict)
        # max_output_age and best_output_age are removed from BaseModel
        self.assertNotIn("max_output_age", result)
        self.assertNotIn("best_output_age", result)
        # examples and current_example_index are processed and removed
        self.assertNotIn("examples", result)
        self.assertNotIn("current_example_index", result)
        # Current example data is extracted
        self.assertEqual(result["user_message"], "test")
        self.assertEqual(result["expected_output"], "result")
        self.assertIn("other_field", result)
        self.assertEqual(result["other_field"], "test")
        
    def test_convert_state_with_invalid_type(self):
        """Test state conversion raises TypeError for invalid input"""
        with self.assertRaises(TypeError) as context:
            AgentState.to_dict("invalid_state")
        
        self.assertIn("State must be either a TypedDict or a BaseModel instance", 
                     str(context.exception))


class TestMaxOutputAgeConditions(unittest.TestCase):
    """Test max_output_age edge cases and conditions."""
    
    def setUp(self):
        mock_llm = Mock(spec=BaseLanguageModel)
        mock_llm.config_specs = []
        self.graph = MetaPromptGraph(llms=mock_llm)
    
    def test_should_continue_with_zero_max_age(self):
        """Test that workflow continues when max_output_age is 0"""
        state = {
            "max_output_age": 0,
            "best_output_age": 5,
            "accepted": False
        }
        result = self.graph._should_exit_on_max_age(state)
        self.assertEqual(result, "continue")
        
    def test_should_continue_with_negative_max_age(self):
        """Test that workflow continues when max_output_age is negative"""
        state = {
            "max_output_age": -1,
            "best_output_age": 5,
            "accepted": False
        }
        result = self.graph._should_exit_on_max_age(state)
        self.assertEqual(result, "continue")
        
    def test_should_stop_when_max_age_exceeded(self):
        """Test that workflow stops when best_output_age >= max_output_age"""
        state = {
            "max_output_age": 3,
            "best_output_age": 3,
            "accepted": False
        }
        result = self.graph._should_exit_on_max_age(state)
        self.assertEqual(result, "__end__")


class TestRecursionErrorHandling(unittest.TestCase):
    """Test GraphRecursionError handling and checkpoint recovery."""
    
    @skip_if_no_api_key
    def test_recursion_error_with_checkpoints(self):
        """Test handling of GraphRecursionError with available checkpoints"""
        llms = {"test": get_test_llm()}
        graph = MetaPromptGraph(llms=llms)
        
        # Create a state that will trigger recursion
        input_state = AgentState(
            examples=[Example(
                user_message="test message",
                expected_output="test output"
            )],
            acceptance_criteria="test criteria",
            max_output_age=1  # Low age to avoid long execution
        )
        
        # Mock the graph to raise recursion error but have checkpoints
        with patch.object(graph.graph, 'invoke') as mock_invoke, \
             patch.object(graph.graph, 'get_state') as mock_get_state:
            
            mock_invoke.side_effect = GraphRecursionError("Recursion limit reached")
            mock_checkpoint_state = {
                "best_system_message": "recovered message",
                "best_output": "recovered output",
                "accepted": True
            }
            mock_get_state.return_value = [mock_checkpoint_state]
            
            result = graph.run_meta_prompt_graph(input_state)
            
            self.assertEqual(result, mock_checkpoint_state)
            mock_get_state.assert_called_once()
    
    @skip_if_no_api_key        
    def test_recursion_error_without_checkpoints(self):
        """Test handling of GraphRecursionError without checkpoints"""
        llms = {"test": get_test_llm()}
        graph = MetaPromptGraph(llms=llms)
        
        input_state = AgentState(
            examples=[Example(
                user_message="test message", 
                expected_output="test output"
            )],
            acceptance_criteria="test criteria",
            max_output_age=1
        )
        
        # Mock the graph to raise recursion error with no checkpoints
        with patch.object(graph.graph, 'invoke') as mock_invoke, \
             patch.object(graph.graph, 'get_state') as mock_get_state:
            
            mock_invoke.side_effect = GraphRecursionError("Recursion limit reached")
            mock_get_state.return_value = []  # No checkpoints
            
            result = graph.run_meta_prompt_graph(input_state)
            
            # Should return the original input state
            self.assertEqual(result, input_state)
            mock_get_state.assert_called_once()


class TestWorkflowEdgeCases(unittest.TestCase):
    """Test edge cases in workflow execution."""
    
    def test_call_method_with_custom_recursion_limit(self):
        """Test __call__ method with custom recursion limit"""
        mock_llm = Mock(spec=BaseLanguageModel)
        mock_llm.config_specs = []
        graph = MetaPromptGraph(llms=mock_llm)
        
        test_state = {
            "examples": [{"user_message": "test", "expected_output": "result"}],
            "max_output_age": 1
        }
        
        expected_result = {"result": "test"}
        
        with patch.object(graph, 'run_meta_prompt_graph') as mock_run:
            mock_run.return_value = expected_result
            
            result = graph(test_state, recursion_limit=50)
            
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            self.assertEqual(kwargs.get('recursion_limit'), 50)
            self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()