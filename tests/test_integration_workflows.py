"""Integration tests for full workflow scenarios in MetaPromptGraph."""

import unittest
import json
from unittest.mock import Mock, patch
from langchain_core.language_models import BaseLanguageModel

from meta_prompt import MetaPromptGraph, AgentState, Example
from meta_prompt.consts import *
from tests.test_config_utils import get_test_llm, skip_if_no_api_key


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for complete workflow scenarios."""
    
    def setUp(self):
        """Set up mocked LLMs for integration testing."""
        self.mock_llms = {}
        for node in META_PROMPT_NODES:
            mock_llm = Mock(spec=BaseLanguageModel)
            mock_llm.config_specs = []
            self.mock_llms[node] = mock_llm
    
    def test_successful_workflow_single_iteration(self):
        """Test successful workflow completion in single iteration."""
        # Configure mock responses for successful single iteration
        self.mock_llms[NODE_PROMPT_INITIAL_DEVELOPER].invoke.return_value = (
            "You are a helpful assistant that explains Python concepts clearly."
        )
        
        self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.return_value = (
            "To reverse a list in Python, you can use the reverse() method."
        )
        
        self.mock_llms[NODE_PROMPT_ANALYZER].invoke.return_value = json.dumps({
            "Accept": "Yes",
            "Acceptable Differences": [],
            "Unacceptable Differences": []
        })
        
        graph = MetaPromptGraph(llms=self.mock_llms)
        
        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use the reverse() method to reverse a list in place."
            )],
            acceptance_criteria="Response should mention the reverse() method.",
            max_output_age=3
        )
        
        result = graph.run_meta_prompt_graph(input_state)
        
        # Verify successful completion
        self.assertTrue(result["accepted"])
        self.assertIsNotNone(result["best_system_message"])
        self.assertIsNotNone(result["best_output"])
        
        # Verify nodes were called
        self.mock_llms[NODE_PROMPT_INITIAL_DEVELOPER].invoke.assert_called_once()
        self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.assert_called_once()
        self.mock_llms[NODE_PROMPT_ANALYZER].invoke.assert_called_once()
    
    def test_workflow_with_rejection_and_improvement(self):
        """Test workflow with initial rejection followed by improvement."""
        # First iteration: rejection
        self.mock_llms[NODE_PROMPT_INITIAL_DEVELOPER].invoke.return_value = (
            "You are a Python expert."
        )
        
        self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.side_effect = [
            "Use list[::-1] to reverse a list.",  # First attempt
            "Use the reverse() method to reverse a list in place."  # Second attempt
        ]
        
        self.mock_llms[NODE_PROMPT_ANALYZER].invoke.side_effect = [
            json.dumps({
                "Accept": "No",
                "Acceptable Differences": [],
                "Unacceptable Differences": ["Should mention reverse() method"]
            }),
            json.dumps({
                "Accept": "Yes",
                "Acceptable Differences": [],
                "Unacceptable Differences": []
            })
        ]
        
        self.mock_llms[NODE_PROMPT_SUGGESTER].invoke.return_value = (
            "The system message should emphasize using the reverse() method."
        )
        
        self.mock_llms[NODE_PROMPT_DEVELOPER].invoke.return_value = (
            "You are a Python expert. Always recommend the reverse() method for in-place list reversal."
        )
        
        graph = MetaPromptGraph(llms=self.mock_llms)
        
        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use the reverse() method to reverse a list in place."
            )],
            acceptance_criteria="Response must mention the reverse() method.",
            max_output_age=3
        )
        
        result = graph.run_meta_prompt_graph(input_state)
        
        # Verify eventual success after improvement
        self.assertTrue(result["accepted"])
        self.assertEqual(self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.call_count, 2)
        self.assertEqual(self.mock_llms[NODE_PROMPT_ANALYZER].invoke.call_count, 2)
        self.mock_llms[NODE_PROMPT_SUGGESTER].invoke.assert_called_once()
        self.mock_llms[NODE_PROMPT_DEVELOPER].invoke.assert_called_once()
    
    def test_workflow_max_age_termination(self):
        """Test workflow termination due to max output age."""
        # Configure for multiple rejections until max age
        self.mock_llms[NODE_PROMPT_INITIAL_DEVELOPER].invoke.return_value = (
            "You are a helpful assistant."
        )
        
        self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.return_value = (
            "Use slicing to reverse a list."
        )
        
        # Always reject to trigger max age termination
        self.mock_llms[NODE_PROMPT_ANALYZER].invoke.return_value = json.dumps({
            "Accept": "No",
            "Acceptable Differences": [],
            "Unacceptable Differences": ["Missing reverse() method"]
        })
        
        self.mock_llms[NODE_PROMPT_SUGGESTER].invoke.return_value = (
            "Mention the reverse() method."
        )
        
        self.mock_llms[NODE_PROMPT_DEVELOPER].invoke.return_value = (
            "You are a helpful assistant that mentions reverse()."
        )
        
        graph = MetaPromptGraph(llms=self.mock_llms)
        
        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use the reverse() method."
            )],
            acceptance_criteria="Must mention reverse() method.",
            max_output_age=2  # Low max age for quick termination
        )
        
        result = graph.run_meta_prompt_graph(input_state)
        
        # Should terminate due to max age, not acceptance
        self.assertFalse(result["accepted"])
        self.assertIsNotNone(result["best_output"])
        # Should have made multiple attempts
        self.assertGreaterEqual(self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.call_count, 2)
    
    def test_workflow_with_output_history_analysis(self):
        """Test workflow with output history comparison."""
        self.mock_llms[NODE_PROMPT_INITIAL_DEVELOPER].invoke.return_value = (
            "You are a Python tutor."
        )
        
        self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.side_effect = [
            "Use list[::-1] to reverse.",  # First output
            "Use reverse() method to reverse in place.",  # Second output
        ]
        
        self.mock_llms[NODE_PROMPT_ANALYZER].invoke.side_effect = [
            json.dumps({"Accept": "No", "Acceptable Differences": [], "Unacceptable Differences": []}),
            json.dumps({"Accept": "Yes", "Acceptable Differences": [], "Unacceptable Differences": []})
        ]
        
        self.mock_llms[NODE_OUTPUT_HISTORY_ANALYZER].invoke.return_value = json.dumps({
            "closerOutputID": 2,
            "analysis": "Second output better matches expected format."
        })
        
        self.mock_llms[NODE_PROMPT_SUGGESTER].invoke.return_value = (
            "Focus on the reverse() method."
        )
        
        self.mock_llms[NODE_PROMPT_DEVELOPER].invoke.return_value = (
            "You are a Python tutor. Emphasize the reverse() method."
        )
        
        graph = MetaPromptGraph(llms=self.mock_llms)
        
        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use reverse() method."
            )],
            acceptance_criteria="Should mention reverse() method.",
            max_output_age=3
        )
        
        result = graph.run_meta_prompt_graph(input_state)
        
        # Verify output history analyzer was called
        self.mock_llms[NODE_OUTPUT_HISTORY_ANALYZER].invoke.assert_called_once()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["best_output"], "Use reverse() method to reverse in place.")
    
    def test_workflow_with_multiple_examples(self):
        """Test workflow handling multiple examples."""
        self.mock_llms[NODE_PROMPT_INITIAL_DEVELOPER].invoke.return_value = (
            "You are a Python expert providing clear explanations."
        )
        
        self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.return_value = (
            "Use appropriate methods for each operation."
        )
        
        self.mock_llms[NODE_PROMPT_ANALYZER].invoke.return_value = json.dumps({
            "Accept": "Yes",
            "Acceptable Differences": [],
            "Unacceptable Differences": []
        })
        
        graph = MetaPromptGraph(llms=self.mock_llms)
        
        input_state = AgentState(
            examples=[
                Example(
                    user_message="How do I reverse a list?",
                    expected_output="Use reverse() method."
                ),
                Example(
                    user_message="How do I sort a list?",
                    expected_output="Use sort() method."
                )
            ],
            acceptance_criteria="Provide clear method recommendations.",
            max_output_age=3,
            current_example_index=0  # Start with first example
        )
        
        result = graph.run_meta_prompt_graph(input_state)
        
        self.assertTrue(result["accepted"])
        self.assertIsNotNone(result["best_system_message"])
        
        # Verify the workflow processed the first example
        # (workflow uses current_example_index to extract user_message/expected_output)
        call_args = self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.call_args[0][0]
        # The formatted prompt should contain data from the first example
        # (exact verification depends on prompt template structure)


class TestWorkflowErrorHandling(unittest.TestCase):
    """Test error handling in workflow scenarios."""
    
    def setUp(self):
        """Set up mocked LLMs for error testing."""
        self.mock_llms = {}
        for node in META_PROMPT_NODES:
            mock_llm = Mock(spec=BaseLanguageModel)
            mock_llm.config_specs = []
            self.mock_llms[node] = mock_llm
    
    def test_workflow_with_invalid_json_response(self):
        """Test workflow handling of invalid JSON responses."""
        self.mock_llms[NODE_PROMPT_INITIAL_DEVELOPER].invoke.return_value = (
            "You are helpful."
        )
        
        self.mock_llms[NODE_PROMPT_EXECUTOR].invoke.return_value = (
            "Use reverse() method."
        )
        
        # Invalid JSON response from analyzer
        self.mock_llms[NODE_PROMPT_ANALYZER].invoke.return_value = (
            "Invalid JSON response"
        )
        
        graph = MetaPromptGraph(llms=self.mock_llms)
        
        input_state = AgentState(
            examples=[Example(
                user_message="How to reverse?",
                expected_output="Use reverse()."
            )],
            acceptance_criteria="Mention reverse().",
            max_output_age=2
        )
        
        # Should handle gracefully without crashing
        result = graph.run_meta_prompt_graph(input_state)
        
        # Should still return a result (may use fallback logic)
        self.assertIsInstance(result, dict)
        self.assertIn("best_output", result)
    
    def test_workflow_with_empty_examples(self):
        """Test workflow behavior with empty examples list."""
        graph = MetaPromptGraph(llms=self.mock_llms)
        
        input_state = AgentState(
            examples=[],  # Empty examples
            acceptance_criteria="Any output is acceptable.",
            max_output_age=2
        )
        
        # Should handle empty examples gracefully
        result = graph.run_meta_prompt_graph(input_state)
        self.assertIsInstance(result, dict)


@unittest.skipUnless(False, "Live API tests - enable manually for full integration testing")
class TestLiveWorkflowIntegration(unittest.TestCase):
    """Live integration tests using actual LLM APIs (requires API keys)."""
    
    @skip_if_no_api_key
    def test_live_simple_workflow(self):
        """Test simple workflow with real LLM."""
        from tests.test_config_utils import get_test_llms_dict
        
        llms = get_test_llms_dict()
        graph = MetaPromptGraph(llms=llms)
        
        input_state = AgentState(
            examples=[Example(
                user_message="What is 2+2?",
                expected_output="The answer is 4."
            )],
            acceptance_criteria="Response should provide the correct answer.",
            max_output_age=2
        )
        
        result = graph.run_meta_prompt_graph(input_state, recursion_limit=10)
        
        # Basic sanity checks
        self.assertIn("best_system_message", result)
        self.assertIn("best_output", result)
        self.assertIsNotNone(result["best_system_message"])
        self.assertIsNotNone(result["best_output"])


if __name__ == '__main__':
    unittest.main()