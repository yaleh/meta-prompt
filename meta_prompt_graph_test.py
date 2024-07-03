import unittest
import pprint
import logging
from unittest.mock import MagicMock
from unittest.mock import patch

# Assuming the necessary imports are made for the classes and functions used in meta_prompt_graph.py
from meta_prompt_graph import MetaPromptGraph, AgentState

from langchain_openai import ChatOpenAI

class TestMetaPromptGraph(unittest.TestCase):
    def setUp(self):
        # Mocking the BaseLanguageModel and ChatPromptTemplate for testing
        logging.basicConfig(level=logging.DEBUG)

    def test_prompt_node(self):
        llms = {
            MetaPromptGraph.NODE_PROMPT_INITIAL_DEVELOPER: MagicMock(
                invoke=MagicMock(return_value=MagicMock(content="Mocked response content"))
            )
        }

        # Create an instance of MetaPromptGraph with the mocked language model and template
        graph = MetaPromptGraph(llms=llms)

        # Create a mock AgentState
        state = AgentState(user_message="Test message", expected_output="Expected output")

        # Invoke the _prompt_node method with the mock node, target attribute, and state
        updated_state = graph._prompt_node(
            MetaPromptGraph.NODE_PROMPT_INITIAL_DEVELOPER, "output", state
        )

        # Assertions
        assert updated_state.output == "Mocked response content", \
            "The output attribute should be updated with the mocked response content"

    def test_output_history_analyzer(self):
        # Setup
        llms = {
            "output_history_analyzer": MagicMock(invoke=lambda prompt: MagicMock(content="""# Analysis

    This analysis compares two outputs to the expected output based on specific criteria.

    # Preferred Output ID: B"""))
        }
        prompts = {}
        meta_prompt_graph = MetaPromptGraph(llms=llms, prompts=prompts)
        state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="Use the `[::-1]` slicing technique or the `list.reverse()` method.",
            output="To reverse a list in Python, you can use the `[::-1]` slicing.",
            system_message="To reverse a list, use slicing or the reverse method.",
            best_output="To reverse a list in Python, use the `reverse()` method.",
            best_system_message="To reverse a list, use the `reverse()` method.",
            acceptance_criteria="The output should correctly describe how to reverse a list in Python."
        )

        # Invoke the output history analyzer node
        updated_state = meta_prompt_graph._output_history_analyzer(state)

        # Assertions
        assert updated_state.best_output == state.output, \
            "Best output should be updated to the current output."
        assert updated_state.best_system_message == state.system_message, \
            "Best system message should be updated to the current system message."
        assert updated_state.best_output_age == 0, \
            "Best output age should be reset to 0."

    def test_prompt_analyzer_accept(self):
        llms = {
            MetaPromptGraph.NODE_PROMPT_ANALYZER: MagicMock(
                invoke=lambda prompt: MagicMock(content="Accept: Yes"))
        }
        meta_prompt_graph = MetaPromptGraph(llms)
        state = AgentState(output="Test output", expected_output="Expected output")
        updated_state = meta_prompt_graph._prompt_analyzer(state)
        assert updated_state.accepted == True

    def test_workflow_execution(self):
        # MODEL_NAME = "google/gemma-2-9b-it"
        MODEL_NAME = "anthropic/claude-3.5-sonnet:haiku"
        llm = ChatOpenAI(model_name=MODEL_NAME)

        node_names = MetaPromptGraph.get_node_names()
        llms = {
        }
        for node_name in node_names:
            llms[node_name] = llm

        meta_prompt_graph = MetaPromptGraph(llms=llms, verbose=True)
        input_state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="Use the `[::-1]` slicing technique or the `list.reverse()` method.",
            acceptance_criteria="Similar in meaning, text length and style."
            )
        output_state = meta_prompt_graph(input_state, recursion_limit=25)

        pprint.pp(output_state)
        # if output_state has key 'best_system_message', print it
        assert 'best_system_message' in output_state, \
            "The output state should contain the key 'best_system_message'"
        assert output_state['best_system_message'] is not None, \
            "The best system message should not be None"
        if 'best_system_message' in output_state and output_state['best_system_message'] is not None:
            print(output_state['best_system_message'])

if __name__ == '__main__':
    unittest.main()