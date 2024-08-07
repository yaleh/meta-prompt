import unittest
import pprint
import logging
import functools
from unittest.mock import MagicMock, Mock
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

# Assuming the necessary imports are made for the classes and functions used in meta_prompt_graph.py
from meta_prompt import *

class TestMetaPromptGraph(unittest.TestCase):
    def setUp(self):
        # logging.basicConfig(level=logging.DEBUG)
        pass


    def test_prompt_node(self):
        """
        Test the _prompt_node method of MetaPromptGraph.

        This test case sets up a mock language model that returns a response content and verifies that the
        updated state has the output attribute updated with the mocked response content.
        """
        llms = {
            NODE_PROMPT_INITIAL_DEVELOPER: MagicMock(
                invoke=MagicMock(return_value=MagicMock(content="Mocked response content"))
            )
        }

        # Create an instance of MetaPromptGraph with the mocked language model and template
        graph = MetaPromptGraph(llms=llms)

        # Create a mock AgentState
        state = AgentState(user_message="Test message", expected_output="Expected output")

        # Invoke the _prompt_node method with the mock node, target attribute, and state
        updated_state = graph._prompt_node(
            NODE_PROMPT_INITIAL_DEVELOPER, "output", state
        )

        # Assertions
        assert updated_state.output == "Mocked response content", \
            "The output attribute should be updated with the mocked response content"


    def test_output_history_analyzer(self):
        """
        Test the _output_history_analyzer method of MetaPromptGraph.

        This test case sets up a mock language model that returns an analysis response and verifies that the
        updated state has the best output, best system message, and best output age updated correctly.
        """
        # Setup
        llms = {
            "output_history_analyzer": MagicMock(invoke=lambda prompt: MagicMock(content="""# Analysis

    This analysis compares two outputs to the expected output based on specific criteria.

    # Output ID closer to Expected Output: B"""))
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
        """
        Test the _prompt_analyzer method of MetaPromptGraph when the prompt analyzer accepts the output.

        This test case sets up a mock language model that returns an acceptance response and verifies that the
        updated state has the accepted attribute set to True.
        """
        llms = {
            NODE_PROMPT_ANALYZER: MagicMock(
                invoke=lambda prompt: MagicMock(content="Accept: Yes"))
        }
        meta_prompt_graph = MetaPromptGraph(llms)
        state = AgentState(output="Test output", expected_output="Expected output")
        updated_state = meta_prompt_graph._prompt_analyzer(state)
        assert updated_state.accepted == True


    def test_get_node_names(self):
        """
        Test the get_node_names method of MetaPromptGraph.

        This test case verifies that the get_node_names method returns the correct list of node names.
        """
        graph = MetaPromptGraph()
        node_names = graph.get_node_names()
        self.assertEqual(node_names, META_PROMPT_NODES)


    def test_workflow_execution(self):
        """
        Test the workflow execution of the MetaPromptGraph.

        This test case sets up a MetaPromptGraph with a single language model and
        executes it with a given input state. It then verifies that the output
        state contains the expected keys and values.
        """
        # MODEL_NAME = "anthropic/claude-3.5-sonnet:beta"
        # MODEL_NAME = "meta-llama/llama-3-70b-instruct"
        MODEL_NAME = "deepseek/deepseek-chat"
        # MODEL_NAME = "google/gemma-2-9b-it"
        # MODEL_NAME = "recursal/eagle-7b"
        # MODEL_NAME = "meta-llama/llama-3-8b-instruct"
        llm = ChatOpenAI(model_name=MODEL_NAME)

        meta_prompt_graph = MetaPromptGraph(llms=llm)
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

        # try another similar user message with the generated system message
        user_message = "How can I create a list of numbers in Python?"
        messages = [("system", output_state['best_system_message']), 
                    ("human", user_message)]
        result = llm.invoke(messages)

        # assert attr 'content' in result
        assert hasattr(result, 'content'), \
            "The result should have the attribute 'content'"
        print(result.content)


    def test_workflow_execution_with_llms(self):
        """
        Test the workflow execution of the MetaPromptGraph with multiple LLMs.

        This test case sets up a MetaPromptGraph with multiple language models and
        executes it with a given input state. It then verifies that the output
        state contains the expected keys and values.
        """
        optimizer_llm = ChatOpenAI(model_name="deepseek/deepseek-chat", temperature=0.5)
        executor_llm = ChatOpenAI(model_name="meta-llama/llama-3-8b-instruct", temperature=0.01)

        llms = {
            NODE_PROMPT_INITIAL_DEVELOPER: optimizer_llm,
            NODE_PROMPT_DEVELOPER: optimizer_llm,
            NODE_PROMPT_EXECUTOR: executor_llm,
            NODE_OUTPUT_HISTORY_ANALYZER: optimizer_llm,
            NODE_PROMPT_ANALYZER: optimizer_llm,
            NODE_PROMPT_SUGGESTER: optimizer_llm
        }

        meta_prompt_graph = MetaPromptGraph(llms=llms)
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

        # try another similar user message with the generated system message
        user_message = "How can I create a list of numbers in Python?"
        messages = [("system", output_state['best_system_message']), 
                    ("human", user_message)]
        result = executor_llm.invoke(messages)

        # assert attr 'content' in result
        assert hasattr(result, 'content'), \
            "The result should have the attribute 'content'"
        print(result.content)
        

    def test_simple_workflow_execution(self):
        """
        Test the simple workflow execution of the MetaPromptGraph.

        This test case sets up a MetaPromptGraph with a mock LLM and executes it
        with a given input state. It then verifies that the output state contains
        the expected keys and values.
        """
        # Create a mock LLM that returns predefined responses based on the input messages
        llm = Mock(spec=BaseLanguageModel)
        responses = [
            Mock(type="content", content="Explain how to reverse a list in Python."),  # NODE_PROMPT_INITIAL_DEVELOPER
            Mock(type="content", content="Here's one way: `my_list[::-1]`"),  # NODE_PROMPT_EXECUTOR
            Mock(type="content", content="Accept: Yes"),  # NODE_PPROMPT_ANALYZER
        ]
        llm.invoke = functools.partial(next, iter(responses))

        meta_prompt_graph = MetaPromptGraph(llms=llm)
        input_state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="The output should use the `reverse()` method.",
            acceptance_criteria="The output should be correct and efficient."
        )

        output_state = meta_prompt_graph(input_state)

        self.assertIsNotNone(output_state['best_system_message'])
        self.assertIsNotNone(output_state['best_output'])

        pprint.pp(output_state["best_output"])
        

    def test_iterated_workflow_execution(self):
        """
        Test the iterated workflow execution of the MetaPromptGraph.

        This test case sets up a MetaPromptGraph with a mock LLM and executes it
        with a given input state. It then verifies that the output state contains
        the expected keys and values. The test case simulates an iterated workflow
        where the LLM provides multiple responses based on the input messages.
        """
        # Create a mock LLM that returns predefined responses based on the input messages
        llm = Mock(spec=BaseLanguageModel)
        responses = [
            Mock(type="content", content="Explain how to reverse a list in Python."),  # NODE_PROMPT_INITIAL_DEVELOPER
            Mock(type="content", content="Here's one way: `my_list[::-1]`"),  # NODE_PROMPT_EXECUTOR
            Mock(type="content", content="Accept: No"),  # NODE_PPROMPT_ANALYZER
            Mock(type="content", content="Try using the `reverse()` method instead."),  # NODE_PROMPT_SUGGESTER
            Mock(type="content", content="Explain how to reverse a list in Python. Output in a Markdown List."),  # NODE_PROMPT_DEVELOPER
            Mock(type="content", content="Here's one way: `my_list.reverse()`"),  # NODE_PROMPT_EXECUTOR
            Mock(type="content", content="# Output ID closer to Expected Output: B"), # NODE_OUTPUT_HISTORY_ANALYZER
            Mock(type="content", content="Accept: Yes"),  # NODE_PPROMPT_ANALYZER
        ]
        llm.invoke = lambda _: responses.pop(0)

        meta_prompt_graph = MetaPromptGraph(llms=llm)
        input_state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="The output should use the `reverse()` method.",
            acceptance_criteria="The output should be correct and efficient."
        )

        output_state = meta_prompt_graph(input_state)

        self.assertIsNotNone(output_state['best_system_message'])
        self.assertIsNotNone(output_state['best_output'])

        pprint.pp(output_state["best_output"])


if __name__ == '__main__':
    unittest.main()