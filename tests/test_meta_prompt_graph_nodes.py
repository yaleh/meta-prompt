import json
import unittest
from unittest.mock import MagicMock, Mock

from langchain_core.language_models import BaseLanguageModel

from meta_prompt import *
from meta_prompt.consts import NODE_ACCEPTANCE_CRITERIA_DEVELOPER, NODE_PROMPT_INITIAL_DEVELOPER


class TestMetaPromptGraphNodes(unittest.TestCase):
    def setUp(self):
        self.mock_llm = Mock(spec=BaseLanguageModel)
        self.mock_llm.invoke = MagicMock(
            return_value="Mocked response content")
        self.mock_llm.config_specs = []

        self.meta_prompt_graph = MetaPromptGraph(llms={
            NODE_PROMPT_INITIAL_DEVELOPER: self.mock_llm,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: self.mock_llm,
            NODE_PROMPT_DEVELOPER: self.mock_llm,
            NODE_PROMPT_EXECUTOR: self.mock_llm,
            NODE_OUTPUT_HISTORY_ANALYZER: self.mock_llm,
            NODE_PROMPT_ANALYZER: self.mock_llm,
            NODE_PROMPT_SUGGESTER: self.mock_llm,
        })

    def test_prompt_node(self):
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y=None: "Mocked response content"

        llms = {
            NODE_PROMPT_INITIAL_DEVELOPER: llm
        }

        graph = MetaPromptGraph(llms=llms)
        state = AgentState(
            examples=[Example(user_message="Test message",
                              expected_output="Expected output")]
        )
        updated_state = graph._prompt_node(
            NODE_PROMPT_INITIAL_DEVELOPER, "output", state
        )

        assert (
            updated_state['output'] == "Mocked response content"
        ), "The output attribute should be updated with the mocked response content"

    def test_output_history_analyzer(self):
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y: '{"closerOutputID": 2, "analysis": "The output should use the `reverse()` method."}'
        prompts = {}
        meta_prompt_graph = MetaPromptGraph(llms=llm, prompts=prompts)
        state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use the `[::-1]` slicing technique or the `list.reverse()` method."
            )],
            output="To reverse a list in Python, you can use the `[::-1]` slicing.",
            system_message="To reverse a list, use slicing or the reverse method.",
            best_output="To reverse a list in Python, use the `reverse()` method.",
            best_system_message="To reverse a list, use the `reverse()` method.",
            acceptance_criteria="The output should correctly describe how to reverse a list in Python.",
        )

        updated_state = meta_prompt_graph._output_history_analyzer(state)

        assert (
            updated_state['best_output'] == state['output']
        ), "Best output should be updated to the current output."
        assert (
            updated_state['best_system_message'] == state['system_message']
        ), "Best system message should be updated to the current system message."
        assert (
            updated_state['best_output_age'] == 0
        ), "Best output age should be reset to 0."

    def test_prompt_analyzer_accept(self):
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y: "{\"Accept\": \"Yes\"}"
        meta_prompt_graph = MetaPromptGraph(llms=llm)
        state = AgentState(
            examples=[Example(expected_output="Expected output")],
            output="Test output",
            acceptance_criteria="Acceptance criteria: ...",
            system_message="System message: ...",
            max_output_age=2
        )
        updated_state = meta_prompt_graph._prompt_analyzer(state)
        assert updated_state['accepted'] is True

    def test_run_acceptance_criteria_graph(self):
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y: "{\"Acceptance criteria\": \"Acceptance criteria: ...\"}"
        meta_prompt_graph = MetaPromptGraph(llms=llm)
        state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="The output should use the `reverse()` method."
            )]
        )
        output_state = meta_prompt_graph.run_node_graph(
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER, state)

        # Check if the output state contains the acceptance criteria
        self.assertIsNotNone(output_state["acceptance_criteria"])

        # Check if the acceptance criteria includes the expected content
        self.assertIn("Acceptance criteria: ...",
                      output_state["acceptance_criteria"])

    def test_run_prompt_initial_developer_graph(self):
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y: '{"Initial developer prompt": "Initial developer prompt: ..."}'
        meta_prompt_graph = MetaPromptGraph(llms=llm)
        state = AgentState(
            examples=[
                Example(
                    user_message="How do I reverse a list in Python?",
                    expected_output="Use the `reverse()` method."
                )
            ]
        )
        output_state = meta_prompt_graph.run_node_graph(
            NODE_PROMPT_INITIAL_DEVELOPER, state
        )

        # Check if the output state contains the initial developer prompt
        self.assertIsNotNone(output_state['system_message'])

        # Check if the initial developer prompt includes the expected content
        self.assertIn("Initial developer prompt: ...",
                      output_state['system_message'])
