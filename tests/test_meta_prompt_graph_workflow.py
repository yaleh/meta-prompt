import json
import os
import pprint
import unittest
from unittest.mock import MagicMock, Mock

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from meta_prompt import *
from meta_prompt.consts import NODE_PROMPT_INITIAL_DEVELOPER, NODE_ACCEPTANCE_CRITERIA_DEVELOPER, NODE_PROMPT_DEVELOPER, NODE_PROMPT_EXECUTOR, NODE_OUTPUT_HISTORY_ANALYZER, NODE_PROMPT_ANALYZER, NODE_PROMPT_SUGGESTER


class TestMetaPromptGraphWorkflow(unittest.TestCase):
    def test_workflow_execution(self):
        model_name = os.getenv("TEST_MODEL_NAME_EXECUTOR")
        raw_llm = ChatOpenAI(model_name=model_name)

        llms = {
            NODE_PROMPT_INITIAL_DEVELOPER: raw_llm,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: raw_llm,
            NODE_PROMPT_DEVELOPER: raw_llm,
            NODE_PROMPT_EXECUTOR: raw_llm,
            NODE_OUTPUT_HISTORY_ANALYZER: raw_llm,
            NODE_PROMPT_ANALYZER: raw_llm,
            NODE_PROMPT_SUGGESTER: raw_llm,
        }

        meta_prompt_graph = MetaPromptGraph(llms=llms)
        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use the `[::-1]` slicing technique or the `list.reverse()` method."
            )],
            acceptance_criteria="Similar in meaning, text length and style.",
            max_output_age=2
        )
        output_state = meta_prompt_graph(input_state, recursion_limit=25)

        pprint.pp(output_state)
        assert (
            "best_system_message" in output_state
        ), "The output state should contain the key 'best_system_message'"
        assert (
            output_state["best_system_message"] is not None
        ), "The best system message should not be None"
        if (
            "best_system_message" in output_state
            and output_state["best_system_message"] is not None
        ):
            print(output_state["best_system_message"])

        user_message = "How can I create a list of numbers in Python?"
        messages = [("system", output_state["best_system_message"]),
                    ("human", user_message)]
        result = raw_llm.invoke(messages)

        assert hasattr(
            result, "content"), "The result should have the attribute 'content'"
        print(result.content)

    def test_workflow_execution_with_llms(self):
        optimizer_llm = ChatOpenAI(
            model_name=os.getenv("TEST_MODEL_NAME_OPTIMIZER"), temperature=0.5
        )
        executor_llm = ChatOpenAI(
            model_name=os.getenv("TEST_MODEL_NAME_EXECUTOR"), temperature=0.01
        )

        llms = {
            NODE_PROMPT_INITIAL_DEVELOPER: optimizer_llm,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: optimizer_llm,
            NODE_PROMPT_DEVELOPER: optimizer_llm,
            NODE_PROMPT_EXECUTOR: executor_llm,
            NODE_OUTPUT_HISTORY_ANALYZER: optimizer_llm,
            NODE_PROMPT_ANALYZER: optimizer_llm.bind(response_format={"type": "json_object"}),
            NODE_PROMPT_SUGGESTER: optimizer_llm,
        }

        meta_prompt_graph = MetaPromptGraph(llms=llms)
        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use the `[::-1]` slicing technique or the `list.reverse()` method."
            )],
            acceptance_criteria="Similar in meaning, text length and style.",
            max_output_age=2
        )
        output_state = meta_prompt_graph(input_state, recursion_limit=25)

        pprint.pp(output_state)
        assert (
            "best_system_message" in output_state
        ), "The output state should contain the key 'best_system_message'"
        assert (
            output_state["best_system_message"] is not None
        ), "The best system message should not be None"
        if (
            "best_system_message" in output_state
            and output_state["best_system_message"] is not None
        ):
            print(output_state["best_system_message"])

        user_message = "How can I create a list of numbers in Python?"
        messages = [("system", output_state["best_system_message"]),
                    ("human", user_message)]
        result = executor_llm.invoke(messages)

        assert hasattr(
            result, "content"), "The result should have the attribute 'content'"
        print(result.content)

    def test_simple_workflow_execution(self):
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        responses = [
            "Explain how to reverse a list in Python.",  # NODE_PROMPT_INITIAL_DEVELOPER
            "Here's one way: `my_list[::-1]`",  # NODE_PROMPT_EXECUTOR
            "{\"Accept\": \"Yes\"}",  # NODE_PPROMPT_ANALYZER
        ]
        llm.invoke = lambda x, y=None: responses.pop(0)

        meta_prompt_graph = MetaPromptGraph(llms=llm)
        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="The output should use the `reverse()` method."
            )],
            acceptance_criteria="The output should be correct and efficient.",
            max_output_age=2
        )

        output_state = meta_prompt_graph(input_state)

        self.assertIsNotNone(output_state['best_system_message'])
        self.assertIsNotNone(output_state['best_output'])

        pprint.pp(output_state["best_output"])

    def test_iterated_workflow_execution(self):
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        responses = [
            "Explain how to reverse a list in Python.",  # NODE_PROMPT_INITIAL_DEVELOPER
            "Here's one way: `my_list[::-1]`",  # NODE_PROMPT_EXECUTOR
            "{\"Accept\": \"No\"}",  # NODE_PPROMPT_ANALYZER
            "Try using the `reverse()` method instead.",  # NODE_PROMPT_SUGGESTER
            # NODE_PROMPT_DEVELOPER
            "Explain how to reverse a list in Python. Output in a Markdown List.",
            "Here's one way: `my_list.reverse()`",  # NODE_PROMPT_EXECUTOR
            # NODE_OUTPUT_HISTORY_ANALYZER
            "{\"closerOutputID\": 2, \"analysis\": \"The output should use the `reverse()` method.\"}",
            "{\"Accept\": \"Yes\"}",  # NODE_PPROMPT_ANALYZER
        ]
        llm.invoke = lambda x, y = None: responses.pop(0)

        meta_prompt_graph = MetaPromptGraph(llms=llm)
        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="The output should use the `reverse()` method."
            )],
            acceptance_criteria="The output should be correct and efficient.",
            max_output_age=2
        )

        output_state = meta_prompt_graph(input_state)

        self.assertIsNotNone(output_state['best_system_message'])
        self.assertIsNotNone(output_state['best_output'])

        pprint.pp(output_state["best_output"])

    def test_workflow_execution_multiple_iterations(self):
        mock_initial_developer = Mock(spec=BaseLanguageModel)
        mock_initial_developer.invoke.side_effect = [
            "Initial response",
            "Revised developer prompt."
        ]
        mock_initial_developer.config_specs = []

        mock_acceptance_criteria_developer = Mock(spec=BaseLanguageModel)
        mock_acceptance_criteria_developer.invoke.side_effect = [
            "{\"Accept\": \"No\"}",
            "{\"Accept\": \"Yes\"}"
        ]
        mock_acceptance_criteria_developer.config_specs = []

        mock_prompt_developer = Mock(spec=BaseLanguageModel)
        mock_prompt_developer.invoke.side_effect = [
            "Prompt developer response.",
            "Revised prompt developer response."
        ]
        mock_prompt_developer.config_specs = []

        mock_prompt_executor = Mock(spec=BaseLanguageModel)
        mock_prompt_executor.invoke.side_effect = [
            "Executor initial response.",
            "Revised executor response.",
            "Final executor response."
        ]
        mock_prompt_executor.config_specs = []

        mock_output_history_analyzer = Mock(spec=BaseLanguageModel)
        mock_output_history_analyzer.invoke.side_effect = [
            json.dumps({"closerOutputID": 1, "analysis": "Initial analysis."}),
            json.dumps({"closerOutputID": 2, "analysis": "Revised analysis."})
        ]
        mock_output_history_analyzer.config_specs = []

        mock_prompt_analyzer = Mock(spec=BaseLanguageModel)
        mock_prompt_analyzer.invoke.side_effect = [
            json.dumps({
                "Accept": "No",
                "Acceptable Differences": [],
                "Unacceptable Differences": []
            }),
            json.dumps({
                "Accept": "Yes",
                "Acceptable Differences": [],
                "Unacceptable Differences": []
            })
        ]
        mock_prompt_analyzer.config_specs = []

        mock_prompt_suggester = Mock(spec=BaseLanguageModel)
        mock_prompt_suggester.invoke.side_effect = [
            "Consider refining the prompt.",
            "No suggestion needed."
        ]
        mock_prompt_suggester.config_specs = []

        meta_prompt_graph = MetaPromptGraph(llms={
            NODE_PROMPT_INITIAL_DEVELOPER: mock_initial_developer,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: mock_acceptance_criteria_developer,
            NODE_PROMPT_DEVELOPER: mock_prompt_developer,
            NODE_PROMPT_EXECUTOR: mock_prompt_executor,
            NODE_OUTPUT_HISTORY_ANALYZER: mock_output_history_analyzer,
            NODE_PROMPT_ANALYZER: mock_prompt_analyzer,
            NODE_PROMPT_SUGGESTER: mock_prompt_suggester,
        })

        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use the `reverse()` method."
            )],
            acceptance_criteria="The output should use the `reverse()` method.",
            max_output_age=3
        )

        output_state = meta_prompt_graph(input_state)
        self.assertTrue(output_state['accepted'])
        self.assertEqual(output_state['best_output'],
                         "Final executor response.")

    def test_workflow_execution_with_llms_various_scenarios(self):
        mock_initial_developer = Mock(spec=BaseLanguageModel)
        mock_initial_developer.invoke.return_value = "Initial developer prompt response."
        mock_initial_developer.config_specs = []

        mock_acceptance_criteria_developer = Mock(spec=BaseLanguageModel)
        mock_acceptance_criteria_developer.invoke.return_value = "Acceptance criteria response."
        mock_acceptance_criteria_developer.config_specs = []

        mock_prompt_developer = Mock(spec=BaseLanguageModel)
        mock_prompt_developer.invoke.return_value = "Prompt developer response."
        mock_prompt_developer.config_specs = []

        mock_executor = Mock(spec=BaseLanguageModel)
        mock_executor.invoke.return_value = "Executor output response."
        mock_executor.config_specs = []

        mock_history_analyzer = Mock(spec=BaseLanguageModel)
        mock_history_analyzer.invoke.return_value = json.dumps(
            {"closerOutputID": 2, "analysis": "Good job."}
        )
        mock_history_analyzer.config_specs = []

        mock_analyzer = Mock(spec=BaseLanguageModel)
        mock_analyzer.invoke.return_value = json.dumps(
            {
                "Accept": "Yes",
                "Acceptable Differences": [],
                "Unacceptable Differences": [],
            }
        )
        mock_analyzer.config_specs = []

        mock_suggester = Mock(spec=BaseLanguageModel)
        mock_suggester.invoke.return_value = "Suggestions response."
        mock_suggester.config_specs = []

        meta_prompt_graph = MetaPromptGraph(llms={
            NODE_PROMPT_INITIAL_DEVELOPER: mock_initial_developer,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: mock_acceptance_criteria_developer,
            NODE_PROMPT_DEVELOPER: mock_prompt_developer,
            NODE_PROMPT_EXECUTOR: mock_executor,
            NODE_OUTPUT_HISTORY_ANALYZER: mock_history_analyzer,
            NODE_PROMPT_ANALYZER: mock_analyzer,
            NODE_PROMPT_SUGGESTER: mock_suggester,
        })

        input_state = AgentState(
            examples=[Example(
                user_message="Explain how to reverse a list in Python.",
                expected_output="Use the `reverse()` method."
            )],
            acceptance_criteria="The output should include the `reverse()` method.",
            max_output_age=2
        )

        output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
        self.assertEqual(output_state['best_output'],
                         "Executor output response.")
        self.assertTrue(output_state['accepted'])

    def test_workflow_execution_with_llms_output_quality(self):
        mock_initial_developer = Mock(spec=BaseLanguageModel)
        mock_initial_developer.invoke.return_value = "Initial prompt response."
        mock_initial_developer.config_specs = []

        mock_acceptance_criteria = Mock(spec=BaseLanguageModel)
        mock_acceptance_criteria.invoke.return_value = "Acceptance criteria response."
        mock_acceptance_criteria.config_specs = []

        mock_prompt_developer = Mock(spec=BaseLanguageModel)
        mock_prompt_developer.invoke.return_value = "Prompt developer response."
        mock_prompt_developer.config_specs = []

        mock_executor = Mock(spec=BaseLanguageModel)
        mock_executor.invoke.return_value = (
            "Executor provides a clear method using the `reverse()` method."
        )
        mock_executor.config_specs = []

        mock_history_analyzer = Mock(spec=BaseLanguageModel)
        mock_history_analyzer.invoke.return_value = json.dumps(
            {"closerOutputID": 1, "analysis": "Good output."}
        )
        mock_history_analyzer.config_specs = []

        mock_analyzer = Mock(spec=BaseLanguageModel)
        mock_analyzer.invoke.return_value = json.dumps(
            {
                "Accept": "Yes",
                "Acceptable Differences": [],
                "Unacceptable Differences": [],
            }
        )
        mock_analyzer.config_specs = []

        mock_suggester = Mock(spec=BaseLanguageModel)
        mock_suggester.invoke.return_value = "No suggestion needed."
        mock_suggester.config_specs = []

        meta_prompt_graph = MetaPromptGraph(llms={
            NODE_PROMPT_INITIAL_DEVELOPER: mock_initial_developer,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: mock_acceptance_criteria,
            NODE_PROMPT_DEVELOPER: mock_prompt_developer,
            NODE_PROMPT_EXECUTOR: mock_executor,
            NODE_OUTPUT_HISTORY_ANALYZER: mock_history_analyzer,
            NODE_PROMPT_ANALYZER: mock_analyzer,
            NODE_PROMPT_SUGGESTER: mock_suggester,
        })

        input_state = AgentState(
            examples=[Example(
                user_message="Describe the list reversal process in Python.",
                expected_output="Use the `reverse()` method."
            )],
            acceptance_criteria="The output should clearly explain the `reverse()` method.",
            max_output_age=2
        )

        output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
        self.assertIn("reverse()", output_state['best_output'])
        self.assertTrue(output_state['accepted'])

    def test_workflow_execution_with_llms_state_persistence(self):
        mock_initial_developer = Mock(spec=BaseLanguageModel)
        mock_initial_developer.invoke.side_effect = ["Initial prompt."]
        mock_initial_developer.config_specs = []

        mock_executor = Mock(spec=BaseLanguageModel)
        mock_executor.invoke.side_effect = [
            "Executor output.",
            "Revised executor output.",
            "Final executor output."
        ]
        mock_executor.config_specs = []

        mock_history_analyzer = Mock(spec=BaseLanguageModel)
        mock_history_analyzer.invoke.side_effect = [
            json.dumps({"closerOutputID": 1, "analysis": "Good output."}),
            json.dumps({"closerOutputID": 2, "analysis": "Much better."})
        ]
        mock_history_analyzer.config_specs = []

        mock_analyzer = Mock(spec=BaseLanguageModel)
        mock_analyzer.invoke.side_effect = [
            json.dumps({
                "Accept": "No",
                "Acceptable Differences": [],
                "Unacceptable Differences": []
            }),
            json.dumps({
                "Accept": "Yes",
                "Acceptable Differences": [],
                "Unacceptable Differences": [],
            })
        ]
        mock_analyzer.config_specs = []

        mock_suggester = Mock(spec=BaseLanguageModel)
        mock_suggester.invoke.side_effect = [
            "Consider using an alternative method.",
            "No suggestion needed."
        ]
        mock_suggester.config_specs = []

        mock_developer = Mock(spec=BaseLanguageModel)
        mock_developer.invoke.side_effect = [
            "Revised developer prompt.",
            "Final developer prompt."
        ]
        mock_developer.config_specs = []

        mock_acceptance_criteria = Mock(spec=BaseLanguageModel)
        mock_acceptance_criteria.invoke.side_effect = [
            "Acceptance criteria response."]
        mock_acceptance_criteria.config_specs = []

        meta_prompt_graph = MetaPromptGraph(llms={
            NODE_PROMPT_INITIAL_DEVELOPER: mock_initial_developer,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: mock_acceptance_criteria,
            NODE_PROMPT_DEVELOPER: mock_developer,
            NODE_PROMPT_EXECUTOR: mock_executor,
            NODE_OUTPUT_HISTORY_ANALYZER: mock_history_analyzer,
            NODE_PROMPT_ANALYZER: mock_analyzer,
            NODE_PROMPT_SUGGESTER: mock_suggester,
        })

        input_state = AgentState(
            examples=[Example(
                user_message="Explain the list reversal process in Python.",
                expected_output="Use the `reverse()` method."
            )],
            acceptance_criteria="The output should provide a clear explanation of the `reverse()` method.",
            max_output_age=3
        )

        output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
        self.assertEqual(output_state['best_output'], "Final executor output.")
        self.assertTrue(output_state['accepted'])

    def test_workflow_execution_with_thinking_model(self):
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y=None: "<think>Thinking...</think>Response without think tags."

        meta_prompt_graph = MetaPromptGraph(llms=llm, thinking_model=True)
        input_state = AgentState(
            examples=[Example(
                user_message="Test message",
                expected_output="Expected output"
            )],
            acceptance_criteria="The output should be correct.",
            max_output_age=2
        )

        output_state = meta_prompt_graph(input_state)

        self.assertEqual(output_state['best_output'],
                         "Response without think tags.")
