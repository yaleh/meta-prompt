import unittest
from unittest.mock import MagicMock, Mock

from langchain_core.language_models import BaseLanguageModel
from openai import BadRequestError

from meta_prompt import *
from meta_prompt.consts import NODE_PROMPT_INITIAL_DEVELOPER, NODE_ACCEPTANCE_CRITERIA_DEVELOPER, NODE_PROMPT_DEVELOPER, NODE_PROMPT_EXECUTOR, NODE_OUTPUT_HISTORY_ANALYZER, NODE_PROMPT_ANALYZER, NODE_PROMPT_SUGGESTER


class TestMetaPromptGraphErrorHandling(unittest.TestCase):
    def test_workflow_execution_error_handling(self):
        mock_llm = Mock(spec=BaseLanguageModel)

        def invoke_side_effect(*args, **kwargs):
            if invoke_side_effect.call_count == 0:
                invoke_side_effect.call_count += 1
                raise BadRequestError("Bad request", response=Mock(
                    status_code=400, request=Mock()), body=None)
            else:
                return "Valid response after retry"
        invoke_side_effect.call_count = 0

        mock_llm.invoke = MagicMock(side_effect=invoke_side_effect)
        mock_llm.config_specs = []

        meta_prompt_graph = MetaPromptGraph(llms={
            NODE_PROMPT_INITIAL_DEVELOPER: mock_llm,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: mock_llm,
            NODE_PROMPT_DEVELOPER: mock_llm,
            NODE_PROMPT_EXECUTOR: mock_llm,
            NODE_OUTPUT_HISTORY_ANALYZER: mock_llm,
            NODE_PROMPT_ANALYZER: mock_llm,
            NODE_PROMPT_SUGGESTER: mock_llm,
        })

        input_state = AgentState(
            examples=[Example(
                user_message="How do I reverse a list in Python?",
                expected_output="Use the `reverse()` method."
            )],
            acceptance_criteria="The output should use the `reverse()` method.",
            max_output_age=2
        )

        try:
            output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
            self.assertEqual(output_state['output'],
                             "Valid response after retry")
        except BadRequestError as e:
            self.assertEqual(str(e), "Bad request")
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

    def test_workflow_execution_with_llms_error_handling(self):
        mock_optimizer_success_llm = Mock(spec=BaseLanguageModel)
        mock_optimizer_success_llm.invoke.return_value = "Optimizer response."
        mock_optimizer_success_llm.config_specs = []

        mock_optimizer_error_llm = Mock(spec=BaseLanguageModel)
        mock_optimizer_error_llm.invoke.side_effect = \
            BadRequestError(
                "Bad request",
                response=Mock(status_code=400, request=Mock()),
                body=None
            )
        mock_optimizer_error_llm.config_specs = []

        mock_executor_llm = Mock(spec=BaseLanguageModel)
        mock_executor_llm.invoke.return_value = "Executor response."
        mock_executor_llm.config_specs = []

        meta_prompt_graph = MetaPromptGraph(llms={
            NODE_PROMPT_INITIAL_DEVELOPER: mock_optimizer_error_llm,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: mock_optimizer_success_llm,
            NODE_PROMPT_DEVELOPER: mock_optimizer_success_llm,
            NODE_PROMPT_EXECUTOR: mock_executor_llm,
            NODE_OUTPUT_HISTORY_ANALYZER: mock_optimizer_success_llm,
            NODE_PROMPT_ANALYZER: mock_optimizer_success_llm,
            NODE_PROMPT_SUGGESTER: mock_optimizer_success_llm,
        })

        input_state = AgentState(
            examples=[Example(
                user_message="Explain how to reverse a list in Python.",
                expected_output="Use the `reverse()` method."
            )],
            acceptance_criteria="The output should include the `reverse()` method.",
            max_output_age=2
        )

        with self.assertRaises(BadRequestError):
            meta_prompt_graph.run_meta_prompt_graph(input_state)
