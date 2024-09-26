import json
import os
import pprint
import unittest
from unittest.mock import MagicMock, Mock, patch

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from openai import BadRequestError

from meta_prompt import *
from meta_prompt.consts import NODE_ACCEPTANCE_CRITERIA_DEVELOPER

class TestMetaPromptGraph(unittest.TestCase):
    def setUp(self):
        # Initialize common mocks and objects for the tests
        self.mock_llm = Mock(spec=BaseLanguageModel)
        self.mock_llm.invoke = MagicMock(return_value="Mocked response content")
        self.mock_llm.config_specs = []  # Add this line to fix the iteration error
        
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
        """
        Test the _prompt_node method of MetaPromptGraph.

        This test case sets up a mock language model that returns a response content
        and verifies that the updated state has the output attribute updated with
        the mocked response content.
        """
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y=None: "Mocked response content"

        llms = {
            NODE_PROMPT_INITIAL_DEVELOPER: llm
        }

        graph = MetaPromptGraph(llms=llms)
        state = AgentState(
            user_message="Test message", expected_output="Expected output"
        )
        updated_state = graph._prompt_node(
            NODE_PROMPT_INITIAL_DEVELOPER, "output", state
        )

        assert (
            updated_state['output'] == "Mocked response content"
        ), "The output attribute should be updated with the mocked response content"


    def test_output_history_analyzer(self):
        """
        Test the _output_history_analyzer method of MetaPromptGraph.

        This test case sets up a mock language model that returns an analysis
        response and verifies that the updated state has the best output, best
        system message, and best output age updated correctly.
        """
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y: '{"closerOutputID": 2, "analysis": "The output should use the `reverse()` method."}'
        prompts = {}
        meta_prompt_graph = MetaPromptGraph(llms=llm, prompts=prompts)
        state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="Use the `[::-1]` slicing technique or the `list.reverse()` method.",
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
        """
        Test the _prompt_analyzer method of MetaPromptGraph when the prompt analyzer
        accepts the output.

        This test case sets up a mock language model that returns an acceptance
        response and verifies that the updated state has the accepted attribute
        set to True.
        """
        # llms = {
        #     NODE_PROMPT_ANALYZER: lambda prompt: "{\"Accept\": \"Yes\"}"
        # }
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y: "{\"Accept\": \"Yes\"}"
        meta_prompt_graph = MetaPromptGraph(llms=llm)
        state = AgentState(
            output="Test output", expected_output="Expected output",
            acceptance_criteria="Acceptance criteria: ...",
            system_message="System message: ...",
            max_output_age=2
        )
        updated_state = meta_prompt_graph._prompt_analyzer(state)
        assert updated_state['accepted'] is True


    def test_get_node_names(self):
        """
        Test the get_node_names method of MetaPromptGraph.

        This test case verifies that the get_node_names method returns the
        correct list of node names.
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
            user_message="How do I reverse a list in Python?",
            expected_output="Use the `[::-1]` slicing technique or the "
                            "`list.reverse()` method.",
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
        messages = [("system", output_state["best_system_message"]), ("human", user_message)]
        result = raw_llm.invoke(messages)

        assert hasattr(result, "content"), "The result should have the attribute 'content'"
        print(result.content)


    def test_workflow_execution_with_llms(self):
        """
        Test the workflow execution of the MetaPromptGraph with multiple LLMs.

        This test case sets up a MetaPromptGraph with multiple language models and
        executes it with a given input state. It then verifies that the output
        state contains the expected keys and values.
        """
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
            max_output_age=2,
            user_message="How do I reverse a list in Python?",
            expected_output="Use the `[::-1]` slicing technique or the "
                            "`list.reverse()` method.",
            # acceptance_criteria="Similar in meaning, text length and style."
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
        messages = [("system", output_state["best_system_message"]), ("human", user_message)]
        result = executor_llm.invoke(messages)

        assert hasattr(result, "content"), "The result should have the attribute 'content'"
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
        llm.config_specs = []
        responses = [
            "Explain how to reverse a list in Python.",  # NODE_PROMPT_INITIAL_DEVELOPER
            "Here's one way: `my_list[::-1]`",  # NODE_PROMPT_EXECUTOR
            "{\"Accept\": \"Yes\"}",  # NODE_PPROMPT_ANALYZER
        ]
        # everytime llm.invoke was called, it returns a item in responses
        llm.invoke = lambda x, y=None: responses.pop(0)

        meta_prompt_graph = MetaPromptGraph(llms=llm)
        input_state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="The output should use the `reverse()` method.",
            acceptance_criteria="The output should be correct and efficient.",
            max_output_age=2
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
        llm.config_specs = []
        responses = [
            "Explain how to reverse a list in Python.",  # NODE_PROMPT_INITIAL_DEVELOPER
            "Here's one way: `my_list[::-1]`",  # NODE_PROMPT_EXECUTOR
            "{\"Accept\": \"No\"}",  # NODE_PPROMPT_ANALYZER
            "Try using the `reverse()` method instead.",  # NODE_PROMPT_SUGGESTER
            "Explain how to reverse a list in Python. Output in a Markdown List.",  # NODE_PROMPT_DEVELOPER
            "Here's one way: `my_list.reverse()`",  # NODE_PROMPT_EXECUTOR
            "{\"closerOutputID\": 2, \"analysis\": \"The output should use the `reverse()` method.\"}", # NODE_OUTPUT_HISTORY_ANALYZER
            "{\"Accept\": \"Yes\"}",  # NODE_PPROMPT_ANALYZER
        ]
        llm.invoke = lambda x, y = None: responses.pop(0)

        meta_prompt_graph = MetaPromptGraph(llms=llm)
        input_state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="The output should use the `reverse()` method.",
            acceptance_criteria="The output should be correct and efficient.",
            max_output_age=2
        )

        output_state = meta_prompt_graph(input_state)

        self.assertIsNotNone(output_state['best_system_message'])
        self.assertIsNotNone(output_state['best_output'])

        pprint.pp(output_state["best_output"])

    def test_create_acceptance_criteria_workflow(self):
        """
        Test the _create_acceptance_criteria_workflow method of MetaPromptGraph.

        This test case verifies that the workflow created by the
        _create_acceptance_criteria_workflow method contains the correct node and edge.
        """

        llms = {
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: ChatOpenAI(
                model_name=os.getenv("TEST_MODEL_NAME_ACCEPTANCE_CRITERIA_DEVELOPER")
            )
        }
        meta_prompt_graph = MetaPromptGraph(llms=llms)
        workflow = meta_prompt_graph._create_workflow_for_node(
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER
        )

        # Check if the workflow contains the correct node
        self.assertIn(NODE_ACCEPTANCE_CRITERIA_DEVELOPER, workflow.nodes)

        # Check if the workflow contains the correct edge
        self.assertIn((NODE_ACCEPTANCE_CRITERIA_DEVELOPER, END), workflow.edges)

        # compile the workflow
        graph = workflow.compile()
        print(graph)

        # invoke the workflow
        state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="The output should use the `reverse()` method.",
            # system_message="Create acceptance criteria for the task of reversing a list in Python."
        )
        output_state = graph.invoke(state)

        # check if the output state contains the acceptance criteria
        self.assertIsNotNone(output_state['acceptance_criteria'])

        # check if the acceptance criteria includes string '`reverse()`'
        self.assertIn('`reverse()`', output_state['acceptance_criteria'])

        pprint.pp(output_state["acceptance_criteria"])


    def test_run_acceptance_criteria_graph(self):
        """Test the run_acceptance_criteria_graph method of MetaPromptGraph.

        This test case verifies that the run_acceptance_criteria_graph method
        returns a state with acceptance criteria.
        """
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y: "{\"Acceptance criteria\": \"Acceptance criteria: ...\"}"
        meta_prompt_graph = MetaPromptGraph(llms=llm)
        state = AgentState(
            user_message="How do I reverse a list in Python?",
            expected_output="The output should use the `reverse()` method.",
        )
        output_state = meta_prompt_graph.run_node_graph(NODE_ACCEPTANCE_CRITERIA_DEVELOPER, state)

        # Check if the output state contains the acceptance criteria
        self.assertIsNotNone(output_state["acceptance_criteria"])

        # Check if the acceptance criteria includes the expected content
        self.assertIn("Acceptance criteria: ...", output_state["acceptance_criteria"])


    def test_run_prompt_initial_developer_graph(self):
        """Test the run_prompt_initial_developer_graph method of MetaPromptGraph.

        This test case verifies that the run_prompt_initial_developer_graph method
        returns a state with an initial developer prompt.
        """
        llm = Mock(spec=BaseLanguageModel)
        llm.config_specs = []
        llm.invoke = lambda x, y: "{\"Initial developer prompt\": \"Initial developer prompt: ...\"}"
        meta_prompt_graph = MetaPromptGraph(llms=llm)
        state = AgentState(user_message="How do I reverse a list in Python?")
        output_state = meta_prompt_graph.run_node_graph(NODE_PROMPT_INITIAL_DEVELOPER, state)

        # Check if the output state contains the initial developer prompt
        self.assertIsNotNone(output_state['system_message'])

        # Check if the initial developer prompt includes the expected content
        self.assertIn("Initial developer prompt: ...", output_state['system_message'])

    def test_workflow_execution_multiple_iterations(self):
        """
        Simulate multiple iterations to reach an acceptable output with separate mocks for each node.
        """
        # Create separate mocks for each node
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
            user_message="How do I reverse a list in Python?",
            expected_output="Use the `reverse()` method.",
            acceptance_criteria="The output should use the `reverse()` method.",
            max_output_age=3
        )
        
        output_state = meta_prompt_graph(input_state)
        self.assertTrue(output_state['accepted'])
        self.assertEqual(output_state['best_output'], "Final executor response.")

    def test_workflow_execution_error_handling(self):
        """
        Simulate LLM errors and verify that the workflow handles them gracefully.
        """
        mock_llm = Mock(spec=BaseLanguageModel)
        mock_llm.invoke = MagicMock(side_effect=[
            BadRequestError("Bad request", response=Mock(status_code=400, request=Mock()), body=None),
            "Valid response after retry"
        ])
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
            user_message="How do I reverse a list in Python?",
            expected_output="Use the `reverse()` method.",
            acceptance_criteria="The output should use the `reverse()` method.",
            max_output_age=2
        )
        
        with patch.object(meta_prompt_graph, '_output_history_analyzer', side_effect=[GraphRecursionError]):
            output_state = None
            with self.assertRaises((BadRequestError, KeyError)) as context:
                output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
            
            if isinstance(context.exception, BadRequestError):
                self.assertEqual(str(context.exception), "Bad request")
            elif isinstance(context.exception, KeyError):
                self.assertIsInstance(context.exception, KeyError)
            
            # assert that output_state is not set due to the error
            self.assertIsNone(output_state)

    def test_workflow_execution_output_quality(self):
        """
        Implement a basic output quality check and verify that the final output meets criteria.
        """
        mock_llm = Mock(spec=BaseLanguageModel)
        mock_llm.invoke = MagicMock(return_value="Reverse list using reverse() method.")
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
            user_message="How do I reverse a list in Python?",
            expected_output="Use the `reverse()` method.",
            acceptance_criteria="The output should include the `reverse()` method.",
            max_output_age=2
        )
        
        output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
        self.assertIn("reverse()", output_state['best_output'])

    # New Test Cases for test_workflow_execution_with_llms
    def test_workflow_execution_with_llms_various_scenarios(self):
        """
        Test workflow execution with various LLM configurations and responses.
        """
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
            user_message="Explain how to reverse a list in Python.",
            expected_output="Use the `reverse()` method.",
            acceptance_criteria="The output should include the `reverse()` method.",
            max_output_age=2
        )
        
        output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
        self.assertEqual(output_state['best_output'], "Executor output response.")
        self.assertTrue(output_state['accepted'])

    def test_workflow_execution_with_llms_error_handling(self):
        """
        Simulate LLM errors in a multi-LLM setup and verify graceful handling.
        """
        mock_optimizer_llm = Mock(spec=BaseLanguageModel)
        mock_optimizer_llm.invoke.side_effect = [
            BadRequestError(
                "Bad request",
                response=Mock(status_code=400, request=Mock()),
                body=None
            ),
            "Optimizer response after retry",
            "Optimizer response after retry",
        ]
        mock_optimizer_llm.config_specs = []

        mock_executor_llm = Mock(spec=BaseLanguageModel)
        mock_executor_llm.invoke.return_value = "Executor response."
        mock_executor_llm.config_specs = []

        meta_prompt_graph = MetaPromptGraph(llms={
            NODE_PROMPT_INITIAL_DEVELOPER: mock_optimizer_llm,
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: mock_optimizer_llm,
            NODE_PROMPT_DEVELOPER: mock_optimizer_llm,
            NODE_PROMPT_EXECUTOR: mock_executor_llm,
            NODE_OUTPUT_HISTORY_ANALYZER: mock_optimizer_llm,
            NODE_PROMPT_ANALYZER: mock_optimizer_llm,
            NODE_PROMPT_SUGGESTER: mock_optimizer_llm,
        })
        
        input_state = AgentState(
            user_message="Explain how to reverse a list in Python.",
            expected_output="Use the `reverse()` method.",
            acceptance_criteria="The output should include the `reverse()` method.",
            max_output_age=2
        )
        
        with self.assertRaises(BadRequestError):
            meta_prompt_graph.run_meta_prompt_graph(input_state)

    def test_workflow_execution_with_llms_recursion_limit(self):
        """
        Verify recursion limit handling in multi-LLM setup.
        """
        mock_llm = Mock(spec=BaseLanguageModel)
        # TODO: update the response to be a more complex response that can be used to test the recursion limit
        mock_llm.invoke.side_effect = ["Response"] * 30  # Exceed recursion limit
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
            user_message="Describe the process of list reversal in Python.",
            expected_output="Use the `reverse()` method.",
            acceptance_criteria="The output should detail the `reverse()` method.",
            max_output_age=2
        )
        
        # with self.assertRaises(GraphRecursionError):
        output_state = meta_prompt_graph.run_meta_prompt_graph(input_state, recursion_limit=5)
        self.assertIsNotNone(output_state['best_output'])

    def test_workflow_execution_with_llms_output_quality(self):
        """
        Verify that the output from different LLMs meets quality criteria.
        """
        # Create separate mocks for each node
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
            user_message="Describe the list reversal process in Python.",
            expected_output="Use the `reverse()` method.",
            acceptance_criteria="The output should clearly explain the `reverse()` method.",
            max_output_age=2
        )
        
        output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
        self.assertIn("reverse()", output_state['best_output'])
        self.assertTrue(output_state['accepted'])

    def test_workflow_execution_with_llms_state_persistence(self):
        """
        Verify that the agent state is correctly maintained throughout the workflow.
        """
        # Create separate mocks for each node
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
                "Unacceptable Differences": []
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
        mock_acceptance_criteria.invoke.side_effect = ["Acceptance criteria response."]
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
            user_message="Explain the list reversal process in Python.",
            expected_output="Use the `reverse()` method.",
            acceptance_criteria="The output should provide a clear explanation of the `reverse()` method.",
            max_output_age=3
        )
        
        output_state = meta_prompt_graph.run_meta_prompt_graph(input_state)
        self.assertEqual(output_state['best_output'], "Final executor output.")
        self.assertTrue(output_state['accepted'])

if __name__ == '__main__':
    unittest.main()