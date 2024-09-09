import json
import logging
import pprint
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.base import RunnableLike
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import Annotated, Dict, Optional, Union, TypedDict
from .consts import *

def first_non_empty(a, b):
    # return the first non-none value
    return next((s for s in (a, b) if s), None)

def last_non_empty(a, b):
    # return the last non-none value
    return next((s for s in (b, a) if s), None)

class AgentState(TypedDict):
    """
    Represents the state of an agent in a conversation.

    Attributes:
        max_output_age (int): The maximum age of the output.
        user_message (str, optional): The user's message.
        expected_output (str, optional): The expected output.
        acceptance_criteria (str, optional): The acceptance criteria.
        system_message (str, optional): The system message.
        output (str, optional): The output.
        suggestions (str, optional): The suggestions.
        accepted (bool, optional): Whether the output is accepted.
        analysis (str, optional): The analysis.
        best_output (str, optional): The best output.
        best_system_message (str, optional): The best system message.
        best_output_age (int, optional): The age of the best output.
    """
    max_output_age: Optional[int]
    user_message: Optional[str]
    expected_output: Optional[str]
    acceptance_criteria: Annotated[Optional[str], last_non_empty]
    system_message: Annotated[Optional[str], last_non_empty]
    output: Optional[str]
    suggestions: Optional[str]
    accepted: Optional[bool]
    analysis: Optional[str]
    best_output: Optional[str]
    best_system_message: Optional[str]
    best_output_age: Optional[int]

class MetaPromptGraph:
    """
    This class represents a graph for meta-prompting in a conversational AI system.

    It manages the state of the conversation, including the user's message, expected 
    output, acceptance criteria, system message, output, suggestions, and analysis.

    The graph consists of nodes that represent different stages of the conversation, 
    such as prompting the developer, executing the output, analyzing the output 
    history, and suggesting new prompts.

    The class provides methods to create the workflow, initialize the graph, and 
    invoke the graph with a given state.

    The MetaPromptGraph class is responsible for orchestrating the conversation 
    flow and deciding the next step based on the current state of the 
    conversation. It uses language models and prompt templates to generate 
    responses and analyze the output.
    """
    @classmethod
    def get_node_names(cls):
        """
        Returns a list of node names in the meta-prompt graph.

        This method initializes language models and prompt templates for each node.

        Returns:
            list: List of node names.
        """
        return META_PROMPT_NODES

    def __init__(
        self,
        llms: Union[BaseLanguageModel, Dict[str, BaseLanguageModel]] = {},
        prompts: Dict[str, ChatPromptTemplate] = {},
        aggressive_exploration: bool = False,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        """
        Initializes the MetaPromptGraph instance.

        Args:
            llms: The language models for the graph nodes.
            prompts: The custom prompt templates for the graph nodes.
            aggressive_exploration: Whether to use aggressive exploration.
            logger: The logger for the graph.
            verbose: Whether to set the logger level to DEBUG.

        Initializes the logger, sets the language models and prompt templates
        for the graph nodes, and updates the prompt templates with custom ones
        if provided.
        """
        self.logger = logger or logging.getLogger(__name__)
        if self.logger is not None:
            self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        if isinstance(llms, BaseLanguageModel):
            self.llms = {node: llms for node in self.get_node_names()}
        else:
            self.llms: Dict[str, BaseLanguageModel] = llms
        self.prompt_templates: Dict[str,
                                    ChatPromptTemplate] = DEFAULT_PROMPT_TEMPLATES.copy()
        self.prompt_templates.update(prompts)

        self.aggressive_exploration = aggressive_exploration


    def _create_workflow_for_node(self, node: str) -> StateGraph:
        """Create a workflow state graph for the specified node.

        Args:
            node (str): The node name to create the workflow for.

        Returns:
            StateGraph: A state graph representing the workflow.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node(
            node,
            lambda x: self._prompt_node(
                node,
                self._get_target_attribute_for_node(node),
                x
            )
        )
        workflow.add_edge(node, END)
        workflow.set_entry_point(node)
        return workflow


    def _get_target_attribute_for_node(self, node: str) -> str:
        """Get the target attribute for the specified node.

        Args:
            node (str): The node name.

        Returns:
            str: The target attribute for the node.
        """
        # Define a mapping of nodes to their target attributes
        node_to_attribute = {
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER: "acceptance_criteria",
            NODE_PROMPT_INITIAL_DEVELOPER: "system_message",
            NODE_PROMPT_DEVELOPER: "system_message",
            NODE_PROMPT_EXECUTOR: "output",
            NODE_OUTPUT_HISTORY_ANALYZER: "analysis",
            NODE_PROMPT_ANALYZER: "analysis",
            NODE_PROMPT_SUGGESTER: "suggestions"
        }
        return node_to_attribute.get(node, "")


    def _create_workflow(self) -> StateGraph:
        """
        Create a workflow state graph for the meta-prompt.

        Returns:
            StateGraph: A state graph representing the workflow.
        """

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node(
            NODE_PROMPT_DEVELOPER,
            lambda x: self._prompt_node(
                NODE_PROMPT_DEVELOPER, "system_message", x
            )
        )
        workflow.add_node(
            NODE_PROMPT_EXECUTOR,
            lambda x: self._prompt_node(NODE_PROMPT_EXECUTOR, "output", x)
        )
        workflow.add_node(
            NODE_OUTPUT_HISTORY_ANALYZER,
            lambda x: self._output_history_analyzer(x)
        )
        workflow.add_node(
            NODE_PROMPT_ANALYZER,
            lambda x: self._prompt_analyzer(x)
        )
        workflow.add_node(
            NODE_PROMPT_SUGGESTER,
            lambda x: self._prompt_node(
                NODE_PROMPT_SUGGESTER, "suggestions", x
            )
        )

        # Connect nodes
        workflow.add_edge(NODE_PROMPT_DEVELOPER, NODE_PROMPT_EXECUTOR)
        workflow.add_edge(NODE_PROMPT_EXECUTOR, NODE_OUTPUT_HISTORY_ANALYZER)
        workflow.add_edge(NODE_PROMPT_SUGGESTER, NODE_PROMPT_DEVELOPER)

        # Add conditional edges
        workflow.add_conditional_edges(
            NODE_OUTPUT_HISTORY_ANALYZER,
            lambda x: self._should_exit_on_max_age(x),
            {
                "continue": NODE_PROMPT_ANALYZER,
                "rerun": NODE_PROMPT_SUGGESTER,
                END: END
            }
        )
        workflow.add_conditional_edges(
            NODE_PROMPT_ANALYZER,
            lambda x: self._should_exit_on_acceptable_output(x),
            {
                "continue": NODE_PROMPT_SUGGESTER,
                END: END
            }
        )

        # Add optional nodes
        workflow.add_node(
            NODE_PROMPT_INITIAL_DEVELOPER,
            lambda x: self._optional_action(
                "system_message",
                lambda x: self._prompt_node(
                    NODE_PROMPT_INITIAL_DEVELOPER, "system_message", x
                ),
                x
            )
        )
        workflow.add_node(
            NODE_ACCEPTANCE_CRITERIA_DEVELOPER,
            lambda x: self._optional_action(
                "acceptance_criteria",
                lambda x: self._prompt_node(
                    NODE_ACCEPTANCE_CRITERIA_DEVELOPER,
                    "acceptance_criteria",
                    x
                ),
                x
            )
        )

        # Add edges to optional nodes
        workflow.add_edge(START, NODE_PROMPT_INITIAL_DEVELOPER)
        workflow.add_edge(START, NODE_ACCEPTANCE_CRITERIA_DEVELOPER)
        workflow.add_edge(NODE_PROMPT_INITIAL_DEVELOPER, NODE_PROMPT_EXECUTOR)
        workflow.add_edge(NODE_ACCEPTANCE_CRITERIA_DEVELOPER, NODE_PROMPT_EXECUTOR)

        return workflow


    def run_node_graph(self, node: str, state: AgentState) -> AgentState:
        """Run the graph for the specified node with the given state.

        Args:
            node (str): The node name to run.
            state (AgentState): The current state of the agent.

        Returns:
            AgentState: The output state of the agent after invoking the graph.
        """
        self.logger.debug(f"Creating workflow for node: {node}")
        workflow = self._create_workflow_for_node(node)
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "1"}}
        self.logger.debug(f"Invoking graph for node {node} with state: %s", pprint.pformat(state))
        output_state = graph.invoke(state, config)
        self.logger.debug(f"Output state for node {node}: %s", pprint.pformat(output_state))
        return output_state    
    

    def run_meta_prompt_graph(
        self, state: AgentState, recursion_limit: int = 25
    ) -> AgentState:
        """
        Invoke the meta-prompt workflow with the given state and recursion limit.

        This method creates a workflow based on the presence of an initial system
        message, compiles the workflow with a memory saver, and invokes the graph
        with the given state. If a recursion limit is reached, it returns the
        best state found so far.

        Args:
            state (AgentState): The current state of the agent, containing
                necessary context for message formatting.
            recursion_limit (int): The maximum number of recursive calls
                allowed. Defaults to 25.

        Returns:
            AgentState: The output state of the agent after invoking the workflow.
        """
        workflow = self._create_workflow()
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        config = {
            "configurable": {"thread_id": "1"},
            "recursion_limit": recursion_limit,
        }

        try:
            self.logger.debug("Invoking graph with state: %s", pprint.pformat(state))
            output_state = graph.invoke(state, config)
            self.logger.debug("Output state: %s", pprint.pformat(output_state))
            return output_state
        except GraphRecursionError as e:
            self.logger.info("Recursion limit reached. Returning the best state found so far.")
            checkpoint_states = graph.get_state(config)

            if checkpoint_states:
                output_state = checkpoint_states[0]
                return output_state
            else:
                self.logger.info("No checkpoint states found. Returning the input state.")
                return state


    def __call__(
        self, state: AgentState, recursion_limit: int = 25
    ) -> AgentState:
        """Invoke the meta-prompt workflow with the given state and recursion limit.

        Args:
            state (AgentState): The current state of the agent.
            recursion_limit (int): The maximum number of recursive calls allowed.

        Returns:
            AgentState: The output state of the agent after invoking the workflow.
        """
        return self.run_meta_prompt_graph(state, recursion_limit)


    def _optional_action(
        self, target_attribute: str, action: RunnableLike, state: AgentState
    ) -> AgentState:
        """
        Optionally invokes an action if the target attribute is not set or empty.

        Args:
            target_attribute (str): State attribute to be updated.
            action (RunnableLike): Action to be invoked. Defaults to None.
            state (AgentState): Current agent state.

        Returns:
            AgentState: Updated state.
        """
        result = {
            target_attribute: (
                state.get(target_attribute, "")
                if isinstance(state, dict)
                else getattr(state, target_attribute, "")
            )
        }

        if action is not None and not result[target_attribute]:
            result = action(state)

        return result
    

    def _prompt_node(
        self, node: str, target_attribute: str, state: AgentState
    ) -> AgentState:
        """Prompt a specific node with the given state and update the state with the response.

        This method formats messages using the prompt template associated with the node,
        logs the invocation and response, and updates the state with the response content.

        Args:
            node (str): Node identifier to be prompted.
            target_attribute (str): State attribute to be updated with response content.
            state (AgentState): Current agent state with necessary context for message formatting.

        Returns:
            AgentState: Updated state with response content set to the target attribute.
        """

        logger = self.logger.getChild(node)
        formatted_messages = (
            self.prompt_templates[node].format_messages(
                **(state.model_dump() if isinstance(state, BaseModel) else state)
            )
        )

        for message in formatted_messages:
            logger.debug(
                {
                    'node': node,
                    'action': 'invoke',
                    'type': message.type,
                    'message': message.content
                }
            )

        response = self.llms[node].invoke(formatted_messages)
        logger.debug(
            {
                'node': node,
                'action': 'response',
                'type': response.type,
                'message': response.content
            }
        )

        return {target_attribute: response.content}


    def _output_history_analyzer(self, state: AgentState) -> AgentState:
        """
        Analyzes the output history and updates the best output and its age.

        This method checks if the best output is initialized, formats the prompt for
        the output history analyzer, invokes the language model, and updates the
        best output and its age based on the response.

        Args:
            state (AgentState): Current state of the agent with necessary context
                for message formatting.

        Returns:
            AgentState: Updated state with the best output and its age.
        """
        logger = self.logger.getChild(NODE_OUTPUT_HISTORY_ANALYZER)

        if state["best_output"] is None:
            state["best_output"] = state["output"]
            state["best_system_message"] = state["system_message"]
            state["best_output_age"] = 0
            logger.debug("Best output initialized to the current output:\n%s",
                         state["output"])
            return state

        prompt = self.prompt_templates[NODE_OUTPUT_HISTORY_ANALYZER].format_messages(
            **state)

        for message in prompt:
            logger.debug({
                'node': NODE_OUTPUT_HISTORY_ANALYZER,
                'action': 'invoke',
                'type': message.type,
                'message': message.content
            })

        chain = (
            self.prompt_templates[NODE_OUTPUT_HISTORY_ANALYZER] | self.llms[NODE_OUTPUT_HISTORY_ANALYZER] | JsonOutputParser()
        )
        analysis_dict = chain.invoke(state)

        logger.debug({
            'node': NODE_OUTPUT_HISTORY_ANALYZER,
            'action': 'response',
            'message': json.dumps(analysis_dict)
        })

        closer_output_id = analysis_dict["closerOutputID"]

        if (state["best_output"] is None or
            closer_output_id == 2 or
            (self.aggressive_exploration and closer_output_id != 1)):
            result_dict = {
                "best_output": state["output"],
                "best_system_message": state["system_message"],
                "best_output_age": 0
            }
            logger.debug("Best output updated to the current output:\n%s",
                         result_dict["best_output"])
        else:
            result_dict = {
                "output": state["best_output"],
                "system_message": state["best_system_message"],
                "best_output_age": state["best_output_age"] + 1
            }
            logger.debug("Best output age incremented to %s",
                         result_dict["best_output_age"])

        return result_dict


    def _prompt_analyzer(self, state: AgentState) -> AgentState:
        """
        Analyzes the prompt and updates the state with the analysis and 
        acceptance status.

        Args:
            state (AgentState): The current state of the agent, containing 
                necessary context for message formatting.

        Returns:
            AgentState: The updated state of the agent with the analysis 
                and acceptance status.
        """
        logger = self.logger.getChild(NODE_PROMPT_ANALYZER)
        prompt = self.prompt_templates[NODE_PROMPT_ANALYZER].format_messages(
            **state)

        for message in prompt:
            logger.debug({
                'node': NODE_PROMPT_ANALYZER,
                'action': 'invoke',
                'type': message.type,
                'message': message.content
            })

        chain = (
            self.prompt_templates[NODE_PROMPT_ANALYZER] | self.llms[NODE_PROMPT_ANALYZER] | JsonOutputParser()
        )
        result = chain.invoke(state)

        logger.debug({
            'node': NODE_PROMPT_ANALYZER,
            'action': 'response',
            'message': json.dumps(result)
        })

        result_dict = {
            "analysis": json.dumps(result),
            "accepted": result["Accept"] == "Yes"
        }
        logger.debug("Accepted: %s", result_dict["accepted"])

        return result_dict


    def _should_exit_on_max_age(self, state: AgentState) -> str:
        """
        Determines whether to exit the workflow based on the maximum output age.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            str: The decision to continue, rerun, or end the workflow.
        """
        if state["max_output_age"] <= 0:
            return "continue"  # always continue if max age is 0

        if state["best_output_age"] >= state["max_output_age"]:
            return END

        if state["best_output_age"] > 0:
            # skip prompt_analyzer and prompt_suggester, goto prompt_developer
            return "rerun"

        return "continue"


    def _should_exit_on_acceptable_output(self, state: AgentState) -> str:
        """
        Determines whether to exit the workflow based on the acceptance status of 
        the output.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            str: The decision to continue or end the workflow.
        """
        return "continue" if not state["accepted"] else END

