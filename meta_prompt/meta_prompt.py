import typing
import pprint
import logging
from typing import Dict, Any, Callable, List, Union, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel
from .consts import *

class AgentState(BaseModel):
    max_output_age: int = 0
    user_message: Optional[str] = None
    expected_output: Optional[str] = None
    acceptance_criteria: Optional[str] = None
    system_message: Optional[str] = None
    output: Optional[str] = None
    suggestions: Optional[str] = None
    accepted: bool = False
    analysis: Optional[str] = None
    best_output: Optional[str] = None
    best_system_message: Optional[str] = None
    best_output_age: int = 0

class MetaPromptGraph:
    @classmethod
    def get_node_names(cls):
        return META_PROMPT_NODES

    def __init__(self,
                 llms: Union[BaseLanguageModel,
                             Dict[str, BaseLanguageModel]] = {},
                 prompts: Dict[str, ChatPromptTemplate] = {},
                 logger: Optional[logging.Logger] = None,
                 verbose=False):
        self.logger = logger or logging.getLogger(__name__)
        if self.logger is not None:
            if verbose:
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.INFO)

        if isinstance(llms, BaseLanguageModel):
            self.llms: Dict[str, BaseLanguageModel] = {
                node: llms for node in self.get_node_names()}
        else:
            self.llms: Dict[str, BaseLanguageModel] = llms
        self.prompt_templates: Dict[str,
                                    ChatPromptTemplate] = DEFAULT_PROMPT_TEMPLATES.copy()
        self.prompt_templates.update(prompts)

    def _create_workflow(self, including_initial_developer: bool = True) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node(NODE_PROMPT_DEVELOPER,
                          lambda x: self._prompt_node(
                              NODE_PROMPT_DEVELOPER,
                              "system_message",
                              x))
        workflow.add_node(NODE_PROMPT_EXECUTOR,
                          lambda x: self._prompt_node(
                              NODE_PROMPT_EXECUTOR,
                              "output",
                              x))
        workflow.add_node(NODE_OUTPUT_HISTORY_ANALYZER,
                          lambda x: self._output_history_analyzer(x))
        workflow.add_node(NODE_PROMPT_ANALYZER,
                          lambda x: self._prompt_analyzer(x))
        workflow.add_node(NODE_PROMPT_SUGGESTER,
                          lambda x: self._prompt_node(
                              NODE_PROMPT_SUGGESTER,
                              "suggestions",
                              x))

        workflow.add_edge(NODE_PROMPT_DEVELOPER, NODE_PROMPT_EXECUTOR)
        workflow.add_edge(NODE_PROMPT_EXECUTOR, NODE_OUTPUT_HISTORY_ANALYZER)
        workflow.add_edge(NODE_PROMPT_SUGGESTER, NODE_PROMPT_DEVELOPER)

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

        if including_initial_developer:
            workflow.add_node(NODE_PROMPT_INITIAL_DEVELOPER,
                              lambda x: self._prompt_node(
                                  NODE_PROMPT_INITIAL_DEVELOPER,
                                  "system_message",
                                  x))
            workflow.add_edge(NODE_PROMPT_INITIAL_DEVELOPER,
                              NODE_PROMPT_EXECUTOR)
            workflow.set_entry_point(NODE_PROMPT_INITIAL_DEVELOPER)
        else:
            workflow.set_entry_point(NODE_PROMPT_EXECUTOR)

        return workflow

    def __call__(self, state: AgentState, recursion_limit: int = 25) -> AgentState:
        workflow = self._create_workflow(including_initial_developer=(
            state.system_message is None or state.system_message == ""))

        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "1"},
                  "recursion_limit": recursion_limit}

        try:
            self.logger.debug("Invoking graph with state: %s",
                              pprint.pformat(state))

            output_state = graph.invoke(state, config)

            self.logger.debug("Output state: %s", pprint.pformat(output_state))

            return output_state
        except GraphRecursionError as e:
            self.logger.info(
                "Recursion limit reached. Returning the best state found so far.")
            checkpoint_states = graph.get_state(config)

            # if the length of states is bigger than 0, print the best system message and output
            if len(checkpoint_states) > 0:
                output_state = checkpoint_states[0]
                return output_state
            else:
                self.logger.info(
                    "No checkpoint states found. Returning the input state.")

        return state

    def _prompt_node(self, node, target_attribute: str, state: AgentState) -> AgentState:
        logger = self.logger.getChild(node)
        prompt = self.prompt_templates[node].format_messages(
            **state.model_dump())

        for message in prompt:
            logger.debug({'node': node, 'action': 'invoke',
                         'type': message.type, 'message': message.content})
        response = self.llms[node].invoke(
            self.prompt_templates[node].format_messages(**state.model_dump()))
        logger.debug({'node': node, 'action': 'response',
                     'type': response.type, 'message': response.content})

        setattr(state, target_attribute, response.content)
        return state

    def _output_history_analyzer(self, state: AgentState) -> AgentState:
        logger = self.logger.getChild(NODE_OUTPUT_HISTORY_ANALYZER)

        if state.best_output is None:
            state.best_output = state.output
            state.best_system_message = state.system_message
            state.best_output_age = 0

            logger.debug(
                "Best output initialized to the current output:\n%s", state.output)

            return state

        prompt = self.prompt_templates[NODE_OUTPUT_HISTORY_ANALYZER].format_messages(
            **state.model_dump())

        for message in prompt:
            logger.debug({'node': NODE_OUTPUT_HISTORY_ANALYZER, 'action': 'invoke',
                         'type': message.type, 'message': message.content})

        response = self.llms[NODE_OUTPUT_HISTORY_ANALYZER].invoke(prompt)
        logger.debug({'node': NODE_OUTPUT_HISTORY_ANALYZER, 'action': 'response',
                     'type': response.type, 'message': response.content})

        analysis = response.content

        if state.best_output is None or "# Preferred Output ID: B" in analysis:
            state.best_output = state.output
            state.best_system_message = state.system_message
            state.best_output_age = 0

            logger.debug(
                "Best output updated to the current output:\n%s", state.output)
        else:
            state.best_output_age += 1

            logger.debug("Best output age incremented to %s",
                         state.best_output_age)

        return state

    def _prompt_analyzer(self, state: AgentState) -> AgentState:
        logger = self.logger.getChild(NODE_PROMPT_ANALYZER)
        prompt = self.prompt_templates[NODE_PROMPT_ANALYZER].format_messages(
            **state.model_dump())

        for message in prompt:
            logger.debug({'node': NODE_PROMPT_ANALYZER, 'action': 'invoke',
                         'type': message.type, 'message': message.content})

        response = self.llms[NODE_PROMPT_ANALYZER].invoke(prompt)
        logger.debug({'node': NODE_PROMPT_ANALYZER, 'action': 'response',
                     'type': response.type, 'message': response.content})

        state.analysis = response.content
        state.accepted = "Accept: Yes" in response.content

        logger.debug("Accepted: %s", state.accepted)

        return state

    def _should_exit_on_max_age(self, state: AgentState) -> str:
        if state.max_output_age <= 0:
            # always continue if max age is 0
            return "continue"
        
        if state.best_output_age >= state.max_output_age:
            return END
        
        if state.best_output_age > 0:
            # skip prompt_analyzer and prompt_suggester, goto prompt_developer
            return "rerun" 
        
        return "continue"

    def _should_exit_on_acceptable_output(self, state: AgentState) -> str:
        return "continue" if not state.accepted else END