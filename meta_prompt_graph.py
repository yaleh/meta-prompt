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
    NODE_PROMPT_INITIAL_DEVELOPER = "prompt_initial_developer"
    NODE_PROMPT_DEVELOPER = "prompt_developer"
    NODE_PROMPT_EXECUTOR = "prompt_executor"
    NODE_OUTPUT_HISTORY_ANALYZER = "output_history_analyzer"
    NODE_PROMPT_ANALYZER = "prompt_analyzer"
    NODE_PROMPT_SUGGESTER = "prompt_suggester"

    DEFAULT_PROMPT_TEMPLATES = {
        NODE_PROMPT_INITIAL_DEVELOPER: ChatPromptTemplate.from_messages([
            ("system", """# Expert Prompt Engineer

You are an expert prompt engineer tasked with creating system messages for AI
assistants.

## Instructions

1. Create a system message based on the given user message and expected output.
2. Ensure the system message can handle similar user messages.
3. Output only the system message, without any additional content.
4. Expected Output text should not appear in System Message as an example. But
   it's OK to use some similar text as an example instead.
5. Format the system message well, with no more than 80 characters per line
   (except for raw text).

## Output

Provide only the system message, adhering to the above guidelines.
"""),
            ("human", """# User Message
             
{user_message}

# Expected Output

{expected_output}
""")
        ]),
        NODE_PROMPT_DEVELOPER: ChatPromptTemplate.from_messages([
            ("system", """# Expert Prompt Engineer

You are an expert prompt engineer tasked with updating system messages for AI
assistants. You Update System Message according to Suggestions, to improve
Output and match Expected Output more closely.

## Instructions

1. Update the system message based on the given Suggestion, User Message, and
   Expected Output.
2. Ensure the updated system message can handle similar user messages.
3. Modify only the content mentioned in the Suggestion. Do not change the
   parts that are not related to the Suggestion.
4. Output only the updated system message, without any additional content.
5. Expected Output text should not appear in System Message as an example. But
   it's OK to use some similar text as an example instead.
   * Remove the Expected Output text or text highly similar to Expected Output
     from System Message, if it's present.
6. Format the system message well, with no more than 80 characters per line
   (except for raw text).

## Output

Provide only the updated System Message, adhering to the above guidelines.
"""),
            ("human", """# Current system message

{system_message}

# User Message

{user_message}

# Expected Output

{expected_output}

# Suggestions

{suggestions}
""")
        ]),
        NODE_PROMPT_EXECUTOR: ChatPromptTemplate.from_messages([
            ("system", "{system_message}"),
            ("human", "{user_message}")
        ]),
        NODE_OUTPUT_HISTORY_ANALYZER: ChatPromptTemplate.from_messages([
            ("system", """You are a text comparing program. You read the Acceptance Criteria, compare the
compare the exptected output with two different outputs, and decide which one is
more similar to the expected output.

You output the following analysis according to the Acceptance Criteria:

* Your analysis in a Markdown list.
* The ID of the output that is more similar to the Expected Output as Preferred
    Output ID, with the following format:
    
```
# Analysis

...

# Preferred Output ID: [ID]
```

If both outputs are equally similar to the expected output, output the following:

```
# Analysis

...

# Draw
```
"""),
            ("human", """
# Output ID: A

```
{best_output}
```

# Output ID: B

```
{output}
```

# Acceptance Criteria

{acceptance_criteria}

# Expected Output

```
{expected_output}
```
""")
        ]),
        NODE_PROMPT_ANALYZER: ChatPromptTemplate.from_messages([
            ("system", """
You are a text comparing program. You compare the following output texts and provide a
detailed analysis according to `Acceptance Criteria`. Then you decide whether `Actual Output`
is acceptable.

# Expected Output

```
{expected_output}
```

# Actual Output

```
{output}
```

----

Provide your analysis in the following format:

```
- Acceptable Differences: [List acceptable differences succinctly]
- Unacceptable Differences: [List unacceptable differences succinctly]
- Accept: [Yes/No]
```

* Compare Expected Output and Actual Output with the guidance of Accept Criteria.
* Only set 'Accept' to 'Yes', if Accept Criteria are all met. Otherwise, set 'Accept' to 'No'.
* List only the acceptable differences according to Accept Criteria in 'acceptable Differences' section.
* List only the unacceptable differences according to Accept Criteria in 'Unacceptable Differences' section.

# Acceptance Criteria

```
{acceptance_criteria}
```
""")
        ]),
        NODE_PROMPT_SUGGESTER: ChatPromptTemplate.from_messages([
            ("system", """
Read the following inputs and outputs of an LLM prompt, and also analysis about them.
Then suggest how to improve System Prompt.

System Prompt:
```
{system_message}
```
User Message:
```
{user_message}
```
Expected Output: 
```
{expected_output}
```
Actual Output: 
```
{output}
```

Acceptance Criteria:
```
{acceptance_criteria}
```

Analysis:
```
{analysis}
```

* The goal is to improve the System Prompt to match the Expected Output better.
* Ignore all Acceptable Differences and focus on Unacceptable Differences.
* Suggest formal changes first, then semantic changes.
* Provide your suggestions in a Markdown list, nothing else. Output only the
    suggestions related with Unacceptable Differences.
    * Use `... should ...` to clearly state the desired output.
    * Figue out the contexts of the System Message that conflict with the suggestions,
    and suggest modification or deletion.
* Expected Output text should not appear in System Message as an example. But
    it's OK to use some similar text as an example instead.
    * Ask to remove the Expected Output text or text highly similar to Expected Output
    from System Message, if it's present.
* Provide format examples or detected format name, if System Message does not.
    * Specify the detected format name (e.g. XML, JSON, etc.) of Expected Output, if
    System Message does not mention it.
""")
        ])
    }

    @classmethod
    def get_node_names(cls):
        return [
            cls.NODE_PROMPT_INITIAL_DEVELOPER,
            cls.NODE_PROMPT_DEVELOPER,
            cls.NODE_PROMPT_EXECUTOR,
            cls.NODE_OUTPUT_HISTORY_ANALYZER,
            cls.NODE_PROMPT_ANALYZER,
            cls.NODE_PROMPT_SUGGESTER
        ]

    def __init__(self,
                 llms: Dict[str, BaseLanguageModel] = {},
                 prompts: Dict[str, ChatPromptTemplate] = {},
                 verbose = False):
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.llms: Dict[str, BaseLanguageModel] = llms
        self.prompt_templates: Dict[str, ChatPromptTemplate] = self.DEFAULT_PROMPT_TEMPLATES.copy()
        self.prompt_templates.update(prompts)

        # create workflow
        self.workflow = StateGraph(AgentState)

        self.workflow.add_node(self.NODE_PROMPT_INITIAL_DEVELOPER,
                               lambda x: self._prompt_node(
                                   self.NODE_PROMPT_INITIAL_DEVELOPER,
                                   "system_message",
                                   x))
        self.workflow.add_node(self.NODE_PROMPT_DEVELOPER,
                               lambda x: self._prompt_node(
                                   self.NODE_PROMPT_DEVELOPER,
                                   "system_message",
                                   x))
        self.workflow.add_node(self.NODE_PROMPT_EXECUTOR,
                               lambda x: self._prompt_node(
                                   self.NODE_PROMPT_EXECUTOR,
                                   "output",
                                   x))
        self.workflow.add_node(self.NODE_OUTPUT_HISTORY_ANALYZER,
                               lambda x: self._output_history_analyzer(x))
        self.workflow.add_node(self.NODE_PROMPT_ANALYZER,
                               lambda x: self._prompt_analyzer(x))
        self.workflow.add_node(self.NODE_PROMPT_SUGGESTER,
                               lambda x: self._prompt_node(
                                   self.NODE_PROMPT_SUGGESTER,
                                   "suggestions",
                                   x))

        self.workflow.set_entry_point(self.NODE_PROMPT_INITIAL_DEVELOPER)

        self.workflow.add_edge(self.NODE_PROMPT_INITIAL_DEVELOPER, self.NODE_PROMPT_EXECUTOR)
        self.workflow.add_edge(self.NODE_PROMPT_DEVELOPER, self.NODE_PROMPT_EXECUTOR)
        self.workflow.add_edge(self.NODE_PROMPT_EXECUTOR, self.NODE_OUTPUT_HISTORY_ANALYZER)
        self.workflow.add_edge(self.NODE_PROMPT_SUGGESTER, self.NODE_PROMPT_DEVELOPER)

        self.workflow.add_conditional_edges(
            self.NODE_OUTPUT_HISTORY_ANALYZER,
            lambda x: self._should_exit_on_max_age(x),
            {
                "continue": self.NODE_PROMPT_ANALYZER,
                "rerun": self.NODE_PROMPT_SUGGESTER,
                END: END
            }
        )

        self.workflow.add_conditional_edges(
            self.NODE_PROMPT_ANALYZER,
            lambda x: self._should_exit_on_acceptable_output(x),
            {
                "continue": self.NODE_PROMPT_SUGGESTER,
                END: END
            }
        )

    def __call__(self, state: AgentState, recursion_limit: int = 25) -> AgentState:
        memory = MemorySaver()
        graph = self.workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "1"}, "recursion_limit": recursion_limit}

        try:
            self.logger.debug("Invoking graph with state: %s", pprint.pformat(state))

            output_state = graph.invoke(state, config)

            self.logger.debug("Output state: %s", pprint.pformat(output_state))

            return output_state
        except GraphRecursionError as e:
            self.logger.info("Recursion limit reached. Returning the best state found so far.")
            checkpoint_states = graph.get_state(config)

            # if the length of states is bigger than 0, print the best system message and output
            if len(checkpoint_states) > 0:
                output_state = checkpoint_states[0]
                return output_state
            else:
                self.logger.info("No checkpoint states found. Returning the input state.")
            
        return state

    def _prompt_node(self, node, target_attribute: str, state: AgentState) -> AgentState:
        prompt = self.prompt_templates[node].format_messages(**state.model_dump())

        self.logger.debug("Invoking %s with prompt: %s", node, pprint.pformat(prompt))
        response = self.llms[node].invoke(self.prompt_templates[node].format_messages(**state.model_dump()))
        self.logger.debug("Response: %s", pprint.pformat(response.content))
        
        setattr(state, target_attribute, response.content)
        return state

    def _output_history_analyzer(self, state: AgentState) -> AgentState:
        if state.best_output is None:
            state.best_output = state.output
            state.best_system_message = state.system_message
            state.best_output_age = 0

            self.logger.debug("Best output initialized to the current output: \n %s", state.output)

            return state

        prompt = self.prompt_templates[self.NODE_OUTPUT_HISTORY_ANALYZER].format_messages(**state.model_dump())

        self.logger.debug("Invoking %s with prompt: %s",
                          self.NODE_OUTPUT_HISTORY_ANALYZER, 
                          pprint.pformat(prompt))
        response = self.llms[self.NODE_OUTPUT_HISTORY_ANALYZER].invoke(prompt)
        self.logger.debug("Response: %s", pprint.pformat(response.content))

        analysis = response.content

        if state.best_output is None or "# Preferred Output ID: B" in analysis:
            state.best_output = state.output
            state.best_system_message = state.system_message
            state.best_output_age = 0

            self.logger.debug("Best output updated to the current output: \n %s", state.output)
        else:
            state.best_output_age += 1

            self.logger.debug("Best output age incremented to %s", state.best_output_age)

        return state

    def _prompt_analyzer(self, state: AgentState) -> AgentState:
        prompt = self.prompt_templates[self.NODE_PROMPT_ANALYZER].format_messages(**state.model_dump())

        self.logger.debug("Invoking %s with prompt: %s",
                          self.NODE_PROMPT_ANALYZER,
                          pprint.pformat(prompt))
        response = self.llms[self.NODE_PROMPT_ANALYZER].invoke(prompt)
        self.logger.debug("Response: %s", pprint.pformat(response.content))

        state.analysis = response.content
        state.accepted = "Accept: Yes" in response.content

        self.logger.debug("Accepted: %s", state.accepted)

        return state

    def _should_exit_on_max_age(self, state: AgentState) -> str:
        if state.max_output_age <= 0 or state.best_output_age < state.max_output_age:
            return "continue"
        else:
            return "rerun"

    def _should_exit_on_acceptable_output(self, state: AgentState) -> str:
        return "continue" if not state.accepted else END