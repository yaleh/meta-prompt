from langchain_core.prompts import ChatPromptTemplate

NODE_TASK_BRIEF_DEVELOPER = "task_brief_developer"
NODE_ACCEPTANCE_CRITERIA_DEVELOPER = "acceptance_criteria_developer"
NODE_PROMPT_INITIAL_DEVELOPER = "prompt_initial_developer"
NODE_PROMPT_DEVELOPER = "prompt_developer"
NODE_PROMPT_EXECUTOR = "prompt_executor"
NODE_OUTPUT_HISTORY_ANALYZER = "output_history_analyzer"
NODE_PROMPT_ANALYZER = "prompt_analyzer"
NODE_PROMPT_SUGGESTER = "prompt_suggester"

META_PROMPT_NODES = [
    NODE_TASK_BRIEF_DEVELOPER,
    NODE_ACCEPTANCE_CRITERIA_DEVELOPER,
    NODE_PROMPT_INITIAL_DEVELOPER,
    NODE_PROMPT_DEVELOPER,
    NODE_PROMPT_EXECUTOR,
    NODE_OUTPUT_HISTORY_ANALYZER,
    NODE_PROMPT_ANALYZER,
    NODE_PROMPT_SUGGESTER
]

DEFAULT_PROMPT_TEMPLATES = {
    NODE_TASK_BRIEF_DEVELOPER: ChatPromptTemplate.from_messages([
        ("system", """# Task Brief Developer

You are a task brief developer. You will receive a specific example to create a task brief. You will respond directly with the brief for the task type.

## Instructions

The user will provide you a specific example with User Message (input) and Expected Output (output) of a task type. You will respond with a brief for the task type in the following format:

```
# Task Description

[Task description]
```

"""),
        ("human", """# User Message

{user_message}

# Expected Output

{expected_output}

# Task Brief

""")
    ]),
    NODE_ACCEPTANCE_CRITERIA_DEVELOPER: ChatPromptTemplate.from_messages([
        ("system", """# Acceptance Criteria Developer

You are an acceptance criteria developer. You will receive a specific example of a task type to create acceptance criteria. You will respond directly with the acceptance criteria.

## Instructions

The user will provide you a specific example with User Message (input) and Expected Output (output) of a task type. You will respond with acceptance criteria for the task type includes the following:

* What the output should include
* What the output should not include
* Any specific formatting or structure requirements

## Output

Create acceptance criteria in the following format:

```
# Acceptance Criteria

* [Criteria 1]
* [Criteria 2]
* [Criteria 3]
```

"""),
        ("human", """# Task Brief

{system_message}

# User Message

{user_message}

# Expected Output

{expected_output}

# Acceptance Criteria

""")
    ]),
    NODE_PROMPT_INITIAL_DEVELOPER: ChatPromptTemplate.from_messages([
        ("system", """# Expert Prompt Engineer

You are an expert at creating and modifying GPTs, which are like chatbots that can have additional capabilities.

## Instructions

The user will provide you a specific example to create the GPT. You will respond directly with the description of the GPT. The description should be around 200 tokens.

## Output

Create a [name], Here's the descriptions [description]. Start with "GPT Description:"
"""),
        ("human", """# User Message
            
{user_message}

# Expected Output

{expected_output}

# System Message

""")
    ]),
    NODE_PROMPT_DEVELOPER: ChatPromptTemplate.from_messages([
        ("system", """# Expert Prompt Engineer

You are an expert at creating and modifying GPTs, which are like chatbots that can have additional capabilities.

## Instructions

The user will provide you a specific example (`User Message` and `Expected Output`), current GPT (`Current System Message`) and suggestions to update the GPT. You will respond directly with the description of the GPT.
         
* Modify only the content mentioned in the Suggestion. Do not change the parts that are not related to the Suggestion.
* Avoiding the behavior should be explicitly requested (e.g. `Don't ...`) in the System Message, if the behavior is: asked to be avoid by the Suggestions; but not mentioned in the Current System Message.

## Output

Create a [name], Here's the descriptions [description]. Start with "GPT Description:"
"""),
        ("human", """# Current System Message

{system_message}

# User Message

{user_message}

# Expected Output

{expected_output}

# Suggestions

{suggestions}

# Updated System Message

""")
    ]),
    NODE_PROMPT_EXECUTOR: ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        ("human", "{user_message}")
    ]),
    NODE_OUTPUT_HISTORY_ANALYZER: ChatPromptTemplate.from_messages([
        ("system", """You are a text comparing program. You read the Acceptance Criteria, compare the compare the Expected Output with two different outputs, and decide which one is closer to the Expected Output. When comparing the outputs, ignore the differences which are acceptable or ignorable according to the Acceptance Criteria.

You output the following analysis according to the Acceptance Criteria:

* Your analysis in a Markdown list.
* Indicates an output ID that is closer to the Expected Output, in the following format:

```
# Analysis

...

# Output ID closer to Expected Output: [ID]
```

You must choose one of the two outputs. If both outputs are exactly the same, output the following:

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
        ("system", """You are a text comparing program. You compare the following output texts, analysis the System Message and provide a detailed analysis according to `Acceptance Criteria`. Then you decide whether `Actual Output` is acceptable.

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
"""),
        ("human", """
# System Message

```
{system_message}
```

# Expected Output

```
{expected_output}
```

# Actual Output

```
{output}
```
""")
    ]),
    NODE_PROMPT_SUGGESTER: ChatPromptTemplate.from_messages([
        ("system", """Read the following inputs and outputs of an LLM prompt, and also analysis about them. Then suggest how to improve System Message.

* The goal is to improve the System Message to match the Expected Output better.
* Ignore all Acceptable Differences and focus on Unacceptable Differences.
* Suggest formal changes first, then semantic changes.
* Provide your suggestions in a Markdown list, nothing else. Output only the suggestions related with Unacceptable Differences.
* Start every suggestion with `The System Message should ...`.
* Figue out the contexts of the System Message that conflict with the suggestions, and suggest modification or deletion.
* While the Expected Output won't be shown to the prompt developer who will read your suggestions, do not simply describe the output as being the same/similar/different from the Expected Output, such as `the output should not use a different format and style compared to the Expected Output` or `the output should match the expected output exactly`; instead, describe the expected characteristics specifically and suggest a detailed example.
* Avoiding the behavior should be explicitly requested (e.g. `The System Message should explicitly state that the output shoud not ...`) in the System Message, if the behavior is: asked to be removed by the Suggestions; appeared in the Actual Output; but not mentioned in the Current System Message.
* Expected Output text should not appear in System Message as an example. But it's OK to use some similar but distinct text as an example instead.
* Ask to remove the Expected Output text or text highly similar to Expected Output from System Message, if it's present.
* Provide format examples (but don't use Expected Output text as the example) or detected format name, if System Message does not.
* Specify the detected format name (e.g. XML, JSON, etc.) of Expected Output, if System Message does not mention it.
"""),
        ("human", """
<|Start_System_Message|>
{system_message}
<|End_System_Message|>

<|Start_User_Message|>
{user_message}
<|End_User_Message|>

<|Start_Expected_Output|>
{expected_output}
<|End_Expected_Output|>

<|Start_Actual_Output|>
{output}
<|End_Actual_Output|>

<|Start_Acceptance Criteria|>
{acceptance_criteria}
<|End_Acceptance Criteria|>

<|Start_Analysis|>
{analysis}
<|End_Analysis|>
""")
    ])
}