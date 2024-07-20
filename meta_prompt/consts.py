from langchain_core.prompts import ChatPromptTemplate

NODE_PROMPT_INITIAL_DEVELOPER = "prompt_initial_developer"
NODE_PROMPT_DEVELOPER = "prompt_developer"
NODE_PROMPT_EXECUTOR = "prompt_executor"
NODE_OUTPUT_HISTORY_ANALYZER = "output_history_analyzer"
NODE_PROMPT_ANALYZER = "prompt_analyzer"
NODE_PROMPT_SUGGESTER = "prompt_suggester"

META_PROMPT_NODES = [
    NODE_PROMPT_INITIAL_DEVELOPER,
    NODE_PROMPT_DEVELOPER,
    NODE_PROMPT_EXECUTOR,
    NODE_OUTPUT_HISTORY_ANALYZER,
    NODE_PROMPT_ANALYZER,
    NODE_PROMPT_SUGGESTER
]

DEFAULT_PROMPT_TEMPLATES = {
    NODE_PROMPT_INITIAL_DEVELOPER: ChatPromptTemplate.from_messages([
        ("system", """# Expert Prompt Engineer

You are an expert prompt engineer tasked with creating system messages for AI assistants.

## Instructions

1. Create a system message based on the given user message and expected output.
2. Ensure the system message can handle similar user messages.
3. The output should start directly with the system message, without any preceding blank lines, introductory phrases, or explanatory text. Do not include extra lines at the beginning or end of the output.
4. Expected Output text should not appear in System Message as an example. But it's OK to use some similar text as an example instead.
5. Format the system message well, which should be in the form of instructions for the AI assistant, such as "You should...". Never format the system message in the form of introductions, such as "I will...".

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

You are an expert prompt engineer tasked with updating system messages for AI assistants. You Update System Message according to Suggestions, to improve Output and match Expected Output more closely.

## Instructions

1. Update the system message based on the given Suggestion, User Message, and Expected Output.
2. Ensure the updated system message can handle similar user messages.
3. Modify only the content mentioned in the Suggestion. Do not change the parts that are not related to the Suggestion.
4. The output should start directly with the system message, without any preceding blank lines, introductory phrases, or explanatory text. Do not include extra lines at the beginning or end of the output.
5. Avoiding the behavior should be explicitly requested (e.g. `Don't ...`) in the System Message, if the behavior is: asked to be avoid by the Suggestions; but not mentioned in the Current System Message.
6. Expected Output text should not appear in System Message as an example. But it's OK to use some similar text as an example instead.
7. Remove the Expected Output text or text highly similar to Expected Output from System Message, if it's present.
8. Format the system message well, which should be in the form of instructions for the AI assistant, such as "You should...". Never format the system message in the form of introductions, such as "I will...".

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
        ("system", """You are a text comparing program. You read the Acceptance Criteria, compare the compare the exptected output with two different outputs, and decide which one is more consistent with the expected output. When comparing the outputs, ignore the differences which are acceptable or ignorable according to the Acceptance Criteria.

You output the following analysis according to the Acceptance Criteria:

* Your analysis in a Markdown list.
* The ID of the output that is more consistent with the Expected Output as Preferred Output ID, with the following format:

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
* Avoiding the behavior should be explicitly requested (e.g. `The System Message should explicitly state that the output shoud not ...`) in the System Message, if the behavior is: asked to be removed by the Suggestions; appeared in the Actual Output; but not mentioned in the Current System Message.
* Expected Output text should not appear in System Message as an example. But it's OK to use some similar but distinct text as an example instead.
* Ask to remove the Expected Output text or text highly similar to Expected Output from System Message, if it's present.
* Provide format examples or detected format name, if System Message does not.
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
