from langchain_core.prompts import ChatPromptTemplate

# NODE_TASK_BRIEF_DEVELOPER = "task_brief_developer"
NODE_ACCEPTANCE_CRITERIA_DEVELOPER = "acceptance_criteria_developer"
NODE_PROMPT_INITIAL_DEVELOPER = "prompt_initial_developer"
NODE_PROMPT_DEVELOPER = "prompt_developer"
NODE_PROMPT_EXECUTOR = "prompt_executor"
NODE_OUTPUT_HISTORY_ANALYZER = "output_history_analyzer"
NODE_PROMPT_ANALYZER = "prompt_analyzer"
NODE_PROMPT_SUGGESTER = "prompt_suggester"

META_PROMPT_NODES = [
#     NODE_TASK_BRIEF_DEVELOPER,
    NODE_ACCEPTANCE_CRITERIA_DEVELOPER,
    NODE_PROMPT_INITIAL_DEVELOPER,
    NODE_PROMPT_DEVELOPER,
    NODE_PROMPT_EXECUTOR,
    NODE_OUTPUT_HISTORY_ANALYZER,
    NODE_PROMPT_ANALYZER,
    NODE_PROMPT_SUGGESTER
]

DEFAULT_PROMPT_TEMPLATES = {
#     NODE_TASK_BRIEF_DEVELOPER: ChatPromptTemplate.from_messages([
#         ("system", """# Task Brief Developer

# You are a task brief developer. You will receive a specific example to create a task brief. You will respond directly with the brief for the task type.

# ## Instructions

# The user will provide you a specific example with User Message (input) and Expected Output (output) of a task type. You will respond with a brief for the task type in the following format:

# ```
# # Task Description

# [Task description]
# ```

# """),
#         ("human", """# User Message

# {user_message}

# # Expected Output

# {expected_output}

# # Task Brief

# """)
#     ]),

    NODE_ACCEPTANCE_CRITERIA_DEVELOPER: ChatPromptTemplate.from_messages([
        ("system", """# Acceptance Criteria Developer

You are an acceptance criteria developer. You will receive a specific example of a task type to create acceptance criteria. You will respond directly with the acceptance criteria.

## Instructions

The user will provide you a specific example with User Message (input) and Expected Output (output) of a task type. You will respond with acceptance criteria for the task type, by comparing with Expected Output (which may be referenced as EO), includes the following:

* What the output should include
* What the output should not include
* Language requirements
* Formatting requirements
* Structure requirements
* Style requirements
* Any specific requirements

## Output

Create acceptance criteria in the following format:

```
# Acceptance Criteria
         
* [Criteria 1]
* [Criteria 2]
* ...
* Unacceptable differences (compared with EO):
  * ...
* Acceptable differences (compared with EO):
  * ...
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
        ("system", """{{
"task_description": "You are a text comparing program. Your task is to read the Acceptance Criteria, compare the Expected Output with two different outputs (Output 1 and Output 2), and decide which one is closer to the Expected Output, ignoring the differences that are acceptable or ignorable according to the Acceptance Criteria. Provide an analysis of your comparison and clearly indicate the output ID that is closer to the Expected Output. Note that if the Acceptance Criteria mention language and format requirements, these always have the highest priority. Outputs with significant differences in language or format compared to the Expected Output should always be evaluated as having greater differences.",
"requirements": [
    "Read and understand the provided Acceptance Criteria carefully.",
    "Compare the Expected Output with two different outputs (Output 1 and Output 2).",
    "Ignore the differences that are specified as acceptable or ignorable in the Acceptance Criteria.",
    "Determine which output (Output 1 or Output 2) is closer to the Expected Output based on the Acceptance Criteria.",
    "Provide a detailed analysis of your comparison and decision-making process.",
    "Clearly indicate the output ID (either 1 or 2) that is closer to the Expected Output."
],
"output_format": {{
    "type": "object",
    "properties": {{
    "analysis": {{
        "type": "string",
        "description": "A detailed analysis explaining the comparison and decision-making process based on the Acceptance Criteria."
    }},
    "closerOutputID": {{
        "type": "integer",
        "description": "The output ID (1 or 2) that is closer to the Expected Output, or 0 if both outputs are equally close."
    }}
    }},
    "required": [
    "analysis",
    "closerOutputID"
    ]
}},
"output_example": {{
    "analysis": "The Acceptance Criteria specified that the output should be in English and follow a specific JSON format. Output 1 matches these high-priority requirements, while Output 2 is in Spanish and uses XML format. Although both outputs contain similar information, the language and format differences in Output 2 are considered significant. Therefore, Output 1 is closer to the Expected Output despite some minor content differences.",
    "closerOutputID": 1
}},

"evaluation_criteria": [
    "The analysis should demonstrate a clear understanding of the Acceptance Criteria, with the highest priority given to language and format requirements if specified.",
    "The comparison should accurately identify and ignore acceptable or ignorable differences, while emphasizing significant language or format discrepancies.",
    "The decision should be based on a thorough analysis of the outputs in relation to the Expected Output, prioritizing language and format matching when required.",
    "The output ID indicated as closer to the Expected Output should align with the analysis, reflecting the importance of language and format requirements."
],
"error_handling": [
    "If the Acceptance Criteria are unclear or contradictory, provide an analysis explaining the ambiguity and suggest possible interpretations.",
    "If neither output is closer to the Expected Output, provide an analysis explaining why and use \"closerOutputID\": 0."
],
"ethical_considerations": [
    "Ensure that the comparison process is unbiased and solely based on the Acceptance Criteria.",
    "Do not introduce personal opinions or preferences into the analysis."
],
"conclusion": "Confirm that your output adheres to the specified language and format, includes a detailed analysis, and clearly indicates the closer output ID based on the Acceptance Criteria."
}}
"""),
        ("human", """<|Start_Output_ID_1|>{best_output}<|End_Output_ID_1|>
<|Start_Output_ID_2|>{output}<|End_Output_ID_2|>
<|Start_Acceptance_Criteria|>{acceptance_criteria}<|End_Acceptance_Criteria|>
<|Start_Expected_Output|>{expected_output}<|End_Expected_Output|>
""")
    ]),
    NODE_PROMPT_ANALYZER: ChatPromptTemplate.from_messages([
        ("system", """**TASK:** Compare the Expected Output with the Actual Output according to the Acceptance Criteria. Provide a JSON output with your analysis.

**Requirements:**
- Compare Expected and Actual Outputs strictly following the Acceptance Criteria.
- Set `Accept` to "Yes" only if all criteria are met; otherwise, set it to "No."
- List acceptable and unacceptable differences based on the criteria.

**Output Format:** JSON with:
- `Accept: (Yes/No)`
- `Acceptable Differences: []`
- `Unacceptable Differences: []`

**Example Output:**
```json
{{
    "Accept": "No",
    "Acceptable Differences": [
        "Spelling variations: 'colour' vs 'color'"
    ],
    "Unacceptable Differences": [
        "Missing section: 'Conclusion'",
        "Incorrect date format: '2023/10/12' vs '12-10-2023'"
    ]
}}
```

# Acceptance Criteria

{acceptance_criteria}
"""),
        ("human", """# Expected Output

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