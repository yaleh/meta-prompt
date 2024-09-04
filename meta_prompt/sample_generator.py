import json
from openai import BadRequestError
import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import YamlOutputParser

# Define prompt strings as constants
DESCRIPTION_PROMPT = [
    ("system", """Given the JSON example(s) for a task type:
     
{raw_example}

Provide a concise description of the task type, including the format and style
of the input and output. If there are multiple examples, provide an overall
description and ignore unique parts.

Format your response as follows:
Task Description: [Your description here]
""")
]

DESCRIPTION_UPDATING_PROMPT = [
    ("system", """Given the task type description and suggestions, update the task type description according to the suggestions.
     
1. Input Information:
   - You will receive a task type description and suggestions for updating the description.
   - Carefully read and understand the provided information.

2. Task Analysis:
   - Identify the core elements and characteristics of the task.
   - Consider possible generalization dimensions such as task domain, complexity, input/output format, application scenarios, etc.

3. Update Task Description:
   - Apply the suggestions to update the task description. Don't change anything that is not suggested.
   - Ensure the updated description is clear, specific, and directly related to the task.

4. Output Format:
   - Format your response as follows:
     
Task Description: [Your updated description here]
     
   - Output the updated `Task Description` only. Don't output anything else.

5. Completeness Check:
   - Ensure all important aspects of the task description are covered.
   - Check for any missing key information or dimensions.

6. Quantity Requirement:
   - Provide at least 5 specification suggestions across different dimensions.
"""),
    ("user", """***Task Description:***

{description}

***Suggestions:***

{suggestions}

""")
]


SPECIFICATION_SUGGESTIONS_PROMPT = [
    ("system", """{{
  "prompt": "Generate suggestions to narrow the task scope for a given task type and example:\n\n1. Analyze the task description and input/output examples.\n2. Identify 3~5 relevant dimensions (e.g., purpose, input/output format, language, steps, criteria, constraints).\n3. Create 3~5 actionable suggestions (no more than 20 words for each) to narrow the task scope based on the above dimensions. Make sure the suggestions are compatible with the provided example.\n4. Start each suggestion with a verb.\n5. Output in JSON format, following `output_format`.\n", 
  "output_format": "{{\n  \"dimensions\": [\n    {{ \"dimension\": \"...\" }},\n    {{ \"dimension\": \"...\" }}\n  ],\n  \"suggestions\": [\n    {{ \"suggestion\": \"...\" }},\n    {{ \"suggestion\": \"...\" }}\n  ]\n}}\n", 
  "task_description": "\n{description}\n", 
  "examples": "\n{raw_example}\n"
}}
""")
]

GENERALIZATION_SUGGESTIONS_PROMPT = [
    ("system", """{{
  "prompt": "Generate task generalization suggestions for a given task type and example:\n\n1. Analyze the task description and input/output examples.\n2. Identify 3~5 relevant dimensions (e.g., purpose, input/output format, language, steps, criteria, constraints).\n3. Create 3~5 actionable suggestions (no more than 20 words for each) to expand the scope of the task based on the above dimensions. Make sure the suggestions are compatible with the provided example.\n4. Start each suggestion with a verb.\n5. Output in JSON format, following `output_format`.\n", 
  "output_format": "{{\n  \"dimensions\": [\n    {{ \"dimension\": \"...\" }},\n    {{ \"dimension\": \"...\" }}\n  ],\n  \"suggestions\": [\n    {{ \"suggestion\": \"...\" }},\n    {{ \"suggestion\": \"...\" }}\n  ]\n}}\n", 
  "task_description": "\n{description}\n", 
  "examples": "\n{raw_example}\n"
}}
""")
]

INPUT_ANALYSIS_PROMPT = [
    ("system", """For the specific task type, analyze the possible task inputs across multiple dimensions.
     
Conduct a detailed analysis and enumerate:

1. Core Attributes: Identify the fundamental properties or characteristics of this input type.
1. Variation Dimensions: For each dimension that may vary, specify:
   - Dimension name
   - Possible range of values or options
   - Impact on input nature or task difficulty
1. Constraints: List any rules or limitations that must be adhered to.
1. Edge Cases: Describe extreme or special scenarios that may test the robustness of task processing.
1. External Factors: Enumerate factors that might influence input generation or task completion.
1. Potential Extensions: Propose ways to expand or modify this input type to create new variants.

Format your response as follows:
Input Analysis: [Your analysis here]
"""),
    ("user", """Task Description:

{description}

""")
]

BRIEFS_PROMPT = [
    ("system", """{{
  "prompt": "Given the task type description, and input analysis, generate descriptions for {generating_batch_size} new examples with detailed attributes based on this task type. But don't provide any detailed task output.\n\nUse the input analysis to create diverse and comprehensive example briefs that cover various input dimensions and attribute ranges.\n\nFormat your response as a JSON object following `output_format`.",
  "output_format": "{{
    "new_example_briefs": [
      {{
        "example_brief": "..."
      }},
      {{
        "example_brief": "..."
      }},
      ...
    ]
  }},
  "task_description": "{description}",
  "input_analysis": "{input_analysis}",
  "generating_batch_size": "{generating_batch_size}"
}}
""")
]

EXAMPLES_FROM_BRIEFS_PROMPT = [
    ("system", """{{
  "prompt": "Given the task type description, brief descriptions for new examples, and JSON example(s), generate {generating_batch_size} more input/output examples for this task type, strictly following the brief descriptions and task type description. Ensure that the new examples are consistent with the brief descriptions and do not introduce any new information not present in the briefs. Output in JSON format, following `output_format`. Validate the generated new examples with the task type description and brief descriptions.",
  "output_format": "{{
    "examples": [
      {{
        "input": "...",
        "output": "..."
      }},
      {{
        "input": "...",
        "output": "..."
      }},
      ...
    ]
  }},
  "task_description": "{description}",
  "new_example_briefs": {new_example_briefs},
  "raw_example": "{raw_example}"
}}
""")
]

EXAMPLES_DIRECTLY_PROMPT = [
    ("system", """{{
  "prompt": "Given the task type description, and input/output example(s), generate {generating_batch_size} new input/output examples for this task type. Output in JSON format, following `output_format`.",
  "output_format": "{{
    "examples": [
      {{
        "input": "...",
        "output": "..."
      }},
      {{
        "input": "...",
        "output": "..."
      }},
      ...
    ]
  }},
  "task_description": "{description}",
  "examples": "{raw_example}"
}}
""")
]

class TaskDescriptionGenerator:
    def __init__(self, model):        
        self.description_prompt = ChatPromptTemplate.from_messages(DESCRIPTION_PROMPT)
        self.description_updating_prompt = ChatPromptTemplate.from_messages(DESCRIPTION_UPDATING_PROMPT)
        self.specification_suggestions_prompt = ChatPromptTemplate.from_messages(SPECIFICATION_SUGGESTIONS_PROMPT)
        self.generalization_suggestions_prompt = ChatPromptTemplate.from_messages(GENERALIZATION_SUGGESTIONS_PROMPT)
        self.input_analysis_prompt = ChatPromptTemplate.from_messages(INPUT_ANALYSIS_PROMPT)
        self.briefs_prompt = ChatPromptTemplate.from_messages(BRIEFS_PROMPT)
        self.examples_from_briefs_prompt = ChatPromptTemplate.from_messages(EXAMPLES_FROM_BRIEFS_PROMPT)
        self.examples_directly_prompt = ChatPromptTemplate.from_messages(EXAMPLES_DIRECTLY_PROMPT)

        json_model = model.bind(response_format={"type": "json_object"})
        # json_model = model

        output_parser = StrOutputParser()
        json_parse = JsonOutputParser()

        self.description_chain = self.description_prompt | model | output_parser
        self.description_updating_chain = self.description_updating_prompt | model | output_parser
        self.specification_suggestions_chain = (self.specification_suggestions_prompt | json_model | json_parse).with_retry(
            retry_if_exception_type=(BadRequestError,), # Retry only on ValueError
            wait_exponential_jitter=True, # Add jitter to the exponential backoff
            stop_after_attempt=2 # Try twice
        ).with_fallbacks([RunnableLambda(lambda x: {"dimensions": [], "suggestions": []})])
        self.generalization_suggestions_chain = (self.generalization_suggestions_prompt | json_model | json_parse).with_retry(
            retry_if_exception_type=(BadRequestError,), # Retry only on ValueError
            wait_exponential_jitter=True, # Add jitter to the exponential backoff
            stop_after_attempt=2 # Try twice
        ).with_fallbacks([RunnableLambda(lambda x: {"dimensions": [], "suggestions": []})])
        self.input_analysis_chain = self.input_analysis_prompt | model | output_parser
        # self.briefs_chain = self.briefs_prompt | model | output_parser
        self.briefs_chain = self.briefs_prompt | json_model | json_parse | RunnableLambda(lambda x: x["new_example_briefs"])
        self.examples_from_briefs_chain = (self.examples_from_briefs_prompt | json_model | json_parse).with_retry(
            retry_if_exception_type=(BadRequestError,), # Retry only on ValueError
            wait_exponential_jitter=True, # Add jitter to the exponential backoff
            stop_after_attempt=2 # Try twice
        ).with_fallbacks([RunnableLambda(lambda x: {"examples": []})])
        self.examples_directly_chain = (self.examples_directly_prompt | json_model | json_parse).with_retry(
            retry_if_exception_type=(BadRequestError,), # Retry only on ValueError
            wait_exponential_jitter=True, # Add jitter to the exponential backoff
            stop_after_attempt=2 # Try twice
        ).with_fallbacks([RunnableLambda(lambda x: {"examples": []})])

        # New sub-chain for loading and validating input
        self.input_loader = RunnableLambda(self.load_and_validate_input)

        self.chain = (
            self.input_loader
            | RunnablePassthrough.assign(raw_example=lambda x: json.dumps(x["example"], ensure_ascii=False))
            | RunnablePassthrough.assign(description=self.description_chain)
            | {
                "description": lambda x: x["description"],
                "examples_from_briefs": RunnablePassthrough.assign(input_analysis=self.input_analysis_chain)
                | RunnablePassthrough.assign(new_example_briefs=self.briefs_chain)
                | RunnablePassthrough.assign(examples=self.examples_from_briefs_chain | (lambda x: x["examples"])),
                "examples_directly": self.examples_directly_chain,
                "suggestions": {
                    "specification": self.specification_suggestions_chain,
                    "generalization": self.generalization_suggestions_chain
                } | RunnableLambda(lambda x: [item['suggestion'] for sublist in [v['suggestions'] for v in x.values()] for item in sublist])
            }
            | RunnablePassthrough.assign(
                additional_examples=lambda x: (
                    list(x["examples_from_briefs"]["examples"])
                    + list(x["examples_directly"]["examples"])
                )
            )
        )

    def parse_input_str(self, input_str):
        try:
            example_dict = json.loads(input_str)
        except ValueError:
            try:
                example_dict = yaml.safe_load(input_str)
            except yaml.YAMLError as e:
                raise ValueError("Invalid input format. Expected a JSON or YAML object.") from e
        
        # If example_dict is a list, filter out invalid items
        if isinstance(example_dict, list):
            example_dict = [item for item in example_dict if isinstance(item, dict) and 'input' in item and 'output' in item]

        # If example_dict is not a list, check if it's a valid dict
        elif not isinstance(example_dict, dict) or 'input' not in example_dict or 'output' not in example_dict:
            raise ValueError("Invalid input format. Expected an object with 'input' and 'output' fields.")
        
        return example_dict

    def load_and_validate_input(self, input_dict):
        input_str = input_dict["input_str"]
        generating_batch_size = input_dict.get("generating_batch_size")

        try:
            example_dict = self.parse_input_str(input_str)
            # Move the original content to a key named 'example'
            input_dict = {"example": example_dict}
            if generating_batch_size is not None:
                input_dict["generating_batch_size"] = generating_batch_size

            return input_dict

        except Exception as e:
            raise RuntimeError(f"An error occurred during processing: {str(e)}")

    def process(self, input_str, generating_batch_size=3):
        input_dict = {"input_str": input_str, "generating_batch_size": generating_batch_size}
        result = self.chain.invoke(input_dict)
        return result

    def generate_description(self, input_str, generating_batch_size=3):
        chain = (
            self.input_loader 
            | RunnablePassthrough.assign(raw_example=lambda x: json.dumps(x["example"], ensure_ascii=False))
            | RunnablePassthrough.assign(description=self.description_chain)
            | {
                "description": lambda x: x["description"],
                "suggestions": {
                    "specification": self.specification_suggestions_chain,
                    "generalization": self.generalization_suggestions_chain
                } | RunnableLambda(lambda x: [item['suggestion'] for sublist in [v['suggestions'] for v in x.values() if 'suggestions' in v] for item in sublist if 'suggestion' in item])
            }
        )
        return chain.invoke({
            "input_str": input_str,
            "generating_batch_size": generating_batch_size
        })
    
    def update_description(self, input_str, description, suggestions):
        # package array suggestions into a JSON array
        suggestions_str = json.dumps(suggestions, ensure_ascii=False)

        # return the updated description with new suggestions
        chain = (
            RunnablePassthrough.assign(
                description=self.description_updating_chain
            )
            | {
                "description": lambda x: x["description"],
                "suggestions": {
                    "specification": self.specification_suggestions_chain,
                    "generalization": self.generalization_suggestions_chain
                } | RunnableLambda(lambda x: [item['suggestion'] for sublist in [v['suggestions'] for v in x.values() if 'suggestions' in v] for item in sublist if 'suggestion' in item])
            }
        )
        return chain.invoke({
            "raw_example": input_str,
            "description": description,
            "suggestions": suggestions_str
        })

    def generate_suggestions(self, input_str, description):
        chain = RunnablePassthrough.assign(
            suggestions={
                "specification": self.specification_suggestions_chain,
                "generalization": self.generalization_suggestions_chain
            } | RunnableLambda(lambda x: [item['suggestion'] for sublist in [v['suggestions'] for v in x.values() if 'suggestions' in v] for item in sublist if 'suggestion' in item])
        )
        return chain.invoke({
            "description": description,
            "raw_example": input_str
        })

    def analyze_input(self, description):
        return self.input_analysis_chain.invoke(description)

    def generate_briefs(self, description, input_analysis, generating_batch_size):
        return self.briefs_chain.invoke({
            "description": description,
            "input_analysis": input_analysis,
            "generating_batch_size": generating_batch_size
        })

    def generate_examples_from_briefs(self, description, new_example_briefs, input_str, generating_batch_size=3):
        chain = (
            self.input_loader
            | RunnablePassthrough.assign(
                raw_example = lambda x: json.dumps(x["example"], ensure_ascii=False),
                description = lambda x: description,
                new_example_briefs = lambda x: new_example_briefs
            )
            | self.examples_from_briefs_chain
        )
        return chain.invoke({
            "description": description,
            "new_example_briefs": new_example_briefs,
            "input_str": input_str,
            "generating_batch_size": generating_batch_size
        })

    def generate_examples_directly(self, description, raw_example, generating_batch_size):
        return self.examples_directly_chain.invoke({
            "description": description,
            "raw_example": raw_example,
            "generating_batch_size": generating_batch_size
        })