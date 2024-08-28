import json
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
    ("system", """Given the task type description, and input analysis, generate
descriptions for {generating_batch_size} new examples with detailed attributes
based on this task type. But don't provide any detailed task output.

Use the input analysis to create diverse and comprehensive example briefs that
cover various input dimensions and attribute ranges.

Format your response as a valid YAML object with a single key 'new_example_briefs'
containing a YAML array of {generating_batch_size} objects, each with a
'example_brief' field.
"""),
    ("user", """Task Description:

{description}

Input Analysis:

{input_analysis}

""")
]

EXAMPLES_FROM_BRIEFS_PROMPT = [
    ("system", """Given the task type description, brief descriptions for new examples, 
and JSON example(s), generate {generating_batch_size} more input/output examples for this task type,
strictly based on the brief descriptions. Ensure that the new examples are
consistent with the brief descriptions and do not introduce any new information
not present in the briefs.

Format your response as a valid JSON object with a single key 'examples' 
containing a JSON array of {generating_batch_size} objects, each with 'input' and 'output' fields.
"""),
    ("user", """Task Description:

{description}

New Example Briefs: 

{new_example_briefs}

Example(s):

{raw_example}

""")
]

EXAMPLES_DIRECTLY_PROMPT = [
    ("system", """Given the task type description, and input/output example(s), generate {generating_batch_size}
new input/output examples for this task type.

Format your response as a valid JSON object with a single key 'examples' 
containing a JSON array of {generating_batch_size} objects, each with 'input' and 'output' fields.
"""),
    ("user", """Task Description:

{description}

Example(s):

{raw_example}

""")
]


class TaskDescriptionGenerator:
    def __init__(self, model):        
        self.description_prompt = ChatPromptTemplate.from_messages(DESCRIPTION_PROMPT)
        self.input_analysis_prompt = ChatPromptTemplate.from_messages(INPUT_ANALYSIS_PROMPT)
        self.briefs_prompt = ChatPromptTemplate.from_messages(BRIEFS_PROMPT)
        self.examples_from_briefs_prompt = ChatPromptTemplate.from_messages(EXAMPLES_FROM_BRIEFS_PROMPT)
        self.examples_directly_prompt = ChatPromptTemplate.from_messages(EXAMPLES_DIRECTLY_PROMPT)

        json_model = model.bind(response_format={"type": "json_object"})

        output_parser = StrOutputParser()
        json_parse = JsonOutputParser()

        self.description_chain = self.description_prompt | model | output_parser
        self.input_analysis_chain = self.input_analysis_prompt | model | output_parser
        self.briefs_chain = self.briefs_prompt | model | output_parser
        self.examples_from_briefs_chain = self.examples_from_briefs_prompt | json_model | json_parse
        self.examples_directly_chain = self.examples_directly_prompt | json_model | json_parse

        # New sub-chain for loading and validating input
        self.input_loader = RunnableLambda(self.load_and_validate_input)

        self.chain = (
            self.input_loader
            | RunnablePassthrough.assign(raw_example = lambda x: json.dumps(x["example"], ensure_ascii=False))
            | RunnablePassthrough.assign(description = self.description_chain)
            | {
                "description": lambda x: x["description"],
                "examples_from_briefs": RunnablePassthrough.assign(input_analysis = self.input_analysis_chain)
                    | RunnablePassthrough.assign(new_example_briefs = self.briefs_chain) 
                    | RunnablePassthrough.assign(examples = self.examples_from_briefs_chain | (lambda x: x["examples"])),
                "examples_directly": self.examples_directly_chain
            }
            | RunnablePassthrough.assign(
                additional_examples=lambda x: (
                    list(x["examples_from_briefs"]["examples"])
                    + list(x["examples_directly"]["examples"])
                )
            )
        )

    def load_and_validate_input(self, input_dict):
        input_str = input_dict["input_str"]
        generating_batch_size = input_dict["generating_batch_size"]

        try:
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

            # Move the original content to a key named 'example'
            input_dict = {"example": example_dict, "generating_batch_size": generating_batch_size}

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
            | RunnablePassthrough.assign(raw_example = lambda x: json.dumps(x["example"], ensure_ascii=False))
            | self.description_chain
        )
        return chain.invoke({
            "input_str": input_str,
            "generating_batch_size": generating_batch_size
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