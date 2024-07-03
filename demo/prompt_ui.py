"""
MIT License

Copyright (c) 2023 Yale Huang
Email: calvino.huang@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import re
import gradio as gr

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from demo.default_meta_prompts import *

gpt_models_not_legacy = [
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-0613"
]

gpt_models_legacy = [
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-0301",
    "text-davinci-003",
    "text-davinci-002",
    "code-davinci-002"
]

DEFAULT_MODEL_FOR_GENERATING="gpt-4"
DEFAULT_MODEL_FOR_TESTING="gpt-3.5-turbo"
DEFAULT_MODEL_FOR_OUTPUT_EVALUATING="gpt-3.5-turbo-instruct"
DEFAULT_CURRENT_SYSTEM_PROMPT = ''
DEFAULT_OUTPUT_EVALUATING_PROMPT = 'Find out which is more similar to string S, A or B? Print nothing if there\'s no significant difference between A and B. Else, print the result (letter A or B) only. Do nothing else.'

class PromptUI:
    def __init__(self, advanced_mode = False, enable_other_user_prompts = False):
        self.advanced_mode = advanced_mode
        self.enable_other_user_prompts = enable_other_user_prompts
        self.ui = self.init_ui()

    def init_ui(self):
        with gr.Blocks() as prompt_ui:            
            with gr.Row():
                with gr.Column():
                    self.testing_user_prompt_textbox = gr.Textbox(
                        label="Testing User Prompt", 
                        lines=10, 
                        interactive=True,
                        show_copy_button=True
                        )
                    self.expect_output_textbox = gr.Textbox(
                        label="Expected Output", 
                        lines=5, 
                        interactive=True,
                        show_copy_button=True
                        )
                    self.other_user_prompts_checkbox = gr.Checkbox(
                        label="Other User Prompts", 
                        info="Enable other user prompts in meta prompt?",
                        value=self.enable_other_user_prompts
                        )
                    self.other_user_prompts_textbox = gr.Textbox(
                        label="Other User Prompts",
                        lines=10, 
                        interactive=True,
                        placeholder="Wrap each prompt with a pair of '```'.",
                        visible=self.enable_other_user_prompts,
                        show_copy_button=True
                        )
                    # Add gr.Number here for iterations input
                    self.iterations_number = gr.Number(value=1, label="Optimize Iterations", min=1, max=1000, step=1, decimals=0)
                    # Add button to trigger optimization here
                    self.optimize_btn = gr.Button(value="Optimize Prompt", variant='primary')
                    self.similar_candidate_textbox = gr.Textbox(label="Similarity Delta", lines=1, interactive=True)
                    self.compare_outputs_btn = gr.Button(value="Compare Outputs")

                with gr.Column():
                    self.new_system_prompt_textbox = gr.Textbox(
                        label="New System Prompt", 
                        lines=5, 
                        interactive=True,
                        show_copy_button=True
                        )
                    self.new_output_textbox = gr.Textbox(
                        label="New Output", 
                        lines=5, 
                        interactive=True,
                        show_copy_button=True
                        )
                    with gr.Row():
                        self.run_meta_btn = gr.Button(value="↑ Single Step Optimize")
                        self.run_new_btn = gr.Button(value="⟳ Run New")
                    self.new_system_prompt_changed = gr.Checkbox(
                        label="New System Prompt Changed",
                        value=False,
                        interactive=False
                        )

                with gr.Column():
                    self.current_system_prompt_textbox = gr.Textbox(
                        label="Current System Prompt",
                        value=DEFAULT_CURRENT_SYSTEM_PROMPT,
                        lines=5,
                        interactive=True,
                        show_copy_button=True
                        )
                    self.current_output_textbox = gr.Textbox(
                        label="Current Output", 
                        lines=5, 
                        interactive=True,
                        show_copy_button=True
                        )
                    with gr.Row():
                        self.accept_new_btn = gr.Button(value="→ Accept New Prompt")
                        self.run_current_btn = gr.Button(value="⟳ Run Current")

            with gr.Row(visible=self.advanced_mode):
                with gr.Column():
                    self.meta_system_prompt_textbox = gr.Textbox(label="Meta System Prompt", 
                                                                value=DEFAULT_META_SYSTEM_PROMPT,
                                                                lines=10, 
                                                                interactive=True
                                                                )
                with gr.Column():
                    self.merged_meta_prompt_textbox = gr.Textbox(label="Merged Meta System Prompt", 
                                                                lines=10, 
                                                                interactive=False,
                                                                show_copy_button=True
                                                                )
                    self.merge_prompt_btn = gr.Button(value="Merge Meta System Prompt")
                    # self.chatgpt_output_textbox = gr.Textbox(label="Paste ChatGPT Output", 
                    #                                         lines=10,
                    #                                         interactive=True
                    #                                         )
                    # self.parse_chatgpt_output_btn = gr.Button(value="Parse ChatGPT Output")                                      

            with gr.Row(visible=self.advanced_mode):
                with gr.Column():
                    self.llm_model_meta_dropdown = gr.Dropdown(
                        label="Generating LLM Model", 
                        choices=gpt_models_not_legacy,
                        value=DEFAULT_MODEL_FOR_GENERATING,
                        interactive=True,
                        allow_custom_value=False
                    )

                    self.llm_model_meta_temperature_slider = gr.Slider(
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.01, 
                        value=0.0,
                        interactive=True,
                        label="Generating LLM Model Temperature"
                    )

                    self.llm_model_meta_max_tokens_slider = gr.Slider(
                        minimum=256, 
                        maximum=32000, 
                        step=256, 
                        value=0,
                        interactive=True, 
                        label="Generating LLM Model Token Limit (0 for auto)"
                    )

                    self.llm_model_meta_request_timeout_slider = gr.Slider(
                        minimum=0, 
                        maximum=600, 
                        step=5, 
                        value=600,
                        interactive=True,
                        label="Generating LLM Model Timeout"
                    )

                    self.llm_model_meta_max_retries_slider = gr.Slider(
                        minimum=0, 
                        maximum=30, 
                        step=1, 
                        value=6,
                        interactive=True,
                        label="Generating LLM Model Max Retries"
                    )

                with gr.Column():
                    self.llm_model_test_dropdown = gr.Dropdown(
                        label="Testing LLM Model",
                        choices=gpt_models_not_legacy,
                        value=DEFAULT_MODEL_FOR_TESTING,
                        interactive=True,
                        allow_custom_value=False
                    )

                    self.llm_model_test_temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0, 
                        step=0.01, 
                        value=0.0,
                        interactive=True,
                        label="Testing LLM Model Temperature"
                    )

                    self.llm_model_test_max_tokens_slider = gr.Slider(
                        minimum=256, 
                        maximum=32000, 
                        step=256, 
                        value=0,
                        interactive=True, 
                        label="Testing LLM Model Token Limit  (0 for auto)"
                    )

                    self.llm_model_test_request_timeout_slider = gr.Slider(
                        minimum=0, 
                        maximum=600, 
                        step=5, 
                        value=600, 
                        interactive=True,
                        label="Testing LLM Model Timeout"
                    )

                    self.llm_model_test_max_retries_slider = gr.Slider(
                        minimum=0, 
                        maximum=30, 
                        step=1, 
                        value=6, 
                        interactive=True,
                        label="Testing LLM Model Max Retries"
                    )
                # with gr.Column():
                #     self.llm_model_output_eval_dropdown = gr.Dropdown(label="Output Evaluating LLM Model",
                #                                                       choices=gpt_models_legacy,
                #                                                       value=DEFAULT_MODEL_FOR_OUTPUT_EVALUATING,
                #                                                       interactive=True,
                #                                                       allow_custom_value=False)
                #     self.llm_model_output_eval_slider = gr.Slider(minimum=0.0,
                #                                                          maximum=1.0,
                #                                                          step=0.01,
                #                                                          default=0.0, 
                #                                                          label="Output Evaluating LLM Model of Temperature")


            self.run_new_btn.click(
                self.test_prompt, 
                [
                    self.new_system_prompt_textbox,
                    self.testing_user_prompt_textbox,
                    self.llm_model_test_dropdown,
                    self.llm_model_test_max_retries_slider,
                    self.llm_model_test_max_tokens_slider,
                    self.llm_model_test_request_timeout_slider,
                    self.llm_model_test_temperature_slider
                ], 
                [self.new_output_textbox]
            )
            self.run_current_btn.click(
                self.test_prompt, 
                [
                    self.current_system_prompt_textbox, 
                    self.testing_user_prompt_textbox,
                    self.llm_model_test_dropdown,
                    self.llm_model_test_max_retries_slider,
                    self.llm_model_test_max_tokens_slider,
                    self.llm_model_test_request_timeout_slider,
                    self.llm_model_test_temperature_slider
                ], 
                [self.current_output_textbox]
            )
            self.run_meta_btn.click(
                self.meta_prompt, 
                [
                    self.meta_system_prompt_textbox, 
                    self.current_system_prompt_textbox,
                    self.testing_user_prompt_textbox,
                    self.other_user_prompts_textbox,
                    self.expect_output_textbox,
                    self.current_output_textbox,
                    self.other_user_prompts_checkbox,
                    self.llm_model_meta_dropdown,
                    self.llm_model_meta_max_retries_slider,
                    self.llm_model_meta_max_tokens_slider,
                    self.llm_model_meta_request_timeout_slider,
                    self.llm_model_meta_temperature_slider
                ], 
                [self.new_system_prompt_textbox, self.new_system_prompt_changed]
            )
            self.accept_new_btn.click(self.copy_new_prompts,
                                      [self.new_system_prompt_textbox, self.new_output_textbox],
                                      [self.current_system_prompt_textbox, self.current_output_textbox])
            self.compare_outputs_btn.click(self.compare_outputs,
                                           [self.new_output_textbox, self.current_output_textbox, self.expect_output_textbox],
                                           [self.similar_candidate_textbox])
            # Attach the optimize_prompt function to the button click event.
            # You should implement this function according to your optimization logic.
            self.optimize_btn.click(
                self.optimize_prompt,
                [
                    self.meta_system_prompt_textbox, 
                    self.current_system_prompt_textbox,
                    self.testing_user_prompt_textbox,
                    self.other_user_prompts_textbox,
                    self.expect_output_textbox,
                    self.current_output_textbox,
                    self.iterations_number,
                    self.other_user_prompts_checkbox,
                    self.llm_model_meta_dropdown,
                    self.llm_model_meta_max_retries_slider,
                    self.llm_model_meta_max_tokens_slider,
                    self.llm_model_meta_request_timeout_slider,
                    self.llm_model_meta_temperature_slider,
                    self.llm_model_test_dropdown,
                    self.llm_model_test_max_retries_slider,
                    self.llm_model_test_max_tokens_slider,
                    self.llm_model_test_request_timeout_slider,
                    self.llm_model_test_temperature_slider
                ],
                [self.new_system_prompt_textbox, self.new_system_prompt_changed])
            
            self.merge_prompt_btn.click(self.merge_meta_system_prompt,
                                        [
                                            self.meta_system_prompt_textbox, 
                                            self.current_system_prompt_textbox,
                                            self.other_user_prompts_textbox,
                                            self.testing_user_prompt_textbox, 
                                            self.expect_output_textbox, 
                                            self.current_output_textbox,
                                            self.other_user_prompts_checkbox
                                        ],
                                        [self.merged_meta_prompt_textbox])
            
            self.other_user_prompts_checkbox.change(self.update_enable_other_user_prompts,
                                                    [self.other_user_prompts_checkbox],
                                                    [
                                                        self.other_user_prompts_textbox,
                                                        self.meta_system_prompt_textbox
                                                        ])


        return prompt_ui

    def update_enable_other_user_prompts(self, new_value):
        self.enable_other_user_prompts = new_value
        return \
            gr.Textbox.update(visible=new_value), \
                gr.Textbox.update(
                    value = DEFAULT_META_SYSTEM_PROMPT_WITH_OTHER_PROMPTS if new_value else DEFAULT_META_SYSTEM_PROMPT
                    )
    
    def merge_meta_system_prompt(
            self, 
            meta_system_prompt, 
            current_system_prompt,
            other_user_prompts,
            testing_user_prompt, 
            expect_output, 
            current_output,
            use_other_user_prompts
            ):
        """Merge meta and current system prompts."""

        # converted_prompts = [prompt[0] for prompt in other_user_prompts.values]

        user_prompt = self.generate_user_message(
            current_system_prompt,
            testing_user_prompt,
            other_user_prompts if use_other_user_prompts else None,
            expect_output, 
            current_output
        )

        merged_prompt = f"{meta_system_prompt}\n\n{user_prompt}"

        return merged_prompt

    def copy_new_prompts(self, system_prompt, output):
        """Copy prompts and output from new to current textboxes."""

        return system_prompt, output

    def test_prompt(
            self, 
            system_prompt, 
            user_prompt,
            model,
            max_retries,
            max_tokens,
            request_timeout,
            temperature,
            ):
        # Create the prompt
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        chat_llm = ChatOpenAI(
            model=model,
            max_retries=max_retries,
            max_tokens=None if max_tokens == 0 else max_tokens,
            request_timeout=request_timeout,
            temperature=temperature
        )

        # Get the response from OpenAI
        gpt_response = chat_llm(prompt)

        # Return the output to be placed in the output textbox
        return gpt_response.content

    def generate_user_message(self, current_system_prompt, testing_user_prompt, other_user_prompts, expect_output, current_output):
        # other_prompts_formatted = '\n\n'.join([f"```\n{prompt}\n```" for prompt in other_user_prompts])
        user_message = f"""
* Prompt Template

```
{current_system_prompt}
```

* User Message

```
{testing_user_prompt}
```

* Other User Messages

{other_user_prompts}

* Expected GPT Message

```
{expect_output}
```

* GPT Message

```
{current_output}
```
""" if other_user_prompts is not None else f"""
* Prompt Template

```
{current_system_prompt}
```

* User Message

```
{testing_user_prompt}
```

* Expected GPT Message

```
{expect_output}
```

* GPT Message

```
{current_output}
```
"""
        return user_message

    def meta_prompt(
            self,
            meta_system_prompt, 
            current_system_prompt, 
            testing_user_prompt, 
            other_user_prompts,
            expect_output, 
            current_output,
            use_user_prompts,
            model,
            max_retries,
            max_tokens,
            request_timeout,
            temperature,
            ):

        # Format the user message
        user_message = self.generate_user_message(
            current_system_prompt,
            testing_user_prompt,
            other_user_prompts if use_user_prompts else None, 
            expect_output, 
            current_output
        )

        # Create the prompt
        prompt = [
            SystemMessage(content=meta_system_prompt),
            HumanMessage(content=user_message)
        ]

        chat_llm = ChatOpenAI(
            model=model,
            max_retries=max_retries,
            max_tokens=None if max_tokens == 0 else max_tokens,
            request_timeout=request_timeout,
            temperature=temperature
        )

        # Get the response from OpenAI
        gpt_response = chat_llm(prompt)

        updated_prompt = self.extract_updated_prompt(gpt_response.content)
        changed = not self.detect_no_change(gpt_response.content)

        # Return the output to be placed in the new system prompt textbox
        if updated_prompt:
            return updated_prompt, changed
        else:
            return gpt_response.content, changed

    def extract_updated_prompt(self, gpt_response):
        # Regular expression pattern to find the text enclosed
        pattern = "<!-- BEGIN OF PROMPT -->(.*?)<!-- END OF PROMPT -->"
        
        # Using search method to find the first occurrence of the pattern
        result = re.search(pattern, gpt_response, re.DOTALL)
        
        if result:
            s = result.group(1).strip("\n")
            if s.startswith("```") and s.endswith("```"):
                s = s[3:-3]        
            return s # Return the matched string
        else:
            return None  # If no such pattern is found return None

    def detect_no_change(self, gpt_response):
        # Regular expression pattern to find the exact string
        pattern = "<!-- NO CHANGE TO PROMPT -->"
        
        # Using search method to find the occurrence of the pattern
        result = re.search(pattern, gpt_response)
        
        if result:
            return True  # If the pattern is found return True
        else:
            return False  # If no such pattern is found return False
        
    # def compare_strings(self, a: str, b: str, s: str) -> str:
    #     # Create an instance of ChatOpenAI with the evaluation model
    #     chat_model = OpenAI(temperature=0, model_name=self.llm_model_output_eval_dropdown.value)

    #     # Create a prompt for comparison
    #     prompt = (DEFAULT_OUTPUT_EVALUATING_PROMPT + 
    #             '\n\n' + f'# S\n\n```\n{s}\n```\n\n# A\n\n```\n{a}\n```\n\n# B\n\n```\n{b}\n```\n\n')

    #     # Get the response from OpenAI
    #     response = chat_model(prompt)

    #     # Remove '```' from beginning and end if it exists
    #     if response.startswith("```") and response.endswith("```"):
    #         response = response[3:-3]

    #     # Check the first character of the response and return accordingly
    #     if response.startswith('A'):
    #         return 'A'
    #     elif response.startswith('B'):
    #         return 'B'
    #     else:
    #         return None

    def optimize_prompt(
            self, 
            meta_system_prompt, 
            current_system_prompt, 
            testing_user_prompt,
            other_user_prompts,
            expect_output, 
            current_output, 
            iterations,
            user_other_user_prompts,
            meta_model,
            meta_max_retries,
            meta_max_tokens,
            meta_request_timeout,
            meta_temperature,
            test_model,
            test_max_retries,
            test_max_tokens,
            test_request_timeout,
            test_temperature,
            ):

        changed = False

        # Iterate the specified number of times
        for i in range(int(iterations)):
            # If current_output is None or not provided, get it from test_prompt
            if current_output is None:
                current_output = self.test_prompt(
                    current_system_prompt, 
                    testing_user_prompt,
                    test_model,
                    test_max_retries,
                    test_max_tokens,
                    test_request_timeout,
                    test_temperature,
                    )

            # Call meta_prompt to get an optimized prompt
            new_prompt, changed = self.meta_prompt(
                meta_system_prompt, 
                current_system_prompt, 
                testing_user_prompt,
                other_user_prompts,
                expect_output, 
                current_output,
                user_other_user_prompts,
                meta_model,
                meta_max_retries,
                meta_max_tokens,
                meta_request_timeout,
                meta_temperature,
                )

            # If changed is False, break the loop
            if not changed:
                break

            # If there is an updated prompt and it's different from the current one, update current_system_prompt
            if new_prompt and new_prompt != current_system_prompt:
                current_system_prompt = new_prompt
                # Reset current_output to None so it gets recalculated in the next iteration
                current_output = None

        return current_system_prompt, changed  # Return the optimized system prompt

    def compare_strings(self, alpha: str, beta: str, expected: str) -> str:
        # If both ALPHA and BETA are empty, return None
        if not alpha and not beta:
            return None

        # If either ALPHA or BETA is empty, the non-empty string should be considered more similar to EXPECTED
        if not alpha:
            return 'B'
        if not beta:
            return 'A'

        # If both ALPHA and BETA are identical, return None
        if alpha == beta:
            return None

        # Create the CountVectorizer instance
        vectorizer = CountVectorizer().fit_transform([alpha, beta, expected])
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        alpha_sim = cosine_similarity(vectors[0].reshape(1, -1), vectors[2].reshape(1, -1))
        beta_sim = cosine_similarity(vectors[1].reshape(1, -1), vectors[2].reshape(1, -1))

        # Compare similarities and return the string that is more similar to the expected string
        if alpha_sim > beta_sim:
            return 'A'
        elif beta_sim > alpha_sim:
            return 'B'
        else:
            return None
        
    def delta_similarities(self, alpha: str, beta: str, expected: str) -> float:
        # If both ALPHA and BETA are empty, return 0
        if not alpha and not beta:
            return 0.0

        # If either ALPHA or BETA is empty, the non-empty string should be considered more similar to EXPECTED
        if not alpha:
            return -1.0
        if not beta:
            return 1.0

        # If both ALPHA and BETA are identical, return 0
        if alpha == beta:
            return 0.0

        # Create the CountVectorizer instance
        vectorizer = CountVectorizer().fit_transform([alpha, beta, expected])
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        alpha_sim = cosine_similarity(vectors[0].reshape(1, -1), vectors[2].reshape(1, -1))
        beta_sim = cosine_similarity(vectors[1].reshape(1, -1), vectors[2].reshape(1, -1))

        # Return the difference in similarities
        return alpha_sim[0][0] - beta_sim[0][0]

    def compare_outputs(self, new_output, current_output, expected_output):
        # Compare new output and current output against expected output
        # result = self.compare_strings(new_output, current_output, expected_output)
        result = self.delta_similarities(new_output, current_output, expected_output)

        return result
