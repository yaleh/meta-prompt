import gradio as gr
from gradio import Button, utils
from gradio.flagging import FlagMethod

from confz import CLArgSource, EnvSource, FileSource
from app.config import MetaPromptConfig
from meta_prompt import *
from app.gradio_meta_prompt_utils import *

pre_config_sources = [
    EnvSource(prefix='METAPROMPT_', allow_all=True),
    CLArgSource()
]
pre_config = FileConfig(config_sources=pre_config_sources)

config_sources = [
    FileSource(file=pre_config.config_file, optional=True),
    EnvSource(prefix='METAPROMPT_', allow_all=True),
    CLArgSource()
]

config = MetaPromptConfig(config_sources=config_sources)

flagging_callback = SimplifiedCSVLogger()

# Create a Gradio Blocks context
with gr.Blocks(title='Meta Prompt') as demo:
    # Define the layout
    with gr.Row():
        gr.Markdown(f"""<h1 style='text-align: left; margin-bottom: 1rem'>Meta Prompt</h1>
<p style="text-align:left">A tool for generating and analyzing natural language prompts using multiple language models.</p>
<a href="https://github.com/yaleh/meta-prompt"><img src="https://img.shields.io/badge/GitHub-blue?logo=github" alt="GitHub"></a>""")
    with gr.Row():
        with gr.Column():
            user_message_input = gr.Textbox(
                label="User Message",
                show_copy_button=True
            )
            expected_output_input = gr.Textbox(
                label="Expected Output",
                show_copy_button=True
            )
            with gr.Accordion("Initial System Message & Acceptance Criteria", open=False):

                with gr.Group():
                    initial_system_message_input = gr.Textbox(
                        label="Initial System Message",
                        show_copy_button=True,
                        value=""
                    )
                    with gr.Row():
                        evaluate_initial_system_message_button = gr.Button(
                            value="Evaluate",
                            variant="secondary"
                        )
                        generate_initial_system_message_button = gr.Button(
                            value="Generate",
                            variant="secondary"
                        )

                with gr.Group():
                    acceptance_criteria_input = gr.Textbox(
                        label="Acceptance Criteria (Compared with Expected Output [EO])",
                        show_copy_button=True
                    )
                    generate_acceptance_criteria_button = gr.Button(
                        value="Generate",
                        variant="secondary"
                    )

                recursion_limit_input = gr.Number(
                    label="Recursion Limit",
                    value=config.recursion_limit,
                    precision=0,
                    minimum=1,
                    maximum=config.recursion_limit_max,
                    step=1
                )
                max_output_age = gr.Number(
                    label="Max Output Age",
                    value=config.max_output_age,
                    precision=0,
                    minimum=1,
                    maximum=config.max_output_age_max,
                    step=1
                )
                prompt_template_group = gr.Dropdown(
                    label="Prompt Template Group",
                    choices=list(config.prompt_templates.keys()),
                    value=list(config.prompt_templates.keys())[0]
                )
                aggressive_exploration = gr.Checkbox(
                    label="Aggressive Exploration",
                    value=config.aggressive_exploration
                )
            with gr.Row():
                with gr.Tabs() as llm_tabs:
                    with gr.Tab('Simple') as simple_llm_tab:
                        simple_model_name_input = gr.Dropdown(
                            label="Model Name",
                            choices=config.llms.keys(),
                            value=list(config.llms.keys())[0],
                        )
                        # Connect the inputs and outputs to the function
                        with gr.Row():
                            simple_submit_button = gr.Button(
                                value="Submit", variant="primary")
                            simple_clear_button = gr.ClearButton(
                                [user_message_input, expected_output_input,
                                acceptance_criteria_input, initial_system_message_input],
                                value='Clear All')
                    with gr.Tab('Advanced') as advanced_llm_tab:
                        advanced_optimizer_model_name_input = gr.Dropdown(
                            label="Optimizer Model Name",
                            choices=config.llms.keys(),
                            value=list(config.llms.keys())[0],
                        )
                        advanced_executor_model_name_input = gr.Dropdown(
                            label="Executor Model Name",
                            choices=config.llms.keys(),
                            value=list(config.llms.keys())[0],
                        )
                        # Connect the inputs and outputs to the function
                        with gr.Row():
                            advanced_submit_button = gr.Button(
                                value="Submit", variant="primary")
                            advanced_clear_button = gr.ClearButton(
                                components=[user_message_input, expected_output_input,
                                            acceptance_criteria_input, initial_system_message_input],
                                value='Clear All')
                    with gr.Tab('Expert') as expert_llm_tab:
                        with gr.Row():
                            expert_prompt_initial_developer_model_name_input = gr.Dropdown(
                                label="Initial Developer Model Name",
                                choices=config.llms.keys(),
                                value=list(config.llms.keys())[0],
                            )
                            expert_prompt_initial_developer_temperature_input = gr.Number(
                                label="Initial Developer Temperature", value=0.1,
                                precision=1, minimum=0, maximum=1, step=0.1,
                                interactive=True)

                        with gr.Row():
                            expert_prompt_acceptance_criteria_model_name_input = gr.Dropdown(
                                label="Acceptance Criteria Model Name",
                                choices=config.llms.keys(),
                                value=list(config.llms.keys())[0],
                            )
                            expert_prompt_acceptance_criteria_temperature_input = gr.Number(
                                label="Acceptance Criteria Temperature", value=0.1,
                                precision=1, minimum=0, maximum=1, step=0.1,
                                interactive=True)

                        with gr.Row():
                            expert_prompt_developer_model_name_input = gr.Dropdown(
                                label="Developer Model Name",
                                choices=config.llms.keys(),
                                value=list(config.llms.keys())[0],
                            )
                            expert_prompt_developer_temperature_input = gr.Number(
                                label="Developer Temperature", value=0.1,
                                precision=1, minimum=0, maximum=1, step=0.1,
                                interactive=True)

                        with gr.Row():
                            expert_prompt_executor_model_name_input = gr.Dropdown(
                                label="Executor Model Name",
                                choices=config.llms.keys(),
                                value=list(config.llms.keys())[0],
                            )
                            expert_prompt_executor_temperature_input = gr.Number(
                                label="Executor Temperature", value=0.1,
                                precision=1, minimum=0, maximum=1, step=0.1,
                                interactive=True)

                        with gr.Row():
                            expert_output_history_analyzer_model_name_input = gr.Dropdown(
                                label="History Analyzer Model Name",
                                choices=config.llms.keys(),
                                value=list(config.llms.keys())[0],
                            )
                            expert_output_history_analyzer_temperature_input = gr.Number(
                                label="History Analyzer Temperature", value=0.1,
                                precision=1, minimum=0, maximum=1, step=0.1,
                                interactive=True)

                        with gr.Row():
                            expert_prompt_analyzer_model_name_input = gr.Dropdown(
                                label="Analyzer Model Name",
                                choices=config.llms.keys(),
                                value=list(config.llms.keys())[0],
                            )
                            expert_prompt_analyzer_temperature_input = gr.Number(
                                label="Analyzer Temperature", value=0.1,
                                precision=1, minimum=0, maximum=1, step=0.1,
                                interactive=True)

                        with gr.Row():
                            expert_prompt_suggester_model_name_input = gr.Dropdown(
                                label="Suggester Model Name",
                                choices=config.llms.keys(),
                                value=list(config.llms.keys())[0],
                            )
                            expert_prompt_suggester_temperature_input = gr.Number(
                                label="Suggester Temperature", value=0.1,
                                precision=1, minimum=0, maximum=1, step=0.1,
                                interactive=True)

                        # Connect the inputs and outputs to the function
                        with gr.Row():
                            expert_submit_button = gr.Button(
                                value="Submit", variant="primary")
                            expert_clear_button = gr.ClearButton(
                                components=[user_message_input, expected_output_input,
                                            acceptance_criteria_input, initial_system_message_input],
                                value='Clear All')
        with gr.Column():
            with gr.Group():
                system_message_output = gr.Textbox(
                    label="System Message", show_copy_button=True)
                with gr.Row():
                    evaluate_system_message_button = gr.Button(
                        value="Evaluate", variant="secondary")
                    copy_to_initial_system_message_button = gr.Button(
                        value="Copy to Initial System Message", variant="secondary")
            output_output = gr.Textbox(label="Output", show_copy_button=True)
            analysis_output = gr.Textbox(
                label="Analysis", show_copy_button=True)
            flag_button = gr.Button(
                value="Flag", variant="secondary", visible=config.allow_flagging)
            with gr.Accordion("Details", open=False, visible=config.verbose):
                logs_chatbot = gr.Chatbot(
                    label='Messages', show_copy_button=True, layout='bubble',
                    bubble_full_width=False, render_markdown=False
                )
                clear_logs_button = gr.ClearButton(
                    [logs_chatbot], value='Clear Logs')

    # Load examples
    examples = gr.Examples(config.examples_path, inputs=[
        user_message_input,
        expected_output_input,
        acceptance_criteria_input,
        initial_system_message_input,
        recursion_limit_input,
        simple_model_name_input
    ])

    model_states = {
        "initial_developer": gr.State(value=simple_model_name_input.value), # None | str
        "acceptance_criteria": gr.State(value=simple_model_name_input.value), # None | str
        "developer": gr.State(value=simple_model_name_input.value), # None | str
        "executor": gr.State(value=simple_model_name_input.value), # None | str
        "history_analyzer": gr.State(value=simple_model_name_input.value), # None | str
        "analyzer": gr.State(value=simple_model_name_input.value), # None | str
        "suggester": gr.State(value=simple_model_name_input.value) # None | str
    }

    config_state = gr.State(value=config)

    # set up event handlers
    simple_llm_tab.select(
        on_model_tab_select,
        [
            simple_model_name_input,
            advanced_optimizer_model_name_input,
            advanced_executor_model_name_input,
            expert_prompt_initial_developer_model_name_input,
            expert_prompt_acceptance_criteria_model_name_input,
            expert_prompt_developer_model_name_input,
            expert_prompt_executor_model_name_input,
            expert_output_history_analyzer_model_name_input,
            expert_prompt_analyzer_model_name_input,
            expert_prompt_suggester_model_name_input
        ],
        [
            model_states["initial_developer"],
            model_states["acceptance_criteria"],
            model_states["developer"],
            model_states["executor"],
            model_states["history_analyzer"],
            model_states["analyzer"],
            model_states["suggester"]
        ]
    )
    advanced_llm_tab.select(
        on_model_tab_select,
        [
            simple_model_name_input,
            advanced_optimizer_model_name_input,
            advanced_executor_model_name_input,
            expert_prompt_initial_developer_model_name_input,
            expert_prompt_acceptance_criteria_model_name_input,
            expert_prompt_developer_model_name_input,
            expert_prompt_executor_model_name_input,
            expert_output_history_analyzer_model_name_input,
            expert_prompt_analyzer_model_name_input,
            expert_prompt_suggester_model_name_input
        ],
        [
            model_states["initial_developer"],
            model_states["acceptance_criteria"],
            model_states["developer"],
            model_states["executor"],
            model_states["history_analyzer"],
            model_states["analyzer"],
            model_states["suggester"]
        ]
    )
    expert_llm_tab.select(
        on_model_tab_select,
        [
            simple_model_name_input,
            advanced_optimizer_model_name_input,
            advanced_executor_model_name_input,
            expert_prompt_initial_developer_model_name_input,
            expert_prompt_acceptance_criteria_model_name_input,
            expert_prompt_developer_model_name_input,
            expert_prompt_executor_model_name_input,
            expert_output_history_analyzer_model_name_input,
            expert_prompt_analyzer_model_name_input,
            expert_prompt_suggester_model_name_input
        ],
        [
            model_states["initial_developer"],
            model_states["acceptance_criteria"],
            model_states["developer"],
            model_states["executor"],
            model_states["history_analyzer"],
            model_states["analyzer"],
            model_states["suggester"]
        ]
    )

    generate_acceptance_criteria_button.click(
        generate_acceptance_criteria,
        inputs=[config_state, user_message_input, expected_output_input,
                model_states["acceptance_criteria"],
                prompt_template_group],
        outputs=[acceptance_criteria_input, logs_chatbot]
    )
    generate_initial_system_message_button.click(
        generate_initial_system_message,
        inputs=[config_state, user_message_input, expected_output_input,
                model_states["initial_developer"],
                prompt_template_group],
        outputs=[initial_system_message_input, logs_chatbot]
    )

    evaluate_initial_system_message_button.click(
        evaluate_system_message,
        inputs=[
            config_state,
            initial_system_message_input,
            user_message_input,
            model_states["executor"]
        ],
        outputs=[output_output]
    )
    evaluate_system_message_button.click(
        evaluate_system_message,
        inputs=[
            config_state,
            system_message_output,
            user_message_input,
            model_states["executor"]
        ],
        outputs=[output_output]
    )
    copy_to_initial_system_message_button.click(
        lambda x: x,
        inputs=[system_message_output],
        outputs=[initial_system_message_input]
    )

    simple_clear_button.add([system_message_output, output_output,
                        analysis_output, logs_chatbot])
    advanced_clear_button.add([system_message_output, output_output,
                                analysis_output, logs_chatbot])

    simple_submit_button.click(
        process_message_with_models,
        inputs=[
            config_state,
            user_message_input,
            expected_output_input,
            acceptance_criteria_input,
            initial_system_message_input,
            recursion_limit_input,
            max_output_age,
            model_states["initial_developer"],
            model_states["acceptance_criteria"],
            model_states["developer"],
            model_states["executor"],
            model_states["history_analyzer"],
            model_states["analyzer"],
            model_states["suggester"],
            prompt_template_group,
            aggressive_exploration
        ],
        outputs=[
            system_message_output,
            output_output,
            analysis_output,
            acceptance_criteria_input,
            logs_chatbot
        ]
    )

    advanced_submit_button.click(
        process_message_with_models,
        inputs=[
            config_state,
            user_message_input,
            expected_output_input,
            acceptance_criteria_input,
            initial_system_message_input,
            recursion_limit_input,
            max_output_age,
            model_states["initial_developer"],
            model_states["acceptance_criteria"],
            model_states["developer"],
            model_states["executor"],
            model_states["history_analyzer"],
            model_states["analyzer"],
            model_states["suggester"],
            prompt_template_group,
            aggressive_exploration
        ],
        outputs=[
            system_message_output,
            output_output,
            analysis_output,
            acceptance_criteria_input,
            logs_chatbot
        ]
    )

    expert_submit_button.click(
        process_message_with_models,
        inputs=[
            config_state,
            user_message_input,
            expected_output_input,
            acceptance_criteria_input,
            initial_system_message_input,
            recursion_limit_input,
            max_output_age,
            model_states["initial_developer"],
            model_states["acceptance_criteria"],
            model_states["developer"],
            model_states["executor"],
            model_states["history_analyzer"],
            model_states["analyzer"],
            model_states["suggester"],
            prompt_template_group,
            aggressive_exploration
        ],
        outputs=[
            system_message_output,
            output_output,
            analysis_output,
            acceptance_criteria_input,
            logs_chatbot
        ]
    )

    flagging_inputs = [
        user_message_input,
        expected_output_input,
        acceptance_criteria_input,
        initial_system_message_input
    ]

    # Configure flagging
    if config.allow_flagging:
        flag_method = FlagMethod(flagging_callback, "Flag", "")
        flag_button.click(
            utils.async_lambda(
                lambda: Button(value="Saving...", interactive=False)
            ),
            None,
            flag_button,
            queue=False,
            show_api=False,
        )
        flag_button.click(
            flag_method,
            inputs=flagging_inputs,
            outputs=flag_button,
            preprocess=False,
            queue=False,
            show_api=False,
        )

flagging_callback.setup(flagging_inputs, config.examples_path)

# Launch the Gradio app
demo.launch(server_name=config.server_name, server_port=config.server_port)
