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
        with gr.Column(scale=3):
            input_dataframe = gr.DataFrame(
                label="Input Examples",
                headers=["Input", "Output"],
                value=[],
                datatype=["str", "str"],
                column_widths=["50%", "50%"],
                row_count=(1, "dynamic"),
                col_count=(2, "fixed"),
                interactive=True,
                wrap=True
            )
        with gr.Column(scale=1, min_width=100):
            with gr.Group():
                editable_checkbox = gr.Checkbox(label="Editable", value=True)
                json_file_object = gr.File(
                    label="Import/Export JSON", file_types=[".json"], type="filepath",
                    min_width=80
                )
                export_button = gr.Button("Export to JSON")
                clear_inputs_button = gr.ClearButton(
                    [
                        input_dataframe
                    ],
                    value="Clear Inputs"
                )

    with gr.Row():
        with gr.Column(scale=3):
            selected_example_input = gr.Textbox(
                label="Selected Example Input",
                lines=2,
                show_copy_button=True,
                value="",
            )
            selected_example_output = gr.Textbox(
                label="Selected Example Output",
                lines=2,
                show_copy_button=True,
                value="",
            )

        with gr.Column(scale=1, min_width=100):
            selected_group_mode = gr.State(None)  # None, "update", "append"
            selected_group_index = gr.State(None)  # None, int
            selected_group_input = gr.State("")
            selected_group_output = gr.State("")

            selected_group_input.change(
                fn=lambda x: x,
                inputs=[selected_group_input],
                outputs=[selected_example_input],
            )
            selected_group_output.change(
                fn=lambda x: x,
                inputs=[selected_group_output],
                outputs=[selected_example_output],
            )

            with (selected_input_group := gr.Group(visible=False)):
                with gr.Row():
                    selected_row_index = gr.Number(
                        label="Selected Row Index", value=0, precision=0, interactive=False, visible=False
                    )
                    update_row_button = gr.Button(
                        "Update Selected Row", variant="secondary", visible=False
                    )
                    delete_row_button = gr.Button(
                        "Delete Selected Row", variant="secondary", visible=False
                    )
                    append_example_button = gr.Button(
                        "Append to Input Examples", variant="secondary", visible=False
                    )

                update_row_button.click(
                    fn=update_selected_dataframe_row,
                    inputs=[
                        selected_example_input,
                        selected_example_output,
                        selected_row_index,
                        input_dataframe,
                    ],
                    outputs=[
                        input_dataframe,
                        selected_group_mode,
                        selected_group_index,
                        selected_group_input,
                        selected_group_output,
                    ],
                )

                delete_row_button.click(
                    fn=delete_selected_dataframe_row,
                    inputs=[selected_row_index, input_dataframe],
                    outputs=[
                        input_dataframe,
                        selected_group_mode,
                        selected_group_index,
                        selected_group_input,
                        selected_group_output,
                    ],
                )

                append_example_button.click(
                    fn=append_example_to_input_dataframe,
                    inputs=[
                        selected_example_input,
                        selected_example_output,
                        input_dataframe,
                    ],
                    outputs=[
                        input_dataframe,
                        selected_group_mode,
                        selected_group_index,
                        selected_group_input,
                        selected_group_output,
                    ],
                )

            selected_group_mode.change(
                fn=lambda mode: [
                    gr.update(visible=(mode is not None)),
                    gr.update(visible=(mode == "update")),
                    gr.update(visible=(mode == "update")),
                    gr.update(visible=(mode == "update")),
                    gr.update(visible=(mode == "append")),
                ],
                inputs=[selected_group_mode],
                outputs=[selected_input_group, selected_row_index, update_row_button, delete_row_button, append_example_button],
            )

            selected_group_index.change(
                fn=lambda index: gr.update(value=index),
                inputs=[selected_group_index],
                outputs=[selected_row_index],
            )

    with gr.Tabs() as tabs:

        with gr.Tab("Scope"):

            with gr.Row():
                scope_submit_button = gr.Button("Generate", variant="primary", interactive=False)
                scope_clear_button = gr.ClearButton(
                    [
                    ],
                    value="Clear Outputs"
                )

            examples_output_dataframe = gr.DataFrame(
                # label="Examples",
                headers=["Input", "Output"],
                interactive=False,
                datatype=["str", "str"],
                column_widths=["50%", "50%"],
                row_count=(1, "dynamic"),
                col_count=(2, "fixed"),
                wrap=True
            )

            with gr.Accordion("Model Settings", open=False):
                scope_model_name = gr.Dropdown(
                    label="Model Name",
                    choices=config.llms.keys(),
                    value=list(config.llms.keys())[0],
                )
                temperature = gr.Slider(
                    label="Temperature", value=1.0, minimum=0.0, maximum=1.0, step=0.1
                )
                generating_batch_size = gr.Slider(
                    label="Generating Batch Size", value=3, minimum=1, maximum=10, step=1
                )

            with gr.Accordion("Analysis", open=False):
                with gr.Row():
                    with gr.Column():
                        generate_description_button = gr.Button(
                            "Generate Description", variant="secondary"
                        )
                        description_output = gr.Textbox(
                            label="Description", lines=5, show_copy_button=True
                        )
                    with gr.Column():
                        # Suggestions components
                        generate_suggestions_button = gr.Button(
                            "Generate Suggestions", variant="secondary")
                        suggestions_output = gr.Dropdown(
                            label="Suggestions", choices=[], multiselect=True, allow_custom_value=True)
                        apply_suggestions_button = gr.Button(
                            "Apply Suggestions", variant="secondary")

                with gr.Row():
                    with gr.Column():
                        analyze_input_button = gr.Button(
                            "Analyze Input", variant="secondary"
                        )
                        input_analysis_output = gr.Textbox(
                            label="Input Analysis", lines=5, show_copy_button=True
                        )
                    with gr.Column():
                        generate_briefs_button = gr.Button(
                            "Generate Briefs", variant="secondary"
                        )
                        example_briefs_output = gr.Textbox(
                            label="Example Briefs", lines=5, show_copy_button=True
                        )

                with gr.Row():
                    with gr.Column():
                        generate_examples_directly_button = gr.Button(
                            "Generate Examples Directly", variant="secondary"
                        )
                        examples_directly_output_dataframe = gr.DataFrame(
                            label="Examples Directly",
                            headers=["Input", "Output"],
                            interactive=False,
                            datatype=["str", "str"],
                            column_widths=["50%", "50%"],
                            row_count=(1, "dynamic"),
                            col_count=(2, "fixed"),
                            wrap=True
                        )

                    with gr.Column():
                        generate_examples_from_briefs_button = gr.Button(
                            "Generate Examples from Briefs", variant="secondary"
                        )
                        examples_from_briefs_output_dataframe = gr.DataFrame(
                            label="Examples from Briefs",
                            headers=["Input", "Output"],
                            interactive=False,
                            datatype=["str", "str"],
                            column_widths=["50%", "50%"],
                            row_count=(1, "dynamic"),
                            col_count=(2, "fixed"),
                            wrap=True
                        )

            scope_clear_button.add(
                [
                    description_output,
                    suggestions_output,
                    examples_directly_output_dataframe,
                    input_analysis_output,
                    example_briefs_output,
                    examples_from_briefs_output_dataframe,
                    examples_output_dataframe
                ]
            )

        with gr.Tab("Prompt"):

            with gr.Row():
                prompt_submit_button = gr.Button(value="Submit", variant="primary", interactive=False)
                prompt_clear_button = gr.ClearButton(value='Clear Output')

            with gr.Row():
                with gr.Column():
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
                                pull_task_description_output_button = gr.Button(
                                    value="→ Pull Output", variant="secondary")
                                pull_system_message_output_button = gr.Button(
                                    value="Pull Output ←", variant="secondary")

                        with gr.Group():
                            acceptance_criteria_input = gr.Textbox(
                                label="Acceptance Criteria (Compared with Expected Output [EO])",
                                show_copy_button=True
                            )
                            with gr.Row():
                                evaluate_acceptance_criteria_input_button = gr.Button("Evaluate")
                                generate_acceptance_criteria_button = gr.Button(
                                    value="Generate",
                                    variant="secondary"
                                )
                                pull_acceptance_criteria_output_button = gr.Button(
                                    value="Pull Output ←", variant="secondary")

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

                with gr.Column():
                    with gr.Group():
                        system_message_output = gr.Textbox(
                            label="System Message", show_copy_button=True)
                        with gr.Row():
                            evaluate_system_message_button = gr.Button(
                                value="Evaluate", variant="secondary")
                    output_output = gr.Textbox(
                        label="Output", show_copy_button=True)
                    with gr.Group():
                        acceptance_criteria_output = gr.Textbox(
                            label="Acceptance Criteria", show_copy_button=True)
                        evaluate_acceptance_criteria_output_button = gr.Button(
                            value="Evaluate", variant="secondary")
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
                selected_example_input,
                selected_example_output,
            ])

            prompt_model_tab_state = gr.State(value='Simple')
            model_name_states = {
                # None | str
                "initial_developer": gr.State(value=simple_model_name_input.value),
                # None | str
                "acceptance_criteria": gr.State(value=simple_model_name_input.value),
                # None | str
                "developer": gr.State(value=simple_model_name_input.value),
                # None | str
                "executor": gr.State(value=simple_model_name_input.value),
                # None | str
                "history_analyzer": gr.State(value=simple_model_name_input.value),
                # None | str
                "analyzer": gr.State(value=simple_model_name_input.value),
                # None | str
                "suggester": gr.State(value=simple_model_name_input.value)
            }
            model_temperature_states = {
                "initial_developer": gr.State(value=config.default_llm_temperature),
                "acceptance_criteria": gr.State(value=config.default_llm_temperature),
                "developer": gr.State(value=config.default_llm_temperature),
                "executor": gr.State(value=config.default_llm_temperature),
                "history_analyzer": gr.State(value=config.default_llm_temperature),
                "analyzer": gr.State(value=config.default_llm_temperature),
                "suggester": gr.State(value=config.default_llm_temperature)
            }

            config_state = gr.State(value=config)

            scope_inputs_ready_state = gr.State(value=False)
            prompt_inputs_ready_state = gr.State(value=False)

    # event handlers for inputs
    editable_checkbox.change(
        fn=lambda x: gr.update(interactive=x),
        inputs=[editable_checkbox],
        outputs=[input_dataframe],
    )

    clear_inputs_button.add(
        [selected_group_input, selected_example_output, selected_group_index, selected_group_mode]
    )

    # set up event handlers for the scope tab
    def valid_input_dataframe(x):
        # validate it's not empty and not all the values are ''
        return not x.empty and not x.isnull().any().any() and not x.eq('').any().any()

    input_dataframe.change(
        fn=valid_input_dataframe, # input_dataframe has at least 1 data row and no NaN values
        inputs=[input_dataframe],
        outputs=[scope_inputs_ready_state],
    )

    scope_inputs_ready_state.change(
        fn=lambda x: [gr.update(interactive=x)] * 5,
        inputs=[scope_inputs_ready_state],
        outputs=[scope_submit_button, generate_description_button,
                 generate_examples_directly_button, analyze_input_button, generate_briefs_button],
    )

    json_file_object.change(
        fn=import_json_data,
        inputs=[json_file_object, input_dataframe],
        outputs=[input_dataframe],
    )

    export_button.click(
        fn=export_json_data,
        inputs=[input_dataframe],
        outputs=[json_file_object],
    )

    scope_submit_button.click(
        fn=process_json_data,
        inputs=[
            config_state,
            input_dataframe,
            scope_model_name,
            generating_batch_size,
            temperature,
        ],
        outputs=[
            description_output,
            suggestions_output,
            examples_directly_output_dataframe,
            input_analysis_output,
            example_briefs_output,
            examples_from_briefs_output_dataframe,
            examples_output_dataframe,
        ],
    )

    generate_description_button.click(
        fn=generate_description,
        inputs=[
            config_state,
            input_dataframe,
            scope_model_name,
            temperature
        ],
        outputs=[description_output, suggestions_output],
    )

    generate_examples_directly_button.click(
        fn=generate_examples_from_description,
        inputs=[
            config_state,
            description_output,
            input_dataframe,
            generating_batch_size,
            scope_model_name,
            temperature,
        ],
        outputs=[examples_directly_output_dataframe],
    )

    analyze_input_button.click(
        fn=analyze_input_data,
        inputs=[config_state, description_output, scope_model_name, temperature],
        outputs=[input_analysis_output],
    )

    generate_briefs_button.click(
        fn=generate_example_briefs,
        inputs=[
            config_state,
            description_output,
            input_analysis_output,
            generating_batch_size,
            scope_model_name,
            temperature,
        ],
        outputs=[example_briefs_output],
    )

    generate_examples_from_briefs_button.click(
        fn=generate_examples_using_briefs,
        inputs=[
            config_state,
            description_output,
            example_briefs_output,
            input_dataframe,
            generating_batch_size,
            scope_model_name,
            temperature,
        ],
        outputs=[examples_from_briefs_output_dataframe],
    )

    input_dataframe.select(
        fn=format_selected_input_example_dataframe,
        inputs=[input_dataframe],
        outputs=[
            selected_group_mode,
            selected_group_index,
            selected_group_input,
            selected_group_output,
        ],
    )

    examples_directly_output_dataframe.select(
        fn=format_selected_example,
        inputs=[examples_directly_output_dataframe],
        outputs=[
            selected_group_mode,
            selected_group_index,
            selected_group_input,
            selected_group_output,
        ],
    )

    examples_from_briefs_output_dataframe.select(
        fn=format_selected_example,
        inputs=[examples_from_briefs_output_dataframe],
        outputs=[
            selected_group_mode,
            selected_group_index,
            selected_group_input,
            selected_group_output,
        ],
    )

    examples_output_dataframe.select(
        fn=format_selected_example,
        inputs=[examples_output_dataframe],
        outputs=[
            selected_group_mode,
            selected_group_index,
            selected_group_input,
            selected_group_output,
        ],
    )

    input_dataframe.change(
        fn=input_dataframe_change,
        inputs=[
            input_dataframe,
            selected_group_mode,
            selected_group_index,
            selected_group_input,
            selected_group_output,
        ],
        outputs=[
            selected_group_mode,
            selected_group_index,
            selected_group_input,
            selected_group_output,
        ],
    )

    generate_suggestions_button.click(
        fn=generate_suggestions,
        inputs=[config_state, description_output, input_dataframe, scope_model_name, temperature],
        outputs=[suggestions_output],
    )

    apply_suggestions_button.click(
        fn=apply_suggestions,
        inputs=[config_state, description_output, suggestions_output,
                input_dataframe, scope_model_name, temperature],
        outputs=[description_output, suggestions_output],
    )

    # set up event handlers for the prompt tab
    for item in [selected_example_input, selected_example_output]:
        item.change(
            fn=lambda x, y: all(v is not None and v != '' for v in [x, y]),
            inputs=[selected_example_input, selected_example_output],
            outputs=[prompt_inputs_ready_state],
        )

    prompt_inputs_ready_state.change(
        fn=lambda x: gr.update(interactive=x),
        inputs=[prompt_inputs_ready_state],
        outputs=[prompt_submit_button],
    )

    simple_llm_tab.select(
        on_model_tab_select,
        [
        ],
        [
            prompt_model_tab_state
        ]
    )
    advanced_llm_tab.select(
        on_model_tab_select,
        [
        ],
        [
            prompt_model_tab_state
        ]
    )
    expert_llm_tab.select(
        on_model_tab_select,
        [
        ],
        [
            prompt_model_tab_state
        ]
    )

    for item in [
        prompt_model_tab_state,
        simple_model_name_input,
        advanced_optimizer_model_name_input,
        advanced_executor_model_name_input,
        expert_prompt_initial_developer_model_name_input,
        expert_prompt_initial_developer_temperature_input,
        expert_prompt_acceptance_criteria_model_name_input,
        expert_prompt_acceptance_criteria_temperature_input,
        expert_prompt_developer_model_name_input,
        expert_prompt_developer_temperature_input,
        expert_prompt_executor_model_name_input,
        expert_prompt_executor_temperature_input,
        expert_output_history_analyzer_model_name_input,
        expert_output_history_analyzer_temperature_input,
        expert_prompt_analyzer_model_name_input,
        expert_prompt_analyzer_temperature_input,
        expert_prompt_suggester_model_name_input,
        expert_prompt_suggester_temperature_input,
    ]:
        item.change(
            on_prompt_model_tab_state_change,
            [
                config_state,
                prompt_model_tab_state,
                simple_model_name_input,
                advanced_optimizer_model_name_input,
                advanced_executor_model_name_input,
                expert_prompt_initial_developer_model_name_input,
                expert_prompt_initial_developer_temperature_input,
                expert_prompt_acceptance_criteria_model_name_input,
                expert_prompt_acceptance_criteria_temperature_input,
                expert_prompt_developer_model_name_input,
                expert_prompt_developer_temperature_input,
                expert_prompt_executor_model_name_input,
                expert_prompt_executor_temperature_input,
                expert_output_history_analyzer_model_name_input,
                expert_output_history_analyzer_temperature_input,
                expert_prompt_analyzer_model_name_input,
                expert_prompt_analyzer_temperature_input,
                expert_prompt_suggester_model_name_input,
                expert_prompt_suggester_temperature_input
            ],
            [
                model_name_states["initial_developer"],
                model_temperature_states["initial_developer"],
                model_name_states["acceptance_criteria"],
                model_temperature_states["acceptance_criteria"],
                model_name_states["developer"],
                model_temperature_states["developer"],
                model_name_states["executor"],
                model_temperature_states["executor"],
                model_name_states["history_analyzer"],
                model_temperature_states["history_analyzer"],
                model_name_states["analyzer"],
                model_temperature_states["analyzer"],
                model_name_states["suggester"],
                model_temperature_states["suggester"]
            ],
        )

    generate_acceptance_criteria_button.click(
        generate_acceptance_criteria,
        inputs=[config_state, initial_system_message_input, 
                selected_example_input, selected_example_output,
                model_name_states["acceptance_criteria"],
                model_temperature_states["acceptance_criteria"],
                prompt_template_group],
        outputs=[acceptance_criteria_input, logs_chatbot]
    )
    evaluate_acceptance_criteria_input_button.click(
        fn=evaluate_output,
        inputs=[
            config_state,
            selected_example_output,
            output_output,
            acceptance_criteria_input,
            model_name_states["analyzer"],
            model_temperature_states["analyzer"],
            prompt_template_group
        ],
        outputs=[analysis_output]
    )
    evaluate_acceptance_criteria_output_button.click(
        fn=evaluate_output,
        inputs=[
            config_state,
            selected_example_output,
            output_output,
            acceptance_criteria_output,
            model_name_states["analyzer"],
            model_temperature_states["analyzer"],
            prompt_template_group
        ],
        outputs=[analysis_output]
    )

    generate_initial_system_message_button.click(
        generate_initial_system_message,
        inputs=[config_state, selected_example_input, selected_example_output,
                model_name_states["initial_developer"],
                model_temperature_states["initial_developer"],
                prompt_template_group],
        outputs=[initial_system_message_input, logs_chatbot]
    )

    evaluate_initial_system_message_button.click(
        evaluate_system_message,
        inputs=[
            config_state,
            initial_system_message_input,
            selected_example_input,
            model_name_states["executor"],
            model_temperature_states["executor"]
        ],
        outputs=[output_output]
    )
    evaluate_system_message_button.click(
        evaluate_system_message,
        inputs=[
            config_state,
            system_message_output,
            selected_example_input,
            model_name_states["executor"],
            model_temperature_states["executor"]
        ],
        outputs=[output_output]
    )
    pull_task_description_output_button.click(
        lambda x: x,
        inputs=[description_output],
        outputs=[initial_system_message_input]
    )
    pull_system_message_output_button.click(
        lambda x: x,
        inputs=[system_message_output],
        outputs=[initial_system_message_input]
    )
    pull_acceptance_criteria_output_button.click(
        lambda x: x,
        inputs=[acceptance_criteria_output],
        outputs=[acceptance_criteria_input]
    )

    prompt_clear_button.add([
                             acceptance_criteria_input, initial_system_message_input, 
                             system_message_output, output_output,
                             acceptance_criteria_output, analysis_output, logs_chatbot])

    prompt_submit_button.click(
        process_message_with_models,
        inputs=[
            config_state,
            selected_example_input,
            selected_example_output,
            acceptance_criteria_input,
            initial_system_message_input,
            recursion_limit_input,
            max_output_age,
            model_name_states["initial_developer"],
            model_temperature_states["initial_developer"],
            model_name_states["acceptance_criteria"],
            model_temperature_states["acceptance_criteria"],
            model_name_states["developer"],
            model_temperature_states["developer"],
            model_name_states["executor"],
            model_temperature_states["executor"],
            model_name_states["history_analyzer"],
            model_temperature_states["history_analyzer"],
            model_name_states["analyzer"],
            model_temperature_states["analyzer"],
            model_name_states["suggester"],
            model_temperature_states["suggester"],
            prompt_template_group,
            aggressive_exploration
        ],
        outputs=[
            system_message_output,
            output_output,
            analysis_output,
            acceptance_criteria_output,
            logs_chatbot
        ]
    )

    examples.load_input_event.then(
        lambda: "append",
        None,
        selected_group_mode,
    )

    flagging_inputs = [
        selected_example_input,
        selected_example_output
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
