"""
This module provides a Gradio interface for the Meta Prompt Analyzer.
"""
import gradio as gr
from confz import CLArgSource, EnvSource, FileSource
import pandas as pd
from app.config import MetaPromptConfig
from app.gradio_meta_prompt_utils import (
    evaluate_output,
    evaluate_system_message,
    generate_acceptance_criteria,
    generate_description,
    generate_initial_system_message,
)
from meta_prompt import (
    AgentState, MetaPromptGraph, META_PROMPT_NODES, NODE_ACCEPTANCE_CRITERIA_DEVELOPER,
    NODE_PROMPT_INITIAL_DEVELOPER, NODE_PROMPT_DEVELOPER, NODE_PROMPT_EXECUTOR,
    NODE_OUTPUT_HISTORY_ANALYZER, NODE_PROMPT_ANALYZER, NODE_PROMPT_SUGGESTER,
    DEFAULT_PROMPT_TEMPLATES
)


config_sources = [
    FileSource(file='config.yml', optional=True),
    EnvSource(prefix='METAPROMPT_', allow_all=True),
    CLArgSource(),
]

config = MetaPromptConfig(config_sources=config_sources)

# Define the examples path
examples_path = config.examples_path

def generate_text(
    input_example,
    output_example,
    text_type,
    model_name,
    initial_system_message=None,
    acceptance_criteria=None,
):
    if text_type == "Description":
        examples = pd.DataFrame(
            [["Input", "Output"], [input_example, output_example]],
            columns=["Input", "Output"],
        )
        description, _ = generate_description(
            config, examples, model_name, config.default_llm_temperature
        )
        return description
    elif text_type == "Initial System Message":
        initial_system_message, _ = generate_initial_system_message(
            config,
            input_example,
            output_example,
            model_name,
            config.default_llm_temperature,
            list(config.prompt_templates.keys())[0],
        )
        return initial_system_message
    elif text_type == "Acceptance Criteria":
        acceptance_criteria, _ = generate_acceptance_criteria(
            config,
            initial_system_message,
            input_example,
            output_example,
            model_name,
            config.default_llm_temperature,
            list(config.prompt_templates.keys())[0],
        )
        return acceptance_criteria
    elif text_type == "Output":
        return evaluate_system_message(
            config,
            initial_system_message,
            input_example,
            model_name,
            config.default_llm_temperature,
        )
    elif text_type == "Analysis":
        output = evaluate_system_message(
            config,
            initial_system_message,
            input_example,
            model_name,
            config.default_llm_temperature,
        )
        return evaluate_output(
            config,
            output_example,
            output,
            acceptance_criteria,
            model_name,
            config.default_llm_temperature,
            list(config.prompt_templates.keys())[0],
        )


def create_text_box_with_button(label):
    with gr.Group():
        textbox = gr.Textbox(label=label, lines=5, show_copy_button=True, interactive=True)
        button = gr.Button(f"Generate {label}")
    return textbox, button


def generate_all(input_example, output_example, model_name):
    desc = generate_text(input_example, output_example,
                         "Description", model_name)
    init_sys_msg = generate_text(
        input_example, output_example, "Initial System Message", model_name)
    acc_criteria = generate_text(input_example, output_example,
                                 "Acceptance Criteria", model_name, initial_system_message=init_sys_msg)
    out = generate_text(input_example, output_example, "Output",
                        model_name, initial_system_message=init_sys_msg)
    anal = generate_text(input_example, output_example, "Analysis", model_name,
                         initial_system_message=init_sys_msg, acceptance_criteria=acc_criteria)
    return desc, init_sys_msg, acc_criteria, out, anal


with gr.Blocks(title='Meta Prompt Analyzer') as demo:
    gr.Markdown("# Meta Prompt Analyzer")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                input_example = gr.Textbox(label="Input Example", lines=3, show_copy_button=True)
                output_example = gr.Textbox(label="Output Example", lines=3, show_copy_button=True)

            generate_all_button = gr.Button("Generate All", variant="primary")
            clear_all_button = gr.ClearButton(
                value="Clear All"
            )

            with gr.Row():
                initial_system_message, initial_system_message_button = create_text_box_with_button(
                    "Initial System Message"
                )
                output, output_button = create_text_box_with_button("Output")

            with gr.Row():
                acceptance_criteria, acceptance_criteria_button = create_text_box_with_button(
                    "Acceptance Criteria"
                )
                analysis, analysis_button = create_text_box_with_button("Analysis")

            description, description_button = create_text_box_with_button("Description")

        with gr.Column(scale=1):
            model_name = gr.Dropdown(
                label="Model Name", choices=config.llms.keys(), value=list(config.llms.keys())[0]
            )
            # Add gr.Examples component
            examples = gr.Examples(
                examples_path,
                examples_per_page=5,
                inputs=[input_example, output_example]
            )

    generate_all_button.click(
        generate_all,
        inputs=[input_example, output_example, model_name],
        outputs=[description, initial_system_message, acceptance_criteria, output, analysis]
    )

    clear_all_button.add(
        [input_example, output_example, description, initial_system_message, acceptance_criteria, output, analysis]
    )

    for text_type, button, textbox in [
        ("Description", description_button, description),
        ("Initial System Message", initial_system_message_button, initial_system_message),
        ("Acceptance Criteria", acceptance_criteria_button, acceptance_criteria),
        ("Output", output_button, output),
        ("Analysis", analysis_button, analysis),
    ]:
        button.click(
            generate_text,
            inputs=[
                input_example,
                output_example,
                gr.State(text_type),
                model_name,
                gr.State(initial_system_message.value),
                gr.State(acceptance_criteria.value),
            ],
            outputs=[textbox],
        )

demo.launch(server_name=config.server_name, server_port=config.server_port)