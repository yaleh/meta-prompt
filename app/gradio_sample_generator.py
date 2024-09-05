import gradio as gr

from app.gradio_meta_prompt_utils import *

with gr.Blocks(title="Meta Prompt") as demo:
    gr.Markdown("# Scope")

    input_dataframe = gr.DataFrame(
        label="Input Examples",
        headers=["Input", "Output"],
        datatype=["str", "str"],
        row_count=(1, "dynamic"),
        col_count=(2, "fixed"),
        interactive=False
    )

    selected_group_mode = gr.State(None) # None, "update", "append"
    selected_group_index = gr.State(None) # None, int
    selected_group_input = gr.State("")
    selected_group_output = gr.State("")

    @gr.render(
        inputs=[
            selected_group_mode,
            selected_group_index,
            selected_group_input,
            selected_group_output,
        ],
        triggers=[selected_group_mode.change],
    )
    def selected_group(mode, index, input, output):
        if mode is None:
            return
        with gr.Group():
            if mode == "update":
                with gr.Row():
                    selected_row_index = gr.Number(
                        label="Selected Row Index", value=index, precision=0, interactive=False
                    )
                    delete_row_button = gr.Button(
                        "Delete Selected Row", variant="secondary"
                    )
                with gr.Row():
                    selected_example_input = gr.Textbox(
                        label="Selected Example Input",
                        lines=2,
                        show_copy_button=True,
                        value=input,
                    )
                    selected_example_output = gr.Textbox(
                        label="Selected Example Output",
                        lines=2,
                        show_copy_button=True,
                        value=output,
                    )
                with gr.Row():
                    update_row_button = gr.Button(
                        "Update Selected Row", variant="secondary"
                    )
                    close_button = gr.Button("Close", variant="secondary")

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
            elif mode == "append":
                with gr.Row():
                    selected_example_input = gr.Textbox(
                        label="Selected Example Input",
                        lines=2,
                        show_copy_button=True,
                        value=input,
                    )
                    selected_example_output = gr.Textbox(
                        label="Selected Example Output",
                        lines=2,
                        show_copy_button=True,
                        value=output,
                    )
                with gr.Row():
                    append_example_button = gr.Button(
                        "Append to Input Examples", variant="secondary"
                    )
                    close_button = gr.Button("Close", variant="secondary")
                
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

            close_button.click(
                fn=lambda: None,
                inputs=[],
                outputs=[selected_group_mode],
            )

    with gr.Accordion("Import/Export JSON", open=False):
        json_file_object = gr.File(
            label="Import/Export JSON", file_types=[".json"], type="filepath"
        )
        export_button = gr.Button("Export to JSON")

    with gr.Group():
        submit_button = gr.Button("Generate", variant="primary")

        examples_output_dataframe = gr.DataFrame(
            # label="Examples",
            headers=["Input", "Output"],
            interactive=False,
            datatype=["str", "str"],
            row_count=(1, "dynamic"),
            col_count=(2, "fixed"),
        )

    with gr.Accordion("Model Settings", open=False):
        model_name = gr.Dropdown(
            label="Model Name",
            choices=[
                "llama3-70b-8192",
                "llama3-8b-8192",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "gemma2-9b-it",
            ],
            value="llama3-70b-8192",
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
                generate_suggestions_button = gr.Button("Generate Suggestions", variant="secondary")
                suggestions_output = gr.Dropdown(label="Suggestions", choices=[], multiselect=True, allow_custom_value=True)
                apply_suggestions_button = gr.Button("Apply Suggestions", variant="secondary")

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
                    row_count=(1, "dynamic"),
                    col_count=(2, "fixed"),
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
                    row_count=(1, "dynamic"),
                    col_count=(2, "fixed"),
                )

    clear_button = gr.ClearButton(
        [
            input_dataframe,
            description_output,
            suggestions_output,
            examples_directly_output_dataframe,
            input_analysis_output,
            example_briefs_output,
            examples_from_briefs_output_dataframe,
            examples_output_dataframe
        ],
        value="Clear All"
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

    submit_button.click(
        fn=process_json_data,
        inputs=[
            input_dataframe,
            model_name,
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
        inputs=[input_dataframe, model_name, temperature],
        outputs=[description_output, suggestions_output],
    )

    generate_examples_directly_button.click(
        fn=generate_examples_from_description,
        inputs=[
            description_output,
            input_dataframe,
            generating_batch_size,
            model_name,
            temperature,
        ],
        outputs=[examples_directly_output_dataframe],
    )

    analyze_input_button.click(
        fn=analyze_input_data,
        inputs=[description_output, model_name, temperature],
        outputs=[input_analysis_output],
    )

    generate_briefs_button.click(
        fn=generate_example_briefs,
        inputs=[
            description_output,
            input_analysis_output,
            generating_batch_size,
            model_name,
            temperature,
        ],
        outputs=[example_briefs_output],
    )

    generate_examples_from_briefs_button.click(
        fn=generate_examples_using_briefs,
        inputs=[
            description_output,
            example_briefs_output,
            input_dataframe,
            generating_batch_size,
            model_name,
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

    gr.Markdown("### Manual Flagging", visible=False)
    with gr.Row(visible=False):
        flag_button = gr.Button("Flag")
        flag_reason = gr.Textbox(label="Reason for flagging")

    flagging_callback = gr.CSVLogger()
    flag_button.click(
        lambda *args: flagging_callback.flag(args),
        inputs=[
            input_dataframe,
            model_name,
            generating_batch_size,
            description_output,
            examples_output_dataframe,
            flag_reason,
        ],
        outputs=[],
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
        inputs=[description_output, input_dataframe, model_name, temperature],
        outputs=[suggestions_output],
    )

    apply_suggestions_button.click(
        fn=apply_suggestions,
        inputs=[description_output, suggestions_output, input_dataframe, model_name, temperature],
        outputs=[description_output],
    )

if __name__ == "__main__":
    demo.launch()