import json
import tempfile
import gradio as gr
import pandas as pd
from langchain_openai import ChatOpenAI
from meta_prompt.sample_generator import TaskDescriptionGenerator

def convert_examples_to_json(examples):
    pd_examples = pd.DataFrame(examples)
    pd_examples.columns = pd_examples.columns.str.lower()
    return pd_examples.to_json(orient="records")

def process_json_data(
    examples, model_name, generating_batch_size, temperature
):
    try:
        # Convert the gradio dataframe into a JSON array
        input_json = convert_examples_to_json(examples)

        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3
        )
        generator = TaskDescriptionGenerator(model)
        result = generator.process(input_json, generating_batch_size)

        description = result["description"]
        examples_directly = [
            [example["input"], example["output"]]
            for example in result["examples_directly"]["examples"]
        ]
        input_analysis = result["examples_from_briefs"]["input_analysis"]
        new_example_briefs = result["examples_from_briefs"]["new_example_briefs"]
        examples_from_briefs = [
            [example["input"], example["output"]]
            for example in result["examples_from_briefs"]["examples"]
        ]
        examples = [
            [example["input"], example["output"]]
            for example in result["additional_examples"]
        ]

        return (
            description,
            examples_directly,
            input_analysis,
            new_example_briefs,
            examples_from_briefs,
            examples,
        )
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")
    
def generate_description(examples, model_name, temperature):
    try:
        input_json = convert_examples_to_json(examples)

        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        description = generator.generate_description(input_json)
        return description
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")

def analyze_input_data(description, model_name, temperature):
    try:
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        input_analysis = generator.analyze_input(description)
        return input_analysis
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")
    
def generate_example_briefs(
    description, input_analysis, generating_batch_size, model_name, temperature
):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3
        )
        generator = TaskDescriptionGenerator(model)
        briefs = generator.generate_briefs(
            description, input_analysis, generating_batch_size
        )
        return briefs
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")


def generate_examples_using_briefs(
    description, new_example_briefs, examples, generating_batch_size, model_name, temperature
):
    try:
        input_json = convert_examples_to_json(examples)
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3
        )
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_examples_from_briefs(
            description, new_example_briefs, input_json, generating_batch_size
        )
        examples = [
            [example["input"], example["output"]]
            for example in result["examples"]
        ]
        return examples
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")


def generate_examples_from_description(
    description, raw_example, generating_batch_size, model_name, temperature
):
    try:
        input_json = convert_examples_to_json(raw_example)
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_examples_directly(
            description, input_json, generating_batch_size
        )
        examples = [
            [example["input"], example["output"]] for example in result["examples"]
        ]
        return examples
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")

def format_selected_input_example_dataframe(evt: gr.SelectData, examples):
    if evt.index[0] < len(examples):
        selected_example = examples.iloc[evt.index[0]]
        return "update", evt.index[0]+1, selected_example.iloc[0], selected_example.iloc[1]
    return None, None, None, None

def format_selected_example(evt: gr.SelectData, examples):
    if evt.index[0] < len(examples):
        selected_example = examples.iloc[evt.index[0]]
        return (
            "append",
            None,
            selected_example.iloc[0],
            selected_example.iloc[1],
        )
    return None, None, None, None

def import_json_data(file, input_dataframe):
    if file is not None:
        df = pd.read_json(file.name)
        # Uppercase the first letter of each column name
        df.columns = df.columns.str.title()
        return df
    return input_dataframe

def export_json_data(dataframe):
    if dataframe is not None and not dataframe.empty:
        # Copy the dataframe and lowercase the column names
        df_copy = dataframe.copy()
        df_copy.columns = df_copy.columns.str.lower()
        
        json_str = df_copy.to_json(orient="records", indent=2)

        # create a temporary file with the json string
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(json_str.encode("utf-8"))
            temp_file_path = temp_file.name

        return temp_file_path
    return None


def append_example_to_input_dataframe(
    new_example_input, new_example_output, input_dataframe
):
    try:
        new_row = pd.DataFrame(
            [[new_example_input, new_example_output]], columns=["Input", "Output"]
        )
        updated_df = pd.concat([input_dataframe, new_row], ignore_index=True)
        return updated_df, None, None, None, None
    except KeyError:
        raise gr.Error("Invalid input or output")


def delete_selected_dataframe_row(row_index, input_dataframe):
    if row_index is not None and row_index > 0:
        input_dataframe = input_dataframe.drop(index=row_index - 1).reset_index(
            drop=True
        )
        return input_dataframe, None, None, None, None
    return input_dataframe, None, None, None, None


def update_selected_dataframe_row(
    selected_example_input, selected_example_output, selected_row_index, input_dataframe
):
    if selected_row_index is not None and selected_row_index > 0:
        input_dataframe.iloc[selected_row_index - 1] = [
            selected_example_input,
            selected_example_output,
        ]
        return input_dataframe, None, None, None, None
    return input_dataframe, None, None, None, None


def input_dataframe_change(
    input_dataframe, selected_group_mode, selected_group_index, selected_group_input, selected_group_output
):
    if len(input_dataframe) <= 1:
        return None, None, None, None
    return (
        selected_group_mode,
        selected_group_index,
        selected_group_input,
        selected_group_output,
    )

with gr.Blocks(title="Task Description Generator") as demo:
    gr.Markdown("# Task Description Generator")
    gr.Markdown(
        "Enter a JSON object with 'input' and 'output' fields to generate a task description and additional examples."
    )

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

            with gr.Row():
                close_button = gr.Button("Close", variant="secondary")
            close_button.click(
                fn=lambda: None,
                inputs=[],
                outputs=[selected_group_mode],
            )

    with gr.Row():
        submit_button = gr.Button("Generate", variant="primary")
    with gr.Accordion("Import/Export JSON", open=False):
        json_file_object = gr.File(
            label="Import/Export JSON", file_types=[".json"], type="filepath"
        )
        export_button = gr.Button("Export to JSON")

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
        generate_description_button = gr.Button(
            "Generate Description", variant="secondary"
        )
        description_output = gr.Textbox(
            label="Description", lines=5, show_copy_button=True
        )
        with gr.Row():
            generate_examples_directly_button = gr.Button(
                "Generate Examples Directly", variant="secondary"
            )
            analyze_input_button = gr.Button(
                "Analyze Input", variant="secondary"
            )
        examples_directly_output_dataframe = gr.DataFrame(
            label="Examples Directly",
            headers=["Input", "Output"],
            interactive=False,
            datatype=["str", "str"],
            row_count=(1, "dynamic"),
            col_count=(2, "fixed"),
        )
        input_analysis_output = gr.Textbox(
            label="Input Analysis", lines=5, show_copy_button=True
        )
        generate_briefs_button = gr.Button(
            "Generate Briefs", variant="secondary"
        )
        example_briefs_output = gr.Textbox(
            label="Example Briefs", lines=5, show_copy_button=True
        )
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
    examples_output_dataframe = gr.DataFrame(
        label="Examples",
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
        outputs=[description_output],
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

if __name__ == "__main__":
    demo.launch()