import json
import tempfile
import gradio as gr
import pandas as pd
from langchain_openai import ChatOpenAI
from meta_prompt.sample_generator import TaskDescriptionGenerator

def examples_to_json(examples):
    pd_examples = pd.DataFrame(examples)
    pd_examples.columns = pd_examples.columns.str.lower()
    return pd_examples.to_json(orient="records")

def process_json(
    examples, model_name, generating_batch_size, temperature
):
    try:
        # Convert the gradio dataframe into a JSON array
        input_json = examples_to_json(examples)

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
    
def generate_description_only(examples, model_name, temperature):
    try:
        input_json = examples_to_json(examples)

        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        description = generator.generate_description(input_json)
        return description
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")

def analyze_input(description, model_name, temperature):
    try:
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        input_analysis = generator.analyze_input(description)
        return input_analysis
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")
    
def generate_briefs(
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


def generate_examples_from_briefs(
    description, new_example_briefs, examples, generating_batch_size, model_name, temperature
):
    try:
        input_json = examples_to_json(examples)
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


def generate_examples_directly(
    description, raw_example, generating_batch_size, model_name, temperature
):
    try:
        input_json = examples_to_json(raw_example)
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


def format_selected_example(evt: gr.SelectData, examples):
    if evt.index[0] < len(examples):
        selected_example = examples.iloc[evt.index[0]]
        json_example = json.dumps(
            {"input": selected_example.iloc[0], "output": selected_example.iloc[1]},
            indent=2,
            ensure_ascii=False,
        )
        return json_example
    return ""

def import_json(file):
    if file is not None:
        df = pd.read_json(file.name)
        # Uppercase the first letter of each column name
        df.columns = df.columns.str.title()
        return df
    return None

def export_json(df):
    if df is not None and not df.empty:
        # Copy the dataframe and lowercase the column names
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()
        
        json_str = df_copy.to_json(orient="records", indent=2)

        # create a temporary file with the json string
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(json_str.encode("utf-8"))
            temp_file_path = temp_file.name

        return temp_file_path
    return None

with gr.Blocks(title="Task Description Generator") as demo:
    gr.Markdown("# Task Description Generator")
    gr.Markdown(
        "Enter a JSON object with 'input' and 'output' fields to generate a task description and additional examples."
    )

    with gr.Row():
        with gr.Column(scale=1):  # Inputs column
            input_df = gr.DataFrame(
                label="Input Examples",
                headers=["Input", "Output"],
                datatype=["str", "str"],
                row_count=(1, "dynamic"),
                col_count=(2, "fixed"),
            )
            json_file = gr.File(
                label="Import/Export JSON", file_types=[".json"], type="filepath"
            )
            export_button = gr.Button("Export to JSON")
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
            with gr.Row():
                submit_button = gr.Button("Generate", variant="primary")
                generate_description_button = gr.Button(
                    "Generate Description", variant="secondary"
                )

        with gr.Column(scale=1):  # Outputs column
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
            examples_directly_output = gr.DataFrame(
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
            examples_from_briefs_output = gr.DataFrame(
                label="Examples from Briefs",
                headers=["Input", "Output"],
                interactive=False,
                datatype=["str", "str"],
                row_count=(1, "dynamic"),
                col_count=(2, "fixed"),
            )
            examples_output = gr.DataFrame(
                label="Examples",
                headers=["Input", "Output"],
                interactive=False,
                datatype=["str", "str"],
                row_count=(1, "dynamic"),
                col_count=(2, "fixed"),
            )
            new_example_json = gr.Textbox(
                label="New Example JSON", lines=5, show_copy_button=True
            )

            clear_button = gr.ClearButton(
                [
                    input_df,
                    description_output,
                    input_analysis_output,
                    example_briefs_output,
                    examples_from_briefs_output,
                    examples_output,
                    new_example_json,
                ]
            )

    json_file.change(
        fn=import_json,
        inputs=[json_file],
        outputs=[input_df],
    )

    export_button.click(
        fn=export_json,
        inputs=[input_df],
        outputs=[json_file],
    )

    submit_button.click(
        fn=process_json,
        inputs=[
            input_df,
            model_name,
            generating_batch_size,
            temperature,
        ],
        outputs=[
            description_output,
            examples_directly_output,
            input_analysis_output,
            example_briefs_output,
            examples_from_briefs_output,
            examples_output,
        ],
    )

    generate_description_button.click(
        fn=generate_description_only,
        inputs=[input_df, model_name, temperature],
        outputs=[description_output],
    )

    generate_examples_directly_button.click(
        fn=generate_examples_directly,
        inputs=[
            description_output,
            input_df,
            generating_batch_size,
            model_name,
            temperature,
        ],
        outputs=[examples_directly_output],
    )

    analyze_input_button.click(
        fn=analyze_input,
        inputs=[description_output, model_name, temperature],
        outputs=[input_analysis_output],
    )

    generate_briefs_button.click(
        fn=generate_briefs,
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
        fn=generate_examples_from_briefs,
        inputs=[
            description_output,
            example_briefs_output,
            input_df,
            generating_batch_size,
            model_name,
            temperature,
        ],
        outputs=[examples_from_briefs_output],
    )

    examples_directly_output.select(
        fn=format_selected_example,
        inputs=[examples_directly_output],
        outputs=[new_example_json],
    )

    examples_from_briefs_output.select(
        fn=format_selected_example,
        inputs=[examples_from_briefs_output],
        outputs=[new_example_json],
    )

    examples_output.select(
        fn=format_selected_example,
        inputs=[examples_output],
        outputs=[new_example_json],
    )

    gr.Markdown("### Manual Flagging")
    with gr.Row():
        flag_button = gr.Button("Flag")
        flag_reason = gr.Textbox(label="Reason for flagging")

    flagging_callback = gr.CSVLogger()
    flag_button.click(
        lambda *args: flagging_callback.flag(args),
        inputs=[
            input_df,
            model_name,
            generating_batch_size,
            description_output,
            examples_output,
            flag_reason,
        ],
        outputs=[],
    )

if __name__ == "__main__":
    demo.launch()