import csv
from pathlib import Path
from typing import Any
import gradio as gr
from gradio import CSVLogger, utils
from gradio_client import utils as client_utils
from confz import BaseConfig, CLArgSource, EnvSource, FileSource
from meta_prompt import MetaPromptGraph, AgentState
from langchain_openai import ChatOpenAI
from app.config import MetaPromptConfig

class SimplifiedCSVLogger(CSVLogger):
    """
    A subclass of CSVLogger that logs only the components data to a CSV file, excluding
    flag, username, and timestamp information.
    """

    def flag(
        self,
        flag_data: list[Any],
        flag_option: str = "",
        username: str | None = None,
    ) -> int:
        flagging_dir = self.flagging_dir
        log_filepath = Path(flagging_dir) / "log.csv"
        is_new = not Path(log_filepath).exists()
        headers = [
            getattr(component, "label", None) or f"component {idx}"
            for idx, component in enumerate(self.components)
        ]

        csv_data = []
        for idx, (component, sample) in enumerate(zip(self.components, flag_data)):
            save_dir = Path(
                flagging_dir
            ) / client_utils.strip_invalid_filename_characters(
                getattr(component, "label", None) or f"component {idx}"
            )
            if utils.is_prop_update(sample):
                csv_data.append(str(sample))
            else:
                data = (
                    component.flag(sample, flag_dir=save_dir)
                    if sample is not None
                    else ""
                )
                if self.simplify_file_data:
                    data = utils.simplify_file_data_in_str(data)
                csv_data.append(data)

        with open(log_filepath, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if is_new:
                writer.writerow(utils.sanitize_list_for_csv(headers))
            writer.writerow(utils.sanitize_list_for_csv(csv_data))

        with open(log_filepath, encoding="utf-8") as csvfile:
            line_count = len(list(csv.reader(csvfile))) - 1
        return line_count

class LLMModelFactory:
    def __init__(self):
        pass

    def create(self, model_type: str, **kwargs):
        model_class = globals()[model_type]
        return model_class(**kwargs)
    
llm_model_factory = LLMModelFactory()

def process_message(user_message, expected_output, acceptance_criteria, initial_system_message,
                    recursion_limit: int, model_name: str):
    # Create the input state
    input_state = AgentState(
        user_message=user_message,
        expected_output=expected_output,
        acceptance_criteria=acceptance_criteria,
        system_message=initial_system_message
    )
    
    # Get the output state from MetaPromptGraph
    type = config.llms[model_name].type
    args = config.llms[model_name].model_dump(exclude={'type'})
    llm = llm_model_factory.create(type, **args)
    meta_prompt_graph = MetaPromptGraph(llms=llm)
    output_state = meta_prompt_graph(input_state, recursion_limit=recursion_limit)
    
    # Validate the output state
    system_message = ''
    output = ''
    analysis = ''

    if 'best_system_message' in output_state and output_state['best_system_message'] is not None:
        system_message = output_state['best_system_message']
    else:
        system_message = "Error: The output state does not contain a valid 'best_system_message'"

    if 'best_output' in output_state and output_state['best_output'] is not None:
        output = output_state["best_output"]
    else:
        output = "Error: The output state does not contain a valid 'best_output'"

    if 'analysis' in output_state and output_state['analysis'] is not None:
        analysis = output_state['analysis']
    else:
        analysis = "Error: The output state does not contain a valid 'analysis'"

    return system_message, output, analysis

class FileConfig(BaseConfig):
    config_file: str = 'config.yml'  # default path

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

# Create the Gradio interface
user_message_input = gr.Textbox(label="User Message", show_copy_button=True)
expected_output_input = gr.Textbox(label="Expected Output", show_copy_button=True)
acceptance_criteria_input = gr.Textbox(label="Acceptance Criteria", show_copy_button=True)

system_message_output = gr.Textbox(label="System Message", show_copy_button=True)
output_output = gr.Textbox(label="Output", show_copy_button=True)
analysis_output = gr.Textbox(label="Analysis", show_copy_button=True)

initial_system_message_input = gr.Textbox(label="Initial System Message", show_copy_button=True, value="")
recursion_limit_input = gr.Number(label="Recursion Limit", value=config.recursion_limit,
                                  precision=0, minimum=1, maximum=config.recursion_limit_max, step=1)
model_name_input = gr.Dropdown(
    label="Model Name",
    choices=config.llms.keys(),
    value=list(config.llms.keys())[0],
)

flagging_callback = SimplifiedCSVLogger()

iface = gr.Interface(
    fn=process_message,
    inputs=[
        user_message_input,
        expected_output_input,
        acceptance_criteria_input,
    ],
    outputs=[
        system_message_output,
        output_output,
        analysis_output
    ],
    additional_inputs=[
        initial_system_message_input,
        recursion_limit_input,
        model_name_input
    ],
    title="MetaPromptGraph Chat Interface",
    description="A chat interface for MetaPromptGraph to process user inputs and generate system messages.",
    examples=config.examples_path,
    allow_flagging=config.allow_flagging,
    flagging_dir=config.examples_path,
    flagging_options=["Example"],
    flagging_callback=flagging_callback
)
flagging_callback.setup([user_message_input, expected_output_input, acceptance_criteria_input, initial_system_message_input],config.examples_path)

# Launch the Gradio app
iface.launch(server_name=config.server_name, server_port=config.server_port)
