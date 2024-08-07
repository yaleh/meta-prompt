import csv
import io
import json
import logging
from pathlib import Path
import pprint
from typing import Any, Dict, Union
import gradio as gr
from gradio import CSVLogger, Button, utils
from gradio.flagging import FlagMethod
from gradio_client import utils as client_utils
from confz import BaseConfig, CLArgSource, EnvSource, FileSource
from app.config import MetaPromptConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from meta_prompt import MetaPromptGraph, AgentState
from pythonjsonlogger import jsonlogger

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

def chat_log_2_chatbot_list(chat_log: str):
    chatbot_list = []
    if chat_log is None or chat_log == '':
        return chatbot_list
    for line in chat_log.splitlines():
        try:
            json_line = json.loads(line)
            if 'action' in json_line:
                if json_line['action'] == 'invoke':
                    chatbot_list.append([json_line['message'],None])
                if json_line['action'] == 'response':
                    chatbot_list.append([None,json_line['message']])
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON log output: {e}")
            print(line)
        except KeyError as e:
            print(f"Error accessing key in JSON log output: {e}")
            print(line)
    return chatbot_list

def process_message(user_message, expected_output, acceptance_criteria, initial_system_message,
                    recursion_limit: int, max_output_age: int,
                    llms: Union[BaseLanguageModel, Dict[str, BaseLanguageModel]]):
    # Create the input state
    input_state = AgentState(
        user_message=user_message,
        expected_output=expected_output,
        acceptance_criteria=acceptance_criteria,
        system_message=initial_system_message,
        max_output_age=max_output_age
    )
    
    # Get the output state from MetaPromptGraph
    log_stream = io.StringIO()
    log_handler = None
    logger = None
    if config.verbose:
        log_handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger(MetaPromptGraph.__name__)
        log_handler.setFormatter(jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
        logger.addHandler(log_handler)

    meta_prompt_graph = MetaPromptGraph(llms=llms, verbose=config.verbose, logger=logger)
    output_state = meta_prompt_graph(input_state, recursion_limit=recursion_limit)

    if config.verbose:
        log_handler.close()
        log_output = log_stream.getvalue()
    else:
        log_output = None
    
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

    return system_message, output, analysis, chat_log_2_chatbot_list(log_output)


def process_message_with_single_llm(user_message, expected_output, acceptance_criteria, initial_system_message,
                                    recursion_limit: int, max_output_age: int,
                                    model_name: str):
    # Get the output state from MetaPromptGraph
    type = config.llms[model_name].type
    args = config.llms[model_name].model_dump(exclude={'type'})
    llm = llm_model_factory.create(type, **args)

    return process_message(user_message, expected_output, acceptance_criteria, initial_system_message,
                           recursion_limit, max_output_age, llm)

def process_message_with_2_llms(user_message, expected_output, acceptance_criteria, initial_system_message,
                                       recursion_limit: int, max_output_age: int,
                                       optimizer_model_name: str, executor_model_name: str,):
    # Get the output state from MetaPromptGraph
    optimizer_model = llm_model_factory.create(config.llms[optimizer_model_name].type,
                                               **config.llms[optimizer_model_name].model_dump(exclude={'type'}))
    executor_model = llm_model_factory.create(config.llms[executor_model_name].type,
                                              **config.llms[executor_model_name].model_dump(exclude={'type'}))
    llms = {
        MetaPromptGraph.NODE_PROMPT_INITIAL_DEVELOPER: optimizer_model,
        MetaPromptGraph.NODE_PROMPT_DEVELOPER: optimizer_model,
        MetaPromptGraph.NODE_PROMPT_EXECUTOR: executor_model,
        MetaPromptGraph.NODE_OUTPUT_HISTORY_ANALYZER: optimizer_model,
        MetaPromptGraph.NODE_PROMPT_ANALYZER: optimizer_model,
        MetaPromptGraph.NODE_PROMPT_SUGGESTER: optimizer_model
    }

    return process_message(user_message, expected_output, acceptance_criteria, initial_system_message,
                           recursion_limit, max_output_age, llms)

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
                label="User Message", show_copy_button=True)
            expected_output_input = gr.Textbox(
                label="Expected Output", show_copy_button=True)
            acceptance_criteria_input = gr.Textbox(
                label="Acceptance Criteria", show_copy_button=True)
            initial_system_message_input = gr.Textbox(
                label="Initial System Message", show_copy_button=True, value="")
            recursion_limit_input = gr.Number(label="Recursion Limit", value=config.recursion_limit,
                                              precision=0, minimum=1, maximum=config.recursion_limit_max, step=1)
            max_output_age = gr.Number(label="Max Output Age", value=config.max_output_age,
                                       precision=0, minimum=1, maximum=config.max_output_age_max, step=1)
            with gr.Row():
                with gr.Tab('Simple'):
                    model_name_input = gr.Dropdown(
                        label="Model Name",
                        choices=config.llms.keys(),
                        value=list(config.llms.keys())[0],
                    )
                    # Connect the inputs and outputs to the function
                    with gr.Row():
                        submit_button = gr.Button(value="Submit", variant="primary")
                        clear_button = gr.ClearButton(
                            [user_message_input, expected_output_input,
                             acceptance_criteria_input, initial_system_message_input],
                            value='Clear All')
                with gr.Tab('Advanced'):
                    optimizer_model_name_input = gr.Dropdown(
                        label="Optimizer Model Name",
                        choices=config.llms.keys(),
                        value=list(config.llms.keys())[0],
                    )
                    executor_model_name_input = gr.Dropdown(
                        label="Executor Model Name",
                        choices=config.llms.keys(),
                        value=list(config.llms.keys())[0],
                    )
                    # Connect the inputs and outputs to the function
                    with gr.Row():
                        multiple_submit_button = gr.Button(value="Submit", variant="primary")
                        multiple_clear_button = gr.ClearButton(components=[user_message_input,
                                                                           expected_output_input,
                                                                           acceptance_criteria_input,
                                                                           initial_system_message_input],
                                                               value='Clear All')
        with gr.Column():
            system_message_output = gr.Textbox(
                label="System Message", show_copy_button=True)
            output_output = gr.Textbox(label="Output", show_copy_button=True)
            analysis_output = gr.Textbox(
                label="Analysis", show_copy_button=True)
            flag_button = gr.Button(value="Flag", variant="secondary", visible=config.allow_flagging)
            with gr.Accordion("Details", open=False, visible=config.verbose):
                logs_chatbot = gr.Chatbot(label='Messages', show_copy_button=True, layout='bubble',
                                          bubble_full_width=False, render_markdown=False)
                clear_logs_button = gr.ClearButton([logs_chatbot], value='Clear Logs')

        clear_button.add([system_message_output, output_output,
                         analysis_output, logs_chatbot])
        multiple_clear_button.add([system_message_output, output_output,
                                    analysis_output, logs_chatbot])

    submit_button.click(process_message_with_single_llm,
                        inputs=[user_message_input, expected_output_input, acceptance_criteria_input,
                                initial_system_message_input, recursion_limit_input, max_output_age,
                                model_name_input],
                        outputs=[system_message_output, output_output, analysis_output, logs_chatbot])
    multiple_submit_button.click(process_message_with_2_llms,
                                    inputs=[user_message_input, expected_output_input, acceptance_criteria_input,
                                            initial_system_message_input, recursion_limit_input, max_output_age,
                                            optimizer_model_name_input, executor_model_name_input],
                                    outputs=[system_message_output, output_output, analysis_output, logs_chatbot])

    # Load examples
    examples = config.examples_path
    gr.Examples(examples, inputs=[user_message_input, expected_output_input, acceptance_criteria_input, initial_system_message_input, recursion_limit_input, model_name_input])

    flagging_inputs = [user_message_input, expected_output_input, acceptance_criteria_input, initial_system_message_input]

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
