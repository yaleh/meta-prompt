import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gradio as gr
from gradio import CSVLogger, Button, utils
from gradio.flagging import FlagMethod
from gradio_client import utils as client_utils

from confz import BaseConfig, CLArgSource, EnvSource, FileSource
from app.config import MetaPromptConfig, RoleMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Don't remove this import
from meta_prompt import *
from pythonjsonlogger import jsonlogger

def prompt_templates_confz2langchain(
    prompt_templates: Dict[str, Dict[str, List[RoleMessage]]]
) -> Dict[str, ChatPromptTemplate]:
    """
    Convert a dictionary of prompt templates from the configuration format to 
    the language chain format.

    This function takes a dictionary of prompt templates in the configuration 
    format and converts them to the language chain format. Each prompt template 
    is converted to a ChatPromptTemplate object, which is then stored in a new 
    dictionary with the same keys.

    Args:
        prompt_templates (Dict[str, Dict[str, List[RoleMessage]]]): 
            A dictionary of prompt templates in the configuration format.

    Returns:
        Dict[str, ChatPromptTemplate]: 
            A dictionary of prompt templates in the language chain format.
    """
    return {
        node: ChatPromptTemplate.from_messages(
            [
                (role_message.role, role_message.message)
                for role_message in role_messages
            ]
        )
        for node, role_messages in prompt_templates.items()
    }

class SimplifiedCSVLogger(CSVLogger):
    """
    A subclass of CSVLogger that logs only the components data to a CSV file,
    excluding flag, username, and timestamp information.
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
            save_dir = Path(flagging_dir) / client_utils.strip_invalid_filename_characters(
                getattr(component, "label", None) or f"component {idx}"
            )
            if utils.is_prop_update(sample):
                csv_data.append(str(sample))
            else:
                data = component.flag(sample, flag_dir=save_dir) if sample is not None else ""
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
    """A factory class for creating instances of LLM models.

    This class follows the Singleton pattern, ensuring that only one instance is created.
    The `create` method dynamically instantiates a model based on the provided `model_type`.

    Attributes:
        _instance (LLMModelFactory): A private class variable to store the singleton instance.

    Methods:
        create(model_type: str, **kwargs) -> BaseLanguageModel:
            Dynamically creates and returns an instance of a model based on `model_type`.

    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LLMModelFactory, cls).__new__(cls)
        return cls._instance

    def create(self, model_type: str, **kwargs) -> BaseLanguageModel:
        """Creates and returns an instance of a model based on `model_type`.

        Args:
            model_type (str): The name of the model class to instantiate.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            BaseLanguageModel: An instance of a model that inherits from BaseLanguageModel.

        """
        model_class = globals()[model_type]
        return model_class(**kwargs)


def chat_log_2_chatbot_list(chat_log: str) -> List[List[str]]:
    """
    Convert a chat log string into a list of dialogues for the Chatbot format.

    Args:
        chat_log (str): A JSON formatted chat log where each line represents an
                        action with its message. Expected actions are 'invoke'
                        and 'response'.

    Returns:
        List[List[str]]: A list of dialogue pairs where the first element is a
                         user input and the second element is a bot response.
                         If the action was 'invoke', the first element will be
                         the message, and the second element will be None. If
                         the action was 'response', the first element will be
                         None, and the second element will be the message.
    """
    chatbot_list = []
    if chat_log is None or chat_log == '':
        return chatbot_list
    for line in chat_log.splitlines():
        try:
            json_line = json.loads(line)
            if 'action' in json_line:
                if json_line['action'] == 'invoke':
                    chatbot_list.append([json_line['message'], None])
                if json_line['action'] == 'response':
                    chatbot_list.append([None, json_line['message']])
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON log output: {e}")
            print(line)
        except KeyError as e:
            print(f"Error accessing key in JSON log output: {e}")
            print(line)
    return chatbot_list

def on_model_tab_select(simple_model_name, 
                        advanced_optimizer_model_name, advanced_executor_model_name,
                        expert_prompt_initial_developer_model_name,
                        expert_prompt_acceptance_criteria_developer_model_name,
                        expert_prompt_developer_model_name,
                        expert_prompt_executor_model_name,
                        expert_prompt_history_analyzer_model_name,
                        expert_prompt_analyzer_model_name,
                        expert_prompt_suggester_model_name,
                        event: gr.SelectData):
    if event.value == 'Simple':
        return simple_model_name, \
            simple_model_name, \
            simple_model_name, \
            simple_model_name, \
            simple_model_name, \
            simple_model_name, \
            simple_model_name
    elif event.value == 'Advanced':
        return advanced_optimizer_model_name, \
            advanced_optimizer_model_name, \
            advanced_optimizer_model_name, \
            advanced_executor_model_name, \
            advanced_optimizer_model_name, \
            advanced_optimizer_model_name, \
            advanced_optimizer_model_name
    elif event.value == 'Expert':
        return expert_prompt_initial_developer_model_name, \
            expert_prompt_acceptance_criteria_developer_model_name, \
            expert_prompt_developer_model_name, \
            expert_prompt_executor_model_name, \
            expert_prompt_history_analyzer_model_name, \
            expert_prompt_analyzer_model_name, \
            expert_prompt_suggester_model_name
    else:
        raise ValueError(f"Invalid model tab selected: {event.value}")
    
def evaluate_system_message(system_message, user_message, executor_model_name):
    """
    Evaluate a system message by using it to generate a response from an
    executor model based on the current active tab and provided user message.

    This function retrieves the appropriate language model (LLM) for the
    current active model tab, formats a chat prompt template with the system
    message and user message, invokes the LLM using this formatted prompt, and
    returns the content of the output if it exists.

    Args:
        system_message (str): The system message to use when evaluating the
            response.
        user_message (str): The user's input message for which a response will
            be generated.
        executor_model_state (gr.State): The state object containing the name
            of the executor model to use.

    Returns:
        str: The content of the output generated by the LLM based on the system
            message and user message, if it exists; otherwise, an empty string.

    Raises:
        gr.Error: If there is a Gradio-specific error during the execution of
            this function.
        Exception: For any other unexpected errors that occur during the
            execution of this function.
    """
    llm = initialize_llm(executor_model_name)
    template = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        ("human", "{user_message}")
    ])
    try:
        output = llm.invoke(template.format(
            system_message=system_message, user_message=user_message))
        return output.content if hasattr(output, 'content') else ""
    except gr.Error as e:
        raise e
    except Exception as e:
        raise gr.Error(f"Error: {e}")


def generate_acceptance_criteria(user_message, expected_output, acceptance_criteria_model_name, prompt_template_group):
    """
    Generate acceptance criteria based on the user message and expected output.

    This function uses the MetaPromptGraph's run_acceptance_criteria_graph method
    to generate acceptance criteria.

    Args:
        user_message (str): The user's input message.
        expected_output (str): The anticipated response or outcome from the language
            model based on the user's message.
        acceptance_criteria_model_name (str): The name of the acceptance criteria model to use.
        prompt_template_group (Optional[str], optional): The group of prompt templates
            to use. Defaults to None.

    Returns:
        tuple: A tuple containing the generated acceptance criteria and the chat log.
    """

    log_stream = io.StringIO()
    logger = logging.getLogger(MetaPromptGraph.__name__) if config.verbose else None
    log_handler = logging.StreamHandler(log_stream) if logger else None

    if log_handler:
        log_handler.setFormatter(
            jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        )
        logger.addHandler(log_handler)

    llm = initialize_llm(acceptance_criteria_model_name)
    if prompt_template_group is None:
        prompt_template_group = 'default'
    prompt_templates = prompt_templates_confz2langchain(
        config.prompt_templates[prompt_template_group]
    )
    acceptance_criteria_graph = MetaPromptGraph(llms={
        NODE_ACCEPTANCE_CRITERIA_DEVELOPER: llm
    }, prompts=prompt_templates,
    verbose=config.verbose, logger=logger)
    state = AgentState(
        user_message=user_message,
        expected_output=expected_output
    )
    output_state = acceptance_criteria_graph.run_acceptance_criteria_graph(state)

    if log_handler:
        log_handler.close()
        log_output = log_stream.getvalue()
    else:
        log_output = None
    return output_state.get('acceptance_criteria', ""), chat_log_2_chatbot_list(log_output)


def generate_initial_system_message(
    user_message: str,
    expected_output: str,
    initial_developer_model_name: str,
    prompt_template_group: Optional[str] = None
) -> tuple:
    """
    Generate an initial system message based on the user message and expected output.

    Args:
        user_message (str): The user's input message.
        expected_output (str): The anticipated response or outcome from the language model.
        initial_developer_model_name (str): The name of the initial developer model to use.
        prompt_template_group (Optional[str], optional):
            The group of prompt templates to use. Defaults to None.

    Returns:
        tuple: A tuple containing the initial system message and the chat log.
    """

    log_stream = io.StringIO()
    logger = logging.getLogger(MetaPromptGraph.__name__) if config.verbose else None
    log_handler = logging.StreamHandler(log_stream) if logger else None

    if log_handler:
        log_handler.setFormatter(
            jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        )
        logger.addHandler(log_handler)

    llm = initialize_llm(initial_developer_model_name)

    if prompt_template_group is None:
        prompt_template_group = 'default'
    prompt_templates = prompt_templates_confz2langchain(
        config.prompt_templates[prompt_template_group]
    )

    initial_system_message_graph = MetaPromptGraph(
        llms={NODE_PROMPT_INITIAL_DEVELOPER: llm},
        prompts=prompt_templates,
        verbose=config.verbose,
        logger=logger
    )

    state = AgentState(
        user_message=user_message,
        expected_output=expected_output
    )

    output_state = initial_system_message_graph.run_prompt_initial_developer_graph(state)

    if log_handler:
        log_handler.close()
        log_output = log_stream.getvalue()
    else:
        log_output = None

    system_message = output_state.get('system_message', "")
    return system_message, chat_log_2_chatbot_list(log_output)


def process_message_with_models(
    user_message: str, expected_output: str, acceptance_criteria: str,
    initial_system_message: str, recursion_limit: int, max_output_age: int,
    initial_developer_model_name: str, acceptance_criteria_model_name: str,
    developer_model_name: str, executor_model_name: str, history_analyzer_model_name: str,
    analyzer_model_name: str, suggester_model_name: str,
    prompt_template_group: Optional[str] = None,
    aggressive_exploration: bool = False
) -> tuple:
    """
    Process a user message by executing the MetaPromptGraph with provided language models and input state.

    This function sets up the initial state of the conversation, logs the execution if verbose mode is enabled,
    and extracts the best system message, output, and analysis from the output state of the MetaPromptGraph.

    Args:
        user_message (str): The user's input message to be processed by the language model(s).
        expected_output (str): The anticipated response or outcome from the language model(s) based on the user's message.
        acceptance_criteria (str): Criteria that determines whether the output is acceptable or not.
        initial_system_message (str): Initial instruction given to the language model(s) before processing the user's message.
        recursion_limit (int): The maximum number of times the MetaPromptGraph can call itself recursively.
        max_output_age (int): The maximum age of output messages that should be considered in the conversation history.
        initial_developer_model_name (str): The name of the initial developer model to use.
        acceptance_criteria_model_name (str): The name of the acceptance criteria model to use.
        developer_model_name (str): The name of the developer model to use.
        executor_model_name (str): The name of the executor model to use.
        history_analyzer_model_name (str): The name of the history analyzer model to use.
        analyzer_model_name (str): The name of the analyzer model to use.
        suggester_model_name (str): The name of the suggester model to use.
        prompt_template_group (Optional[str], optional): The group of prompt templates to use. Defaults to None.
        aggressive_exploration (bool, optional): Whether to use aggressive exploration. Defaults to False.

    Returns:
        tuple: A tuple containing the best system message, output, analysis, acceptance criteria, and chat log in JSON format.
    """
    input_state = AgentState(
        user_message=user_message,
        expected_output=expected_output,
        acceptance_criteria=acceptance_criteria,
        system_message=initial_system_message,
        max_output_age=max_output_age
    )

    log_stream = io.StringIO()
    logger = logging.getLogger(MetaPromptGraph.__name__) if config.verbose else None
    log_handler = logging.StreamHandler(log_stream) if logger else None
    if log_handler:
        log_handler.setFormatter(jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'))
        logger.addHandler(log_handler)

    if prompt_template_group is None:
        prompt_template_group = 'default'
    prompt_templates = prompt_templates_confz2langchain(config.prompt_templates[prompt_template_group])
    llms = {
        NODE_PROMPT_INITIAL_DEVELOPER: initialize_llm(initial_developer_model_name),
        NODE_ACCEPTANCE_CRITERIA_DEVELOPER: initialize_llm(acceptance_criteria_model_name),
        NODE_PROMPT_DEVELOPER: initialize_llm(developer_model_name),
        NODE_PROMPT_EXECUTOR: initialize_llm(executor_model_name),
        NODE_OUTPUT_HISTORY_ANALYZER: initialize_llm(history_analyzer_model_name),
        NODE_PROMPT_ANALYZER: initialize_llm(analyzer_model_name),
        NODE_PROMPT_SUGGESTER: initialize_llm(suggester_model_name)
    }
    meta_prompt_graph = MetaPromptGraph(llms=llms, prompts=prompt_templates,
                                        aggressive_exploration=aggressive_exploration,
                                        verbose=config.verbose, logger=logger)
    try:
        output_state = meta_prompt_graph(input_state, recursion_limit=recursion_limit)
    except Exception as e:
        if isinstance(e, gr.Error):
            raise e
        else:
            raise gr.Error(f"Error: {e}")           

    if log_handler:
        log_handler.close()
        log_output = log_stream.getvalue()
    else:
        log_output = None

    system_message = output_state.get(
        'best_system_message', "Error: The output state does not contain a valid 'best_system_message'")
    output = output_state.get(
        'best_output', "Error: The output state does not contain a valid 'best_output'")
    analysis = output_state.get(
        'analysis', "Error: The output state does not contain a valid 'analysis'")
    acceptance_criteria = output_state.get(
        'acceptance_criteria', "Error: The output state does not contain a valid 'acceptance_criteria'")

    return (system_message, output, analysis, acceptance_criteria, chat_log_2_chatbot_list(log_output))


def initialize_llm(model_name: str, model_config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Initialize and return a language model (LLM) based on its name.

    This function retrieves the configuration for the specified language model
    from the application's configuration, creates an instance of the appropriate
    type of language model using that configuration, and returns it.

    Args:
        model_name (str): The name of the language model to initialize. This
            should correspond to a key in the 'llms' section of the application's
            configuration.
        model_config (Optional[Dict[str, Any]], optional): Optional model
            configurations. Defaults to None.

    Returns:
        Any: An instance of the specified type of language model, initialized
            with its configured settings.

    Raises:
        KeyError: If no configuration exists for the specified model name.
        NotImplementedError: If an unrecognized type is configured for the
            language model. This should not occur under normal circumstances
            because the LLMModelFactory class checks and validates the type when
            creating a new language model.
    """
    try:
        llm_config = config.llms[model_name]
        model_type = llm_config.type
        dumped_config = llm_config.model_dump(exclude={'type'})

        if model_config:
            dumped_config.update(model_config)

        return LLMModelFactory().create(model_type, **dumped_config)
    except KeyError:
        raise KeyError(f"No configuration exists for the model name: {model_name}")
    except NotImplementedError:
        raise NotImplementedError(
            f"Unrecognized type configured for the language model: {model_type}"
        )


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
        "initial_developer": gr.State(value=None), # None | str
        "acceptance_criteria": gr.State(value=None), # None | str
        "developer": gr.State(value=None), # None | str
        "executor": gr.State(value=None), # None | str
        "history_analyzer": gr.State(value=None), # None | str
        "analyzer": gr.State(value=None), # None | str
        "suggester": gr.State(value=None) # None | str
    }

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
        inputs=[user_message_input, expected_output_input,
                model_states["acceptance_criteria"],
                prompt_template_group],
        outputs=[acceptance_criteria_input, logs_chatbot]
    )
    generate_initial_system_message_button.click(
        generate_initial_system_message,
        inputs=[user_message_input, expected_output_input,
                model_states["initial_developer"],
                prompt_template_group],
        outputs=[initial_system_message_input, logs_chatbot]
    )

    evaluate_initial_system_message_button.click(
        evaluate_system_message,
        inputs=[
            initial_system_message_input,
            user_message_input,
            model_states["executor"]
        ],
        outputs=[output_output]
    )
    evaluate_system_message_button.click(
        evaluate_system_message,
        inputs=[
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
