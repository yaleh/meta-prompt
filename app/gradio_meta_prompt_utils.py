from typing import Any, Dict, List, Optional

import json
import logging
from pathlib import Path
import csv
import io
import tempfile

import pandas as pd

import gradio as gr
from gradio import CSVLogger, utils
from gradio_client import utils as client_utils

from confz import BaseConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Don't remove this import
from pythonjsonlogger import jsonlogger

from app.config import MetaPromptConfig, RoleMessage
from meta_prompt import *
from meta_prompt.sample_generator import TaskDescriptionGenerator

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

def on_prompt_model_tab_state_change(config, model_tab_select_state,
                              simple_model_name, advanced_optimizer_model_name, advanced_executor_model_name,
                              expert_prompt_initial_developer_model_name,
                              expert_prompt_initial_developer_temperature,
                              expert_prompt_acceptance_criteria_developer_model_name,
                              expert_prompt_acceptance_criteria_temperature,
                              expert_prompt_developer_model_name,
                              expert_prompt_developer_temperature,
                              expert_prompt_executor_model_name,
                              expert_prompt_executor_temperature,
                              expert_prompt_history_analyzer_model_name,
                              expert_prompt_history_analyzer_temperature,
                              expert_prompt_analyzer_model_name,
                              expert_prompt_analyzer_temperature,
                              expert_prompt_suggester_model_name,
                              expert_prompt_suggester_temperature):
    if model_tab_select_state == 'Simple':
        return simple_model_name, \
            config.default_llm_temperature, \
            simple_model_name, \
            config.default_llm_temperature, \
            simple_model_name, \
            config.default_llm_temperature, \
            simple_model_name, \
            config.default_llm_temperature, \
            simple_model_name, \
            config.default_llm_temperature, \
            simple_model_name, \
            config.default_llm_temperature, \
            simple_model_name, \
            config.default_llm_temperature
    elif model_tab_select_state == 'Advanced':
        return advanced_optimizer_model_name, \
            config.default_llm_temperature, \
            advanced_optimizer_model_name, \
            config.default_llm_temperature, \
            advanced_optimizer_model_name, \
            config.default_llm_temperature, \
            advanced_executor_model_name, \
            config.default_llm_temperature, \
            advanced_optimizer_model_name, \
            config.default_llm_temperature, \
            advanced_optimizer_model_name, \
            config.default_llm_temperature
    elif model_tab_select_state == 'Expert':
        return expert_prompt_initial_developer_model_name, \
            expert_prompt_initial_developer_temperature, \
            expert_prompt_acceptance_criteria_developer_model_name, \
            expert_prompt_acceptance_criteria_temperature, \
            expert_prompt_developer_model_name, \
            expert_prompt_developer_temperature, \
            expert_prompt_executor_model_name, \
            expert_prompt_executor_temperature, \
            expert_prompt_history_analyzer_model_name, \
            expert_prompt_history_analyzer_temperature, \
            expert_prompt_analyzer_model_name, \
            expert_prompt_analyzer_temperature, \
            expert_prompt_suggester_model_name, \
            expert_prompt_suggester_temperature
    else:
        raise ValueError(f"Invalid model tab selected: {model_tab_select_state}")

def on_model_tab_select(event: gr.SelectData):
    return event.value

def evaluate_system_message(config, system_message, user_message, executor_model_name, executor_temperature):
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
    llm = initialize_llm(config, executor_model_name, {'temperature': executor_temperature})
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


def generate_acceptance_criteria(config, user_message, expected_output, acceptance_criteria_model_name, acceptance_criteria_temperature, prompt_template_group):
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

    llm = initialize_llm(config, acceptance_criteria_model_name, {'temperature': acceptance_criteria_temperature})
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
    config,
    user_message: str,
    expected_output: str,
    initial_developer_model_name: str,
    initial_developer_temperature: float,
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

    llm = initialize_llm(config, initial_developer_model_name, {'temperature': initial_developer_temperature})

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
    config,
    user_message: str, expected_output: str, acceptance_criteria: str,
    initial_system_message: str, recursion_limit: int, max_output_age: int,
    initial_developer_model_name: str, initial_developer_temperature: float,
    acceptance_criteria_model_name: str, acceptance_criteria_temperature: float,
    developer_model_name: str, developer_temperature: float,
    executor_model_name: str, executor_temperature: float,
    history_analyzer_model_name: str, history_analyzer_temperature: float,
    analyzer_model_name: str, analyzer_temperature: float,
    suggester_model_name: str, suggester_temperature: float,
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
        NODE_PROMPT_INITIAL_DEVELOPER: initialize_llm(config, initial_developer_model_name, {'temperature': initial_developer_temperature}),
        NODE_ACCEPTANCE_CRITERIA_DEVELOPER: initialize_llm(config, acceptance_criteria_model_name, {'temperature': acceptance_criteria_temperature}),
        NODE_PROMPT_DEVELOPER: initialize_llm(config, developer_model_name, {'temperature': developer_temperature}),
        NODE_PROMPT_EXECUTOR: initialize_llm(config, executor_model_name, {'temperature': executor_temperature}),
        NODE_OUTPUT_HISTORY_ANALYZER: initialize_llm(config, history_analyzer_model_name, {'temperature': history_analyzer_temperature}),
        NODE_PROMPT_ANALYZER: initialize_llm(config, analyzer_model_name, {'temperature': analyzer_temperature}),
        NODE_PROMPT_SUGGESTER: initialize_llm(config, suggester_model_name, {'temperature': suggester_temperature})
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


def initialize_llm(config: MetaPromptConfig, model_name: str, model_config: Optional[Dict[str, Any]] = None) -> Any:
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


def convert_examples_to_json(examples):
    pd_examples = pd.DataFrame(examples)
    pd_examples.columns = pd_examples.columns.str.lower()
    return pd_examples.to_json(orient="records")

def process_json_data(
    config,
    examples, model_name, generating_batch_size, temperature
):
    try:
        # Convert the gradio dataframe into a JSON array
        input_json = convert_examples_to_json(examples)

        model = initialize_llm(config, model_name, {'temperature': temperature, 'max_retries': 3})
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
        suggestions = result.get("suggestions", [])
        return (
            description,
            gr.update(choices=suggestions, value=[]),
            examples_directly,
            input_analysis,
            new_example_briefs,
            examples_from_briefs,
            examples,
        )
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")
    
def generate_description(config, examples, model_name, temperature):
    try:
        input_json = convert_examples_to_json(examples)

        model = initialize_llm(config, model_name, {'temperature': temperature, 'max_retries': 3})
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_description(input_json)
        description = result["description"]
        suggestions = result["suggestions"]
        return description, gr.update(choices=suggestions, value=[])
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")

def analyze_input_data(config, description, model_name, temperature):
    try:
        model = initialize_llm(config, model_name, {'temperature': temperature, 'max_retries': 3})
        generator = TaskDescriptionGenerator(model)
        input_analysis = generator.analyze_input(description)
        return input_analysis
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")
    
def generate_example_briefs(
    config, description, input_analysis, generating_batch_size, model_name, temperature
):
    try:
        model = initialize_llm(config, model_name, {'temperature': temperature, 'max_retries': 3})
        generator = TaskDescriptionGenerator(model)
        briefs = generator.generate_briefs(
            description, input_analysis, generating_batch_size
        )
        return briefs
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")


def generate_examples_using_briefs(
    config, description, new_example_briefs, examples, generating_batch_size, model_name, temperature
):
    try:
        input_json = convert_examples_to_json(examples)
        model = initialize_llm(config, model_name, {'temperature': temperature, 'max_retries': 3})
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
    config, description, raw_example, generating_batch_size, model_name, temperature
):
    try:
        input_json = convert_examples_to_json(raw_example)
        model = initialize_llm(config, model_name, {'temperature': temperature, 'max_retries': 3})
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

def generate_suggestions(config, description, examples, model_name, temperature):
    try:
        input_json = convert_examples_to_json(examples)
        model = initialize_llm(config, model_name, {'temperature': temperature, 'max_retries': 3})
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_suggestions(input_json, description)
        return gr.update(choices=result["suggestions"])
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")

def apply_suggestions(config, description, suggestions, examples, model_name, temperature):
    try:
        input_json = convert_examples_to_json(examples)
        model = initialize_llm(config, model_name, {'temperature': temperature, 'max_retries': 3})
        generator = TaskDescriptionGenerator(model)
        result = generator.update_description(input_json, description, suggestions)
        return result["description"]
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")
