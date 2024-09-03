# meta_prompt_utils.py

import json
import logging
import io
from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from meta_prompt import *
from meta_prompt.sample_generator import TaskDescriptionGenerator
from pythonjsonlogger import jsonlogger
from app.config import MetaPromptConfig, RoleMessage
from confz import BaseConfig, CLArgSource, EnvSource, FileSource

def prompt_templates_confz2langchain(
    prompt_templates: Dict[str, Dict[str, List[RoleMessage]]]
) -> Dict[str, ChatPromptTemplate]:
    return {
        node: ChatPromptTemplate.from_messages(
            [
                (role_message.role, role_message.message)
                for role_message in role_messages
            ]
        )
        for node, role_messages in prompt_templates.items()
    }

class LLMModelFactory:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LLMModelFactory, cls).__new__(cls)
        return cls._instance

    def create(self, model_type: str, **kwargs) -> BaseLanguageModel:
        model_class = globals()[model_type]
        return model_class(**kwargs)

def chat_log_2_chatbot_list(chat_log: str) -> List[List[str]]:
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

def get_current_model(simple_model_name: str,
                      advanced_model_name: str,
                      expert_model_name: str,
                      expert_model_config: Optional[Dict[str, Any]] = None,
                      config: MetaPromptConfig = None,
                      active_model_tab: str = "Simple") -> BaseLanguageModel:
    model_mapping = {
        "Simple": simple_model_name,
        "Advanced": advanced_model_name,
        "Expert": expert_model_name
    }
    
    try:
        model_name = model_mapping.get(active_model_tab, simple_model_name)
        model = config.llms[model_name]
        model_type = model.type
        model_config = model.model_dump(exclude={'type'})
    
        if active_model_tab == "Expert" and expert_model_config:
            model_config.update(expert_model_config)
    
        return LLMModelFactory().create(model_type, **model_config)
    
    except KeyError as e:
        logging.error(f"Configuration key error: {e}")
        raise ValueError(f"Invalid model name or configuration: {e}")
    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise RuntimeError(f"Failed to retrieve the model: {e}")

def evaluate_system_message(system_message, user_message, simple_model,
                            advanced_executor_model, expert_executor_model,
                            expert_executor_model_temperature=0.1,
                            config: MetaPromptConfig = None,
                            active_model_tab: str = "Simple"):
    llm = get_current_model(simple_model, advanced_executor_model,
                            expert_executor_model,
                            {"temperature": expert_executor_model_temperature},
                            config, active_model_tab)
    template = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        ("human", "{user_message}")
    ])
    try:
        output = llm.invoke(template.format(
            system_message=system_message, user_message=user_message))
        return output.content if hasattr(output, 'content') else ""
    except Exception as e:
        raise Exception(f"Error: {e}")

def generate_acceptance_criteria(user_message, expected_output,
                                 simple_model, advanced_executor_model,
                                 expert_prompt_acceptance_criteria_model,
                                 expert_prompt_acceptance_criteria_temperature=0.1,
                                 prompt_template_group: Optional[str] = None,
                                 config: MetaPromptConfig = None,
                                 active_model_tab: str = "Simple"):
    log_stream = io.StringIO()
    logger = logging.getLogger(MetaPromptGraph.__name__) if config.verbose else None
    log_handler = logging.StreamHandler(log_stream) if logger else None

    if log_handler:
        log_handler.setFormatter(
            jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        )
        logger.addHandler(log_handler)

    llm = get_current_model(simple_model, advanced_executor_model,
                            expert_prompt_acceptance_criteria_model,
                            {"temperature": expert_prompt_acceptance_criteria_temperature},
                            config, active_model_tab)
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
    simple_model: str,
    advanced_executor_model: str,
    expert_prompt_initial_developer_model: str,
    expert_prompt_initial_developer_temperature: float = 0.1,
    prompt_template_group: Optional[str] = None,
    config: MetaPromptConfig = None,
    active_model_tab: str = "Simple"
) -> tuple:
    log_stream = io.StringIO()
    logger = logging.getLogger(MetaPromptGraph.__name__) if config.verbose else None
    log_handler = logging.StreamHandler(log_stream) if logger else None

    if log_handler:
        log_handler.setFormatter(
            jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        )
        logger.addHandler(log_handler)

    llm = get_current_model(
        simple_model,
        advanced_executor_model,
        expert_prompt_initial_developer_model,
        {"temperature": expert_prompt_initial_developer_temperature},
        config,
        active_model_tab
    )

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

def process_message(
    user_message: str, expected_output: str, acceptance_criteria: str,
    initial_system_message: str, recursion_limit: int, max_output_age: int,
    llms: Union[BaseLanguageModel, Dict[str, BaseLanguageModel]],
    prompt_template_group: Optional[str] = None,
    aggressive_exploration: bool = False,
    config: MetaPromptConfig = None
) -> tuple:
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
    meta_prompt_graph = MetaPromptGraph(llms=llms, prompts=prompt_templates,
                                        aggressive_exploration=aggressive_exploration,
                                        verbose=config.verbose, logger=logger)
    try:
        output_state = meta_prompt_graph(input_state, recursion_limit=recursion_limit)
    except Exception as e:
        raise Exception(f"Error: {e}")           

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

def initialize_llm(model_name: str, model_config: Optional[Dict[str, Any]] = None, config: MetaPromptConfig = None) -> Any:
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

# Sample generator functions

def process_json(input_json, model_name, generating_batch_size, temperature, config: MetaPromptConfig = None):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.process(input_json, generating_batch_size)
        description = result["description"]
        suggestions = result["suggestions"]
        examples_directly = [[example["input"], example["output"]]
                             for example in result["examples_directly"]["examples"]]
        input_analysis = result["examples_from_briefs"]["input_analysis"]
        new_example_briefs = result["examples_from_briefs"]["new_example_briefs"]
        examples_from_briefs = [[example["input"], example["output"]]
                                for example in result["examples_from_briefs"]["examples"]]
        examples = [[example["input"], example["output"]]
                    for example in result["additional_examples"]]
        return description, suggestions, examples_directly, input_analysis, new_example_briefs, examples_from_briefs, examples
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}. Returning default values.")

def generate_description_only(input_json, model_name, temperature, config: MetaPromptConfig = None):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_description(input_json)
        description = result["description"]
        suggestions = result["suggestions"]
        return description, suggestions
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

def analyze_input(description, model_name, temperature, config: MetaPromptConfig = None):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        input_analysis = generator.analyze_input(description)
        return input_analysis
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

def generate_briefs(description, input_analysis, generating_batch_size, model_name, temperature, config: MetaPromptConfig = None):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        briefs = generator.generate_briefs(
            description, input_analysis, generating_batch_size)
        return briefs
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

def generate_examples_from_briefs(description, new_example_briefs, input_str, generating_batch_size, model_name, temperature, config: MetaPromptConfig = None):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_examples_from_briefs(
            description, new_example_briefs, input_str, generating_batch_size)
        examples = [[example["input"], example["output"]]
                    for example in result["examples"]]
        return examples
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

def generate_examples_directly(description, raw_example, generating_batch_size, model_name, temperature, config: MetaPromptConfig = None):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_examples_directly(
            description, raw_example, generating_batch_size)
        examples = [[example["input"], example["output"]]
                    for example in result["examples"]]
        return examples
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

class FileConfig(BaseConfig):
    config_file: str = 'config.yml'  # default path

def load_config():
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

    return MetaPromptConfig(config_sources=config_sources)

# Add any additional utility functions here if needed