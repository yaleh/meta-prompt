import pandas as pd
import streamlit as st
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from meta_prompt import *
from pythonjsonlogger import jsonlogger
from app.config import MetaPromptConfig, RoleMessage
from confz import BaseConfig, CLArgSource, EnvSource, FileSource
import io

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

active_model_tab = "Simple"

def get_current_model(simple_model_name: str,
                        advanced_model_name: str,
                        expert_model_name: str,
                        expert_model_config: Optional[Dict[str, Any]] = None) -> BaseLanguageModel:
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
                            expert_executor_model_temperature=0.1):
    llm = get_current_model(simple_model, advanced_executor_model,
                            expert_executor_model,
                            {"temperature": expert_executor_model_temperature})
    template = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        ("human", "{user_message}")
    ])
    try:
        output = llm.invoke(template.format(
            system_message=system_message, user_message=user_message))
        return output.content if hasattr(output, 'content') else ""
    except Exception as e:
        raise st.error(f"Error: {e}")

def generate_acceptance_criteria(user_message, expected_output,
                                 simple_model, advanced_executor_model,
                                 expert_prompt_acceptance_criteria_model,
                                 expert_prompt_acceptance_criteria_temperature=0.1,
                                 prompt_template_group: Optional[str] = None):
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
                            {"temperature": expert_prompt_acceptance_criteria_temperature})
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
    prompt_template_group: Optional[str] = None
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
        {"temperature": expert_prompt_initial_developer_temperature}
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
    aggressive_exploration: bool = False
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
        raise st.error(f"Error: {e}")           

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

def process_message_with_single_llm(
    user_message: str, expected_output: str, acceptance_criteria: str,
    initial_system_message: str, recursion_limit: int, max_output_age: int,
    model_name: str, prompt_template_group: Optional[str] = None,
    aggressive_exploration: bool = False
) -> tuple:
    llm = initialize_llm(model_name)
    return process_message(
        user_message, expected_output, acceptance_criteria, initial_system_message,
        recursion_limit, max_output_age, llm, prompt_template_group, aggressive_exploration
    )

def process_message_with_2_llms(
    user_message: str, expected_output: str, acceptance_criteria: str,
    initial_system_message: str, recursion_limit: int, max_output_age: int,
    optimizer_model_name: str, executor_model_name: str,
    prompt_template_group: Optional[str] = None,
    aggressive_exploration: bool = False
) -> tuple:
    optimizer_model = initialize_llm(optimizer_model_name)
    executor_model = initialize_llm(executor_model_name)
    llms = {
        NODE_ACCEPTANCE_CRITERIA_DEVELOPER: optimizer_model,
        NODE_PROMPT_INITIAL_DEVELOPER: optimizer_model,
        NODE_PROMPT_DEVELOPER: optimizer_model,
        NODE_PROMPT_EXECUTOR: executor_model,
        NODE_OUTPUT_HISTORY_ANALYZER: optimizer_model,
        NODE_PROMPT_ANALYZER: optimizer_model,
        NODE_PROMPT_SUGGESTER: optimizer_model
    }
    return process_message(
        user_message, expected_output, acceptance_criteria,
        initial_system_message, recursion_limit, max_output_age, llms,
        prompt_template_group, aggressive_exploration
    )

def process_message_with_expert_llms(
    user_message: str, expected_output: str, acceptance_criteria: str,
    initial_system_message: str, recursion_limit: int, max_output_age: int,
    initial_developer_model_name: str, initial_developer_temperature: float,
    acceptance_criteria_model_name: str, acceptance_criteria_temperature: float,
    developer_model_name: str, developer_temperature: float,
    executor_model_name: str, executor_temperature: float,
    output_history_analyzer_model_name: str, output_history_analyzer_temperature: float,
    analyzer_model_name: str, analyzer_temperature: float,
    suggester_model_name: str, suggester_temperature: float,
    prompt_template_group: Optional[str] = None, aggressive_exploration: bool = False
) -> tuple:
    llms = {
        NODE_PROMPT_INITIAL_DEVELOPER: initialize_llm(
            initial_developer_model_name, {"temperature": initial_developer_temperature}
        ),
        NODE_ACCEPTANCE_CRITERIA_DEVELOPER: initialize_llm(
            acceptance_criteria_model_name, {"temperature": acceptance_criteria_temperature}
        ),
        NODE_PROMPT_DEVELOPER: initialize_llm(
            developer_model_name, {"temperature": developer_temperature}
        ),
        NODE_PROMPT_EXECUTOR: initialize_llm(
            executor_model_name, {"temperature": executor_temperature}
        ),
        NODE_OUTPUT_HISTORY_ANALYZER: initialize_llm(
            output_history_analyzer_model_name,
            {"temperature": output_history_analyzer_temperature}
        ),
        NODE_PROMPT_ANALYZER: initialize_llm(
            analyzer_model_name, {"temperature": analyzer_temperature}
        ),
        NODE_PROMPT_SUGGESTER: initialize_llm(
            suggester_model_name, {"temperature": suggester_temperature}
        )
    }
    return process_message(
        user_message,
        expected_output,
        acceptance_criteria,
        initial_system_message,
        recursion_limit,
        max_output_age,
        llms,
        prompt_template_group,
        aggressive_exploration
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

# Streamlit UI
st.title("Meta Prompt")
st.markdown("A tool for generating and analyzing natural language prompts using multiple language models.")

with st.sidebar:
    st.header("Model Settings")
    model_tab = st.selectbox("Select Model Type", ["Simple", "Advanced", "Expert"], key="model_tab")

    if model_tab == "Simple":
        simple_model_name_input = st.selectbox(
            "Model Name",
            config.llms.keys(),
            index=0,
        )
    elif model_tab == "Advanced":
        advanced_optimizer_model_name_input = st.selectbox(
            "Optimizer Model Name",
            config.llms.keys(),
            index=0,
        )
        advanced_executor_model_name_input = st.selectbox(
            "Executor Model Name",
            config.llms.keys(),
            index=1,
        )
    else:  # Expert
        expert_prompt_initial_developer_model_name_input = st.selectbox(
            "Initial Developer Model Name",
            config.llms.keys(),
            index=0,
        )
        expert_prompt_initial_developer_temperature_input = st.slider(
            "Initial Developer Temperature", 0.0, 1.0, 0.1, 0.1
        )

        expert_prompt_acceptance_criteria_model_name_input = st.selectbox(
            "Acceptance Criteria Model Name",
            config.llms.keys(),
            index=0,
        )
        expert_prompt_acceptance_criteria_temperature_input = st.slider(
            "Acceptance Criteria Temperature", 0.0, 1.0, 0.1, 0.1
        )

        expert_prompt_developer_model_name_input = st.selectbox(
            "Developer Model Name", config.llms.keys(), index=0
        )
        expert_prompt_developer_temperature_input = st.slider(
            "Developer Temperature", 0.0, 1.0, 0.1, 0.1
        )

        expert_prompt_executor_model_name_input = st.selectbox(
            "Executor Model Name", config.llms.keys(), index=1
        )
        expert_prompt_executor_temperature_input = st.slider(
            "Executor Temperature", 0.0, 1.0, 0.1, 0.1
        )

        expert_prompt_output_history_analyzer_model_name_input = st.selectbox(
            "Output History Analyzer Model Name",
            config.llms.keys(),
            index=0,
        )
        expert_prompt_output_history_analyzer_temperature_input = st.slider(
            "Output History Analyzer Temperature", 0.0, 1.0, 0.1, 0.1
        )

        expert_prompt_analyzer_model_name_input = st.selectbox(
            "Analyzer Model Name", config.llms.keys(), index=0
        )
        expert_prompt_analyzer_temperature_input = st.slider(
            "Analyzer Temperature", 0.0, 1.0, 0.1, 0.1
        )

        expert_prompt_suggester_model_name_input = st.selectbox(
            "Suggester Model Name", config.llms.keys(), index=0
        )
        expert_prompt_suggester_temperature_input = st.slider(
            "Suggester Temperature", 0.0, 1.0, 0.1, 0.1
        )

    st.header("Prompt Template Settings")
    prompt_template_group_input = st.selectbox(
        "Prompt Template Group", config.prompt_templates.keys(), index=0
    )

    st.header("Advanced Settings")
    recursion_limit_input = st.number_input("Recursion Limit", 1, 100, 16, 1)
    max_output_age_input = st.number_input("Max Output Age", 1, 10, 2, 1)
    aggressive_exploration_input = st.checkbox("Aggressive Exploration", False)

# Initialize session state
if 'shared_input_data' not in st.session_state:
    st.session_state.shared_input_data = pd.DataFrame(columns=["Input", "Output"])
if 'initial_system_message' not in st.session_state:
    st.session_state.initial_system_message = ""
if 'initial_acceptance_criteria' not in st.session_state:
    st.session_state.initial_acceptance_criteria = ""
if 'system_message_output' not in st.session_state:
    st.session_state.system_message_output = ""
if 'output' not in st.session_state:  
    st.session_state.output = ""
if 'analysis' not in st.session_state:
    st.session_state.analysis = ""
if 'acceptance_criteria_output' not in st.session_state:
    st.session_state.acceptance_criteria_output = ""
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

def copy_system_message():
    st.session_state.initial_system_message = system_message_output

def copy_acceptance_criteria():
    st.session_state.initial_acceptance_criteria = acceptance_criteria_output

def clear_session_state():
    st.session_state.shared_input_data = pd.DataFrame(columns=["Input", "Output"])
    st.session_state.initial_system_message = ""
    st.session_state.initial_acceptance_criteria = ""
    st.session_state.system_message_output = ""
    st.session_state.output = ""
    st.session_state.analysis = ""
    st.session_state.acceptance_criteria_output = ""
    st.session_state.chat_log = []

def sync_input_data():
    st.session_state.shared_input_data = data_editor_data.copy()

def pull_sample_description():
    if 'description_output_text' in st.session_state:
        st.session_state.initial_system_message = st.session_state.description_output_text

def generate_callback():
    try:
        first_input_key = data_editor_data["Input"].first_valid_index()
        first_output_key = data_editor_data["Output"].first_valid_index()
        user_message = data_editor_data["Input"][first_input_key].strip()
        expected_output = data_editor_data["Output"][first_output_key].strip()

        input_acceptance_criteria = initial_acceptance_criteria.strip() if 'initial_acceptance_criteria' in st.session_state else ""
        input_system_message = initial_system_message.strip() if 'initial_system_message' in st.session_state else ""

        if active_model_tab == "Simple":
            system_message, output, analysis, acceptance_criteria, chat_log = process_message_with_single_llm(
                user_message,
                expected_output,
                input_acceptance_criteria,
                input_system_message,
                recursion_limit,
                max_output_age,
                simple_model_name,
                prompt_template_group,
                aggressive_exploration,
            )
        elif active_model_tab == "Advanced":
            system_message, output, analysis, acceptance_criteria, chat_log = process_message_with_2_llms(
                user_message,
                expected_output,
                input_acceptance_criteria,
                input_system_message,
                recursion_limit,
                max_output_age,
                advanced_optimizer_model_name_input,
                advanced_executor_model_name_input,
                prompt_template_group,
                aggressive_exploration,
            )
        else:  # Expert
            system_message, output, analysis, acceptance_criteria, chat_log = process_message_with_expert_llms(
                user_message,
                expected_output,
                input_acceptance_criteria,
                input_system_message,
                recursion_limit,
                max_output_age,
                expert_prompt_initial_developer_model_name,
                expert_prompt_initial_developer_temperature_input,
                expert_prompt_acceptance_criteria_model_name,
                expert_prompt_acceptance_criteria_temperature_input,
                expert_prompt_developer_model_name,
                expert_prompt_developer_temperature_input,
                expert_prompt_executor_model_name,
                expert_prompt_executor_temperature_input,
                expert_prompt_output_history_analyzer_model_name,
                expert_prompt_output_history_analyzer_temperature_input,
                expert_prompt_analyzer_model_name,
                expert_prompt_analyzer_temperature_input,
                expert_prompt_suggester_model_name,
                expert_prompt_suggester_temperature_input,
                prompt_template_group,
                aggressive_exploration,
            )
        
        st.session_state.system_message_output = system_message
        st.session_state.output = output
        st.session_state.analysis = analysis
        st.session_state.acceptance_criteria_output = acceptance_criteria
        st.session_state.chat_log = chat_log

    except Exception as e:
        st.error(f"Error: {e}")


if active_model_tab == "Simple":
    simple_model_name = simple_model_name_input
    advanced_executor_model_name = None
    expert_prompt_initial_developer_model_name = None
    expert_prompt_acceptance_criteria_model_name = None
    expert_prompt_developer_model_name = None
    expert_prompt_executor_model_name = None
    expert_prompt_output_history_analyzer_model_name = None
    expert_prompt_analyzer_model_name = None
    expert_prompt_suggester_model_name = None
elif active_model_tab == "Advanced":
    simple_model_name = None
    advanced_executor_model_name = advanced_executor_model_name_input
    expert_prompt_initial_developer_model_name = None
    expert_prompt_acceptance_criteria_model_name = None
    expert_prompt_developer_model_name = None
    expert_prompt_executor_model_name = None
    expert_prompt_output_history_analyzer_model_name = None
    expert_prompt_analyzer_model_name = None
    expert_prompt_suggester_model_name = None
else:  # Expert
    simple_model_name = None
    advanced_executor_model_name = None
    expert_prompt_initial_developer_model_name = (
        expert_prompt_initial_developer_model_name_input
    )
    expert_prompt_acceptance_criteria_model_name = (
        expert_prompt_acceptance_criteria_model_name_input
    )
    expert_prompt_developer_model_name = expert_prompt_developer_model_name_input
    expert_prompt_executor_model_name = expert_prompt_executor_model_name_input
    expert_prompt_output_history_analyzer_model_name = (
        expert_prompt_output_history_analyzer_model_name_input
    )
    expert_prompt_analyzer_model_name = expert_prompt_analyzer_model_name_input
    expert_prompt_suggester_model_name = expert_prompt_suggester_model_name_input

prompt_template_group = prompt_template_group_input
recursion_limit = recursion_limit_input
max_output_age = max_output_age_input
aggressive_exploration = aggressive_exploration_input

data_editor_data = st.data_editor(
    st.session_state.shared_input_data,
    # key="meta_prompt_input_data",
    num_rows="dynamic",
    column_config={
        "Input": st.column_config.TextColumn("Input", width="large"),
        "Output": st.column_config.TextColumn("Output", width="large"),
    },
    hide_index=False,
    use_container_width=True,
)

col1, col2 = st.columns(2)

with col1:
    with st.expander("Advanced Inputs"):

        initial_system_message = st.text_area(
            "Initial System Message",
            # "Default System Message",
            # st.session_state.initial_system_message,
            key="initial_system_message"
        )

        col1_1, col1_2 = st.columns(2)
        with col1_1:
            pull_sample_description_button = st.button("Pull Sample Description", key="pull_sample_description",
                                                       on_click=pull_sample_description)
        with col1_2:
            st.button("Pull Output", key="copy_system_message",
                      on_click=copy_system_message)
        initial_acceptance_criteria = st.text_area(
            "Acceptance Criteria",
            # "Default Acceptance Criteria",
            # st.session_state.initial_acceptance_criteria,
            key="initial_acceptance_criteria"
        )
        st.button("Pull Output", key="copy_acceptance_criteria",
                  on_click=copy_acceptance_criteria)

    col1_1, col1_2, col1_3 = st.columns(3)
    with col1_1:
        generate_button_clicked = st.button("Generate", key="generate_button",
                                            on_click=generate_callback,
                                            type="primary")
    with col1_2:
        sync_button_clicked = st.button("Sync Data", on_click=sync_input_data)
    with col1_3:
        clear_button_clicked = st.button("Clear", on_click=clear_session_state)

with col2:
    system_message_output = st.text_area("System Message",
                                        # st.session_state.system_message_output,
                 key="system_message_output",
                 height=100)

    acceptance_criteria_output = st.text_area(
        "Acceptance Criteria",
        # st.session_state.acceptance_criteria_output,
        key="acceptance_criteria_output",
        height=100)
    st.text_area("Output", st.session_state.output, height=100)
    st.text_area("Analysis", st.session_state.analysis, height=100)

    st.json(st.session_state.chat_log)
