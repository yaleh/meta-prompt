import pandas as pd
import streamlit as st
import json
from app.meta_prompt_utils import *
from meta_prompt.sample_generator import TaskDescriptionGenerator

# Initialize session state
def init_session_state():
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
    if 'description_output_text' not in st.session_state:
        st.session_state.description_output_text = ''
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'input_analysis_output_text' not in st.session_state:
        st.session_state.input_analysis_output_text = ''
    if 'example_briefs_output_text' not in st.session_state:
        st.session_state.example_briefs_output_text = ''
    if 'examples_from_briefs_dataframe' not in st.session_state:
        st.session_state.examples_from_briefs_dataframe = pd.DataFrame(columns=["Input", "Output"])
    if 'examples_directly_dataframe' not in st.session_state:
        st.session_state.examples_directly_dataframe = pd.DataFrame(columns=["Input", "Output"])
    if 'examples_dataframe' not in st.session_state:
        st.session_state.examples_dataframe = pd.DataFrame(columns=["Input", "Output"])
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None

# UI helper functions
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def sync_input_data():
    st.session_state.shared_input_data = st.session_state.data_editor_data.copy()

# Sample Generator Functions

def process_json(input_json, model_name, generating_batch_size, temperature):
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
        st.warning(f"An error occurred: {str(e)}. Returning default values.")
        return "", [], [], "", [], [], []


def generate_description_only(input_json, model_name, temperature):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_description(input_json)
        description = result["description"]
        suggestions = result["suggestions"]
        return description, suggestions
    except Exception as e:
        st.warning(f"An error occurred: {str(e)}")
        return "", []


def analyze_input(description, model_name, temperature):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        input_analysis = generator.analyze_input(description)
        return input_analysis
    except Exception as e:
        st.warning(f"An error occurred: {str(e)}")
        return ""


def generate_briefs(description, input_analysis, generating_batch_size, model_name, temperature):
    try:
        model = ChatOpenAI(
            model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        briefs = generator.generate_briefs(
            description, input_analysis, generating_batch_size)
        return briefs
    except Exception as e:
        st.warning(f"An error occurred: {str(e)}")
        return ""


def generate_examples_from_briefs(description, new_example_briefs, input_str, generating_batch_size, model_name, temperature):
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
        st.warning(f"An error occurred: {str(e)}")
        return []


def generate_examples_directly(description, raw_example, generating_batch_size, model_name, temperature):
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
        st.warning(f"An error occurred: {str(e)}")
        return []


def example_directly_selected():
    if 'selected_example_directly_id' in st.session_state:
        try:
            selected_example_ids = st.session_state.selected_example_directly_id[
                'selection']['rows']
            # set selected examples to the selected rows if there are any
            if selected_example_ids:
                selected_examples = st.session_state.examples_directly_dataframe.iloc[selected_example_ids].to_dict(
                    'records')
                st.session_state.selected_example = pd.DataFrame(selected_examples)  # Convert to DataFrame
            else:
                st.session_state.selected_example = None
        except Exception as e:
            st.session_state.selected_example = None


def example_from_briefs_selected():
    if 'selected_example_from_briefs_id' in st.session_state:
        try:
            selected_example_ids = st.session_state.selected_example_from_briefs_id[
                'selection']['rows']
            # set selected examples to the selected rows if there are any
            if selected_example_ids:
                selected_examples = st.session_state.examples_from_briefs_dataframe.iloc[selected_example_ids].to_dict(
                    'records')
                st.session_state.selected_example = pd.DataFrame(selected_examples)  # Convert to DataFrame
            else:
                st.session_state.selected_example = None
        except Exception as e:
            st.session_state.selected_example = None


def example_selected():
    if 'selected_example_id' in st.session_state:
        try:
            selected_example_ids = st.session_state.selected_example_id['selection']['rows']
            # set selected examples to the selected rows if there are any
            if selected_example_ids:
                selected_examples = st.session_state.examples_dataframe.iloc[selected_example_ids].to_dict(
                    'records')
                st.session_state.selected_example = pd.DataFrame(selected_examples)  # Convert to DataFrame
            else:
                st.session_state.selected_example = None
        except Exception as e:
            st.session_state.selected_example = None

def update_description_output_text():
    input_json = package_input_data()
    result = generate_description_only(input_json, model_name, temperature)
    st.session_state.description_output_text = result[0]
    st.session_state.suggestions = result[1]


def update_input_analysis_output_text():
    st.session_state.input_analysis_output_text = analyze_input(
        description_output, model_name, temperature)


def update_example_briefs_output_text():
    st.session_state.example_briefs_output_text = generate_briefs(
        description_output, input_analysis_output, generating_batch_size, model_name, temperature)


def update_examples_from_briefs_dataframe():
    input_json = package_input_data()
    examples = generate_examples_from_briefs(
        description_output, example_briefs_output, input_json, generating_batch_size, model_name, temperature)
    st.session_state.examples_from_briefs_dataframe = pd.DataFrame(
        examples, columns=["Input", "Output"])


def update_examples_directly_dataframe():
    input_json = package_input_data()
    examples = generate_examples_directly(
        description_output, input_json, generating_batch_size, model_name, temperature)
    st.session_state.examples_directly_dataframe = pd.DataFrame(
        examples, columns=["Input", "Output"])


def generate_examples_dataframe():
    input_json = package_input_data()
    result = process_json(input_json, model_name,
                          generating_batch_size, temperature)
    description, suggestions, examples_directly, input_analysis, new_example_briefs, examples_from_briefs, examples = result
    st.session_state.description_output_text = description
    st.session_state.suggestions = suggestions  # Ensure suggestions are stored in session state
    st.session_state.examples_directly_dataframe = pd.DataFrame(
        examples_directly, columns=["Input", "Output"])
    st.session_state.input_analysis_output_text = input_analysis
    st.session_state.example_briefs_output_text = new_example_briefs
    st.session_state.examples_from_briefs_dataframe = pd.DataFrame(
        examples_from_briefs, columns=["Input", "Output"])
    st.session_state.examples_dataframe = pd.DataFrame(
        examples, columns=["Input", "Output"])
    st.session_state.selected_example = None

def package_input_data():
    data = data_editor_data.to_dict(orient='records')
    lowered_data = [{k.lower(): v for k, v in d.items()} for d in data]
    return json.dumps(lowered_data, ensure_ascii=False)

def export_input_data_to_json():
    input_data_json = package_input_data()
    st.download_button(
        label="Download input data as JSON",
        data=input_data_json,
        file_name="input_data.json",
        mime="application/json"
    )

def import_input_data_from_json():
    try:
        if 'input_file' in st.session_state and st.session_state.input_file is not None:
            data = st.session_state.input_file.getvalue()
            data = json.loads(data)
            data = [{k.capitalize(): v for k, v in d.items()} for d in data]
            st.session_state.shared_input_data = pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Failed to import JSON: {str(e)}")

def apply_suggestions():
    try:
        result = TaskDescriptionGenerator(
            ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)).update_description(
            package_input_data(), st.session_state.description_output_text, st.session_state.selected_suggestions)
        st.session_state.description_output_text = result["description"]
        st.session_state.suggestions = result["suggestions"]
    except Exception as e:
        st.warning(f"Failed to update description: {str(e)}")

def generate_suggestions():
    try:
        description = st.session_state.description_output_text
        input_json = package_input_data()

        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_suggestions(input_json, description)
        st.session_state.suggestions = result["suggestions"]
    except Exception as e:
        st.warning(f"Failed to generate suggestions: {str(e)}")

# Function to add new suggestion to the list and select it
def add_new_suggestion():
    if st.session_state.new_suggestion:
        st.session_state.suggestions.append(st.session_state.new_suggestion)
        st.session_state.new_suggestion = ""  # Clear the input field

def append_selected_to_input_data():
    if st.session_state.selected_example is not None:
        st.session_state.shared_input_data = pd.concat(
            [data_editor_data, st.session_state.selected_example], ignore_index=True)
        st.session_state.selected_example = None

def show_scoping_sidebar():
    if st.session_state.selected_example is not None:
        with st.sidebar:
            st.dataframe(st.session_state.selected_example, hide_index=False)
            st.button("Append to Input Data", on_click=append_selected_to_input_data)

# Meta Prompt Functions
def process_message_with_single_llm(
    user_message: str, expected_output: str, acceptance_criteria: str,
    initial_system_message: str, recursion_limit: int, max_output_age: int,
    model_name: str, prompt_template_group: Optional[str] = None,
    aggressive_exploration: bool = False, config: MetaPromptConfig = None
) -> tuple:
    llm = initialize_llm(model_name, config=config)
    return process_message(
        user_message, expected_output, acceptance_criteria, initial_system_message,
        recursion_limit, max_output_age, llm, prompt_template_group, aggressive_exploration,
        config
    )

def process_message_with_2_llms(
    user_message: str, expected_output: str, acceptance_criteria: str,
    initial_system_message: str, recursion_limit: int, max_output_age: int,
    optimizer_model_name: str, executor_model_name: str,
    prompt_template_group: Optional[str] = None,
    aggressive_exploration: bool = False, config: MetaPromptConfig = None
) -> tuple:
    optimizer_model = initialize_llm(optimizer_model_name, config=config)
    executor_model = initialize_llm(executor_model_name, config=config)
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
        prompt_template_group, aggressive_exploration, config
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
    prompt_template_group: Optional[str] = None, aggressive_exploration: bool = False,
    config: MetaPromptConfig = None
) -> tuple:
    llms = {
        NODE_PROMPT_INITIAL_DEVELOPER: initialize_llm(
            initial_developer_model_name, {"temperature": initial_developer_temperature}, config
        ),
        NODE_ACCEPTANCE_CRITERIA_DEVELOPER: initialize_llm(
            acceptance_criteria_model_name, {"temperature": acceptance_criteria_temperature}, config
        ),
        NODE_PROMPT_DEVELOPER: initialize_llm(
            developer_model_name, {"temperature": developer_temperature}, config
        ),
        NODE_PROMPT_EXECUTOR: initialize_llm(
            executor_model_name, {"temperature": executor_temperature}, config
        ),
        NODE_OUTPUT_HISTORY_ANALYZER: initialize_llm(
            output_history_analyzer_model_name,
            {"temperature": output_history_analyzer_temperature},
            config
        ),
        NODE_PROMPT_ANALYZER: initialize_llm(
            analyzer_model_name, {"temperature": analyzer_temperature}, config
        ),
        NODE_PROMPT_SUGGESTER: initialize_llm(
            suggester_model_name, {"temperature": suggester_temperature}, config
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
        aggressive_exploration,
        config
    )

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

def pull_sample_description():
    st.session_state.initial_system_message = description_output

def update_working_sample_options():
    pass

def generate_callback():
    try:
        # Get the index of the selected sample
        selected_index = selected_sample.split(":")[0].split()[1]
        selected_index = int(selected_index)

        user_message = data_editor_data.loc[selected_index, "Input"].strip()
        expected_output = data_editor_data.loc[selected_index, "Output"].strip()

        input_acceptance_criteria = initial_acceptance_criteria.strip()
        input_system_message = initial_system_message.strip()

        if model_tab == "Simple":
            system_message, output, analysis, acceptance_criteria, chat_log = process_message_with_single_llm(
                user_message,
                expected_output,
                input_acceptance_criteria,
                input_system_message,
                recursion_limit_input,
                max_output_age_input,
                simple_model_name_input,
                prompt_template_group_input,
                aggressive_exploration_input,
                config=config
            )
        elif model_tab == "Advanced":
            system_message, output, analysis, acceptance_criteria, chat_log = process_message_with_2_llms(
                user_message,
                expected_output,
                input_acceptance_criteria,
                input_system_message,
                recursion_limit_input,
                max_output_age_input,
                advanced_optimizer_model_name_input,
                advanced_executor_model_name_input,
                prompt_template_group_input,
                aggressive_exploration_input,
                config=config
            )
        else:  # Expert
            system_message, output, analysis, acceptance_criteria, chat_log = process_message_with_expert_llms(
                user_message,
                expected_output,
                input_acceptance_criteria,
                input_system_message,
                recursion_limit_input,
                max_output_age_input,
                expert_prompt_initial_developer_model_name_input,
                expert_prompt_initial_developer_temperature_input,
                expert_prompt_acceptance_criteria_model_name_input,
                expert_prompt_acceptance_criteria_temperature_input,
                expert_prompt_developer_model_name_input,
                expert_prompt_developer_temperature_input,
                expert_prompt_executor_model_name_input,
                expert_prompt_executor_temperature_input,
                expert_prompt_output_history_analyzer_model_name_input,
                expert_prompt_output_history_analyzer_temperature_input,
                expert_prompt_analyzer_model_name_input,
                expert_prompt_analyzer_temperature_input,
                expert_prompt_suggester_model_name_input,
                expert_prompt_suggester_temperature_input,
                prompt_template_group_input,
                aggressive_exploration_input,
                config=config
            )
        
        st.session_state.system_message_output = system_message
        st.session_state.output = output
        st.session_state.analysis = analysis
        st.session_state.acceptance_criteria_output = acceptance_criteria
        st.session_state.chat_log = chat_log

    except Exception as e:
        st.error(f"Error: {e}")

# Meta Prompt Config

pre_config_sources = [
    EnvSource(prefix='METAPROMPT_', allow_all=True),
    CLArgSource()
]
pre_config = FileConfig(config_sources=pre_config_sources)

# Load configuration
config = MetaPromptConfig(config_sources=[
    FileSource(file=pre_config.config_file, optional=True),
    EnvSource(prefix='METAPROMPT_', allow_all=True),
    CLArgSource()
])

# Initialize session state
init_session_state()

# Streamlit UI

st.title("Meta Prompt")
st.markdown("Enter input-output pairs as the examples for the prompt.")
data_editor_data = st.data_editor(
    st.session_state.shared_input_data,
    key="data_editor",
    num_rows="dynamic",
    column_config={
        "Input": st.column_config.TextColumn("Input", width="large"),
        "Output": st.column_config.TextColumn("Output", width="large"),
    },
    hide_index=False,
    use_container_width=True,
    on_change=update_working_sample_options
)

with st.expander("Data Management"):
    input_file = st.file_uploader(
        label="Import Input Data from JSON",
        type="json",
        key="input_file",
        on_change=import_input_data_from_json
    )
    export_button = st.button(  # Add the export button
        "Export Input Data to JSON", on_click=export_input_data_to_json
    )

tab_scoping, tab_prompting = st.tabs(["Scope", "Prompt"])

with tab_scoping:
    # Streamlit UI
    st.markdown("Define the task scope using the above input-output pairs.")

    submit_button = st.button(
        "Go", type="primary", on_click=generate_examples_dataframe,
        use_container_width=True)

    with st.expander("Model Settings"):
        model_name = st.selectbox(
            "Model Name",
            ["llama3-70b-8192", "llama3-8b-8192", "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant", "gemma2-9b-it"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)
        generating_batch_size = st.slider("Generating Batch Size", 1, 10, 3, 1)

    with st.expander("Description and Analysis"):
        generate_description_button = st.button(
            "Generate Description", on_click=update_description_output_text)
        
        description_output = st.text_area(
            "Description", value=st.session_state.description_output_text, height=100)

        col3, col4, col5 = st.columns(3)
        with col3:
            generate_suggestions_button = st.button("Generate Suggestions", on_click=generate_suggestions)
        with col4:
            generate_examples_directly_button = st.button(
                "Generate Examples Directly", on_click=update_examples_directly_dataframe)
        with col5:
            analyze_input_button = st.button(
                "Analyze Input", on_click=update_input_analysis_output_text)

        # Add multiselect for suggestions
        selected_suggestions = st.multiselect(
            "Suggestions", options=st.session_state.suggestions, key="selected_suggestions")

        # Add button to apply suggestions
        apply_suggestions_button = st.button("Apply Suggestions", on_click=apply_suggestions)

        # Add text input for adding new suggestions
        new_suggestion = st.text_input("Add New Suggestion", key="new_suggestion", on_change=add_new_suggestion)

        examples_directly_output = st.dataframe(st.session_state.examples_directly_dataframe, use_container_width=True,
                                                selection_mode="multi-row", key="selected_example_directly_id",
                                                on_select=example_directly_selected, hide_index=False)
        input_analysis_output = st.text_area(
            "Input Analysis", value=st.session_state.input_analysis_output_text, height=100)
        generate_briefs_button = st.button(
            "Generate Briefs", on_click=update_example_briefs_output_text)
        example_briefs_output = st.text_area(
            "Example Briefs", value=st.session_state.example_briefs_output_text, height=100)
        generate_examples_from_briefs_button = st.button(
            "Generate Examples from Briefs", on_click=update_examples_from_briefs_dataframe)
        examples_from_briefs_output = st.dataframe(st.session_state.examples_from_briefs_dataframe, use_container_width=True,
                                                selection_mode="multi-row", key="selected_example_from_briefs_id",
                                                on_select=example_from_briefs_selected, hide_index=False)

    examples_output = st.dataframe(st.session_state.examples_dataframe, use_container_width=True,
                                selection_mode="multi-row", key="selected_example_id", on_select=example_selected, hide_index=True)

    show_scoping_sidebar()

with tab_prompting:
    # Prompting UI
    st.markdown("Generate the prompt with the above input-output pairs.")

    # Create options for the selectbox
    sample_options = [f"Sample {i}: {row['Input'][:30]}..." for i, row in data_editor_data.iterrows()]

    # Create the selectbox
    selected_sample = st.selectbox(
        "Working Sample",
        options=sample_options,
        index=0,
        # key="working_sample"
    )

    generate_button_clicked = st.button("Generate", key="generate_button",
                                        on_click=generate_callback,
                                        type="primary", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Advanced Inputs"):
            initial_system_message = st.text_area(
                "Initial System Message",
                key="initial_system_message"
            )

            col1_1, col1_2 = st.columns(2)
            with col1_1:
                pull_sample_description_button = st.button("Pull Scope Description", key="pull_sample_description",
                                                        on_click=pull_sample_description)
            with col1_2:
                st.button("Pull Output", key="copy_system_message",
                        on_click=copy_system_message)
            initial_acceptance_criteria = st.text_area(
                "Acceptance Criteria",
                key="initial_acceptance_criteria"
            )
            st.button("Pull Output", key="copy_acceptance_criteria",
                    on_click=copy_acceptance_criteria)

        # New expander for model settings
        with st.expander("Model Settings"):
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

            prompt_template_group_input = st.selectbox(
                "Prompt Template Group", config.prompt_templates.keys(), index=0
            )

            recursion_limit_input = st.number_input("Recursion Limit", 1, 100, 16, 1)
            max_output_age_input = st.number_input("Max Output Age", 1, 10, 2, 1)
            aggressive_exploration_input = st.checkbox("Aggressive Exploration", False)

    with col2:
        system_message_output = st.text_area("System Message",
                                            key="system_message_output",
                                            height=100)

        acceptance_criteria_output = st.text_area(
            "Acceptance Criteria",
            key="acceptance_criteria_output",
            height=100)
        st.text_area("Output", st.session_state.output, height=100)
        st.text_area("Analysis", st.session_state.analysis, height=100)

        st.json(st.session_state.chat_log, expanded=False)
