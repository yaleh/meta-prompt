import pandas as pd
import streamlit as st
import json
from langchain_openai import ChatOpenAI
from meta_prompt.sample_generator import TaskDescriptionGenerator


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


# Session State
if 'sample_generator_input_data' not in st.session_state:
    st.session_state.sample_generator_input_data = pd.DataFrame(columns=["Input", "Output"])

if 'description_output_text' not in st.session_state:
    st.session_state.description_output_text = ''

if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

if 'input_analysis_output_text' not in st.session_state:
    st.session_state.input_analysis_output_text = ''

if 'example_briefs_output_text' not in st.session_state:
    st.session_state.example_briefs_output_text = ''

if 'examples_from_briefs_dataframe' not in st.session_state:
    st.session_state.examples_from_briefs_dataframe = pd.DataFrame(columns=[
                                                                   "Input", "Output"])

if 'examples_directly_dataframe' not in st.session_state:
    st.session_state.examples_directly_dataframe = pd.DataFrame(
        columns=["Input", "Output"])

if 'examples_dataframe' not in st.session_state:
    st.session_state.examples_dataframe = pd.DataFrame(
        columns=["Input", "Output"])

if 'selected_example' not in st.session_state:
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
    data = input_data.to_dict(orient='records')
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
            st.session_state.sample_generator_input_data = pd.DataFrame(data)
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

def sync_input_data():
    st.session_state.meta_prompt_input_data = input_data.copy()
    st.session_state.sample_generator_input_data = input_data.copy()

def clear_session_state():
    st.session_state.sample_generator_input_data = pd.DataFrame(columns=["Input", "Output"])
    st.session_state.description_output_text = ''
    st.session_state.suggestions = []
    st.session_state.input_analysis_output_text = ''
    st.session_state.example_briefs_output_text = ''
    st.session_state.examples_from_briefs_dataframe = pd.DataFrame(columns=["Input", "Output"])
    st.session_state.examples_directly_dataframe = pd.DataFrame(columns=["Input", "Output"])
    st.session_state.examples_dataframe = pd.DataFrame(columns=["Input", "Output"])
    st.session_state.selected_example = None

# Streamlit UI
st.title("LLM Task Example Generator")
st.markdown("Enter input-output pairs in the table below to generate a task description, analysis, and additional examples.")

# Input column
input_data = st.data_editor(
    st.session_state.sample_generator_input_data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Input": st.column_config.TextColumn("Input", width="large"),
        "Output": st.column_config.TextColumn("Output", width="large"),
    },
)

with st.expander("Model Settings"):
    col1, col2 = st.columns(2)
    with col1:
        input_file = st.file_uploader(
            label="Import Input Data from JSON",
            type="json",
            key="input_file",
            on_change=import_input_data_from_json
        )
    with col2:
        export_button = st.button(  # Add the export button
            "Export Input Data to JSON", on_click=export_input_data_to_json
        )

    model_name = st.selectbox(
        "Model Name",
        ["llama3-70b-8192", "llama3-8b-8192", "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant", "gemma2-9b-it"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)
    generating_batch_size = st.slider("Generating Batch Size", 1, 10, 3, 1)

col1, col2, col3 = st.columns(3)
with col1:
    submit_button = st.button(
        "Generate", type="primary", on_click=generate_examples_dataframe)
with col2:
    sync_button = st.button(
        "Sync", on_click=sync_input_data)
with col3:
    clear_button = st.button(
        "Clear", on_click=clear_session_state)

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
                                            on_select=example_directly_selected)
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
                                               on_select=example_from_briefs_selected)

examples_output = st.dataframe(st.session_state.examples_dataframe, use_container_width=True,
                               selection_mode="multi-row", key="selected_example_id", on_select=example_selected)

def append_selected_to_input_data():
    if st.session_state.selected_example is not None:
        st.session_state.sample_generator_input_data = pd.concat(
            [st.session_state.sample_generator_input_data, st.session_state.selected_example], ignore_index=True)
        st.session_state.selected_example = None

def show_sidebar():
    if st.session_state.selected_example is not None:
        with st.sidebar:
            st.dataframe(st.session_state.selected_example)  # Display DataFrame in sidebar
            st.button("Append to Input Data", on_click=append_selected_to_input_data)

show_sidebar()

