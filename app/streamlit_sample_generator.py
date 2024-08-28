import pandas as pd
import streamlit as st
import json
from langchain_community.chat_models import ChatOpenAI
from meta_prompt.sample_generator import TaskDescriptionGenerator

def process_json(input_json, model_name, generating_batch_size, temperature):
    try:
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.process(input_json, generating_batch_size)
        description = result["description"]
        examples_directly = [[example["input"], example["output"]] for example in result["examples_directly"]["examples"]]
        input_analysis = result["examples_from_briefs"]["input_analysis"]
        new_example_briefs = result["examples_from_briefs"]["new_example_briefs"]
        examples_from_briefs = [[example["input"], example["output"]] for example in result["examples_from_briefs"]["examples"]]
        examples = [[example["input"], example["output"]] for example in result["additional_examples"]]
        return description, examples_directly, input_analysis, new_example_briefs, examples_from_briefs, examples
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
def generate_description_only(input_json, model_name, temperature):
    try:
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        description = generator.generate_description(input_json)
        return description
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def analyze_input(description, model_name, temperature):
    try:
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        input_analysis = generator.analyze_input(description)
        return input_analysis
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
def generate_briefs(description, input_analysis, generating_batch_size, model_name, temperature):
    try:
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        briefs = generator.generate_briefs(description, input_analysis, generating_batch_size)
        return briefs
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
def generate_examples_from_briefs(description, new_example_briefs, input_str, generating_batch_size, model_name, temperature):
    try:
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_examples_from_briefs(description, new_example_briefs, input_str, generating_batch_size)
        examples = [[example["input"], example["output"]] for example in result["examples"]]
        return examples
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
def generate_examples_directly(description, raw_example, generating_batch_size, model_name, temperature):
    try:
        model = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
        generator = TaskDescriptionGenerator(model)
        result = generator.generate_examples_directly(description, raw_example, generating_batch_size)
        examples = [[example["input"], example["output"]] for example in result["examples"]]
        return examples
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Session State
if 'description_output_text' not in st.session_state:
    st.session_state.description_output_text = ''

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

def update_description_output_text():
    st.session_state.description_output_text = generate_description_only(input_json, model_name, temperature)

def update_input_analysis_output_text():
    st.session_state.input_analysis_output_text = analyze_input(description_output, model_name, temperature)

def update_example_briefs_output_text():
    st.session_state.example_briefs_output_text = generate_briefs(description_output, input_analysis_output, generating_batch_size, model_name, temperature)

def update_examples_from_briefs_dataframe():
    st.session_state.examples_from_briefs_dataframe = generate_examples_from_briefs(description_output, example_briefs_output, input_json, generating_batch_size, model_name, temperature)

def update_examples_directly_dataframe():
    st.session_state.examples_directly_dataframe = generate_examples_directly(description_output, input_json, generating_batch_size, model_name, temperature)

def generate_examples_dataframe():
    result = process_json(input_json, model_name, generating_batch_size, temperature)
    description, examples_directly, input_analysis, new_example_briefs, examples_from_briefs, examples = result
    st.session_state.description_output_text = description
    st.session_state.examples_directly_dataframe = examples_directly
    st.session_state.input_analysis_output_text = input_analysis
    st.session_state.example_briefs_output_text = new_example_briefs
    st.session_state.examples_from_briefs_dataframe = examples_from_briefs
    st.session_state.examples_dataframe = examples

# Streamlit UI
st.title("Task Description Generator")
st.markdown("Enter a JSON object with 'input' and 'output' fields to generate a task description and additional examples.")

# Input column
input_json = st.text_area("Input JSON", height=200)
model_name = st.selectbox(
    "Model Name",
    ["llama3-70b-8192", "llama3-8b-8192", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
    index=0
)
temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)
generating_batch_size = st.slider("Generating Batch Size", 1, 10, 3, 1)

# Buttons
col1, col2 = st.columns(2)
with col1:
    submit_button = st.button("Generate", type="primary", on_click=generate_examples_dataframe)
with col2:
    generate_description_button = st.button("Generate Description", on_click=update_description_output_text)

# Output column

description_output = st.text_area("Description", value=st.session_state.description_output_text, height=100)

col3, col4 = st.columns(2)
with col3:
    generate_examples_directly_button = st.button("Generate Examples Directly", on_click=update_examples_directly_dataframe)
with col4:
    analyze_input_button = st.button("Analyze Input", on_click=update_input_analysis_output_text)

examples_directly_output = st.dataframe(st.session_state.examples_directly_dataframe, use_container_width=True)
input_analysis_output = st.text_area("Input Analysis", value=st.session_state.input_analysis_output_text, height=100)
generate_briefs_button = st.button("Generate Briefs", on_click=update_example_briefs_output_text)
example_briefs_output = st.text_area("Example Briefs", value=st.session_state.example_briefs_output_text, height=100)
generate_examples_from_briefs_button = st.button("Generate Examples from Briefs", on_click=update_examples_from_briefs_dataframe)
examples_from_briefs_output = st.dataframe(st.session_state.examples_from_briefs_dataframe, use_container_width=True)
examples_output = st.dataframe(st.session_state.examples_dataframe, use_container_width=True)
new_example_json = st.text_area("New Example JSON", height=100)

# Button actions
if submit_button:
    try:
        result = process_json(
            input_json, model_name, generating_batch_size, temperature
        )
        description, examples_directly, input_analysis, new_example_briefs, examples_from_briefs, examples = result
        description_output = description
        examples_directly_output = examples_directly
        input_analysis_output = input_analysis
        example_briefs_output = new_example_briefs
        examples_from_briefs_output = examples_from_briefs
        examples_output = examples
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if generate_examples_directly_button:
    examples_directly_output = generate_examples_directly(description_output, input_json, generating_batch_size, model_name, temperature)

if analyze_input_button:
    input_analysis_output = analyze_input(description_output, model_name, temperature)

if generate_briefs_button:
    example_briefs_output = generate_briefs(description_output, input_analysis_output, generating_batch_size, model_name, temperature)

if generate_examples_from_briefs_button:
    examples_from_briefs_output = generate_examples_from_briefs(description_output, example_briefs_output, input_json, generating_batch_size, model_name, temperature)
