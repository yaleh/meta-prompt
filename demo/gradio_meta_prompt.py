import gradio as gr
from confz import BaseConfig, CLArgSource, EnvSource, FileSource
from meta_prompt import MetaPromptGraph, AgentState
from langchain_openai import ChatOpenAI
from config import MetaPromptConfig

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
iface = gr.Interface(
    fn=process_message,
    inputs=[
        gr.Textbox(label="User Message", show_copy_button=True),
        gr.Textbox(label="Expected Output", show_copy_button=True),
        gr.Textbox(label="Acceptance Criteria", show_copy_button=True),
    ],
    outputs=[
        gr.Textbox(label="System Message", show_copy_button=True),
        gr.Textbox(label="Output", show_copy_button=True),
        gr.Textbox(label="Analysis", show_copy_button=True)
    ],
    additional_inputs=[
        gr.Textbox(label="Initial System Message", show_copy_button=True, value=""),
        gr.Number(label="Recursion Limit", value=25,
                  precision=0, minimum=1, maximum=100, step=1),
        gr.Dropdown(
            label="Model Name",
            choices=config.llms.keys(),
            value=list(config.llms.keys())[0],
        )
    ],
    # stop_btn = gr.Button("Stop", variant="stop", visible=True),
    title="MetaPromptGraph Chat Interface",
    description="A chat interface for MetaPromptGraph to process user inputs and generate system messages.",
    examples=config.examples_path
)

# Launch the Gradio app
iface.launch(server_name=config.server_name, server_port=config.server_port)
