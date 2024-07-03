import gradio as gr
from meta_prompt import MetaPromptGraph, AgentState
from langchain_openai import ChatOpenAI

# Initialize the MetaPromptGraph with the required LLMs
MODEL_NAME = "anthropic/claude-3.5-sonnet:haiku"
# MODEL_NAME = "meta-llama/llama-3-70b-instruct"
# MODEL_NAME = "deepseek/deepseek-chat"
# MODEL_NAME = "google/gemma-2-9b-it"
# MODEL_NAME = "recursal/eagle-7b"
# MODEL_NAME = "meta-llama/llama-3-8b-instruct"
llm = ChatOpenAI(model_name=MODEL_NAME)
meta_prompt_graph = MetaPromptGraph(llms=llm)

def process_message(user_message, expected_output, acceptance_criteria, recursion_limit: int=25):
    # Create the input state
    input_state = AgentState(
        user_message=user_message,
        expected_output=expected_output,
        acceptance_criteria=acceptance_criteria
    )
    
    # Get the output state from MetaPromptGraph
    output_state = meta_prompt_graph(input_state, recursion_limit=recursion_limit)
    
    # Validate the output state
    system_message = ''
    output = ''

    if 'best_system_message' in output_state and output_state['best_system_message'] is not None:
        system_message = output_state['best_system_message']
    else:
        system_message = "Error: The output state does not contain a valid 'best_system_message'"

    if 'best_output' in output_state and output_state['best_output'] is not None:
        output = output_state["best_output"]
    else:
        output = "Error: The output state does not contain a valid 'best_output'"

    return system_message, output

# Create the Gradio interface
iface = gr.Interface(
    fn=process_message,
    inputs=[
        gr.Textbox(label="User Message"),
        gr.Textbox(label="Expected Output"),
        gr.Textbox(label="Acceptance Criteria"),
        gr.Number(label="Recursion Limit", value=25, precision=0, minimum=1, maximum=100, step=1)
    ],
    outputs=[gr.Textbox(label="System Message"), gr.Textbox(label="Output")],
    title="MetaPromptGraph Chat Interface",
    description="A chat interface for MetaPromptGraph to process user inputs and generate system messages.",
    examples="demo/examples"
)

# Launch the Gradio app
iface.launch()
