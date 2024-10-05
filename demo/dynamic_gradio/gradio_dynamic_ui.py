import gradio as gr
import traceback
import sys
from io import StringIO
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import os
from gradio.components import ChatMessage

def auto_update_code(code, auto_update):
    if auto_update:
        # Simplified: only return the code
        return code
    # change nothing
    return gr.update()

def manual_update_code(code):
    # Simplified: only return the code
    return code

def predict(message, history, system_message):
    history_langchain_format = []
    
    # Add system message at the beginning if it's not empty
    if system_message.strip():
        history_langchain_format.append(AIMessage(content=system_message))
    
    for chat_message in history:
        if chat_message["role"] == "user":
            history_langchain_format.append(HumanMessage(content=chat_message["content"]))
        elif chat_message["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=chat_message["content"]))
    
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm.invoke(history_langchain_format)
    
    return ChatMessage(role="assistant", content=gpt_response.content)

def update_code_input(chat_history, last_ai_message):
    if not chat_history:
        return gr.update(), last_ai_message

    last_message = chat_history[-1]
    if last_message['role'] == 'assistant' and last_message['content'] != last_ai_message:
        # Check if the message contains a code block
        if '```' in last_message['content']:
            code_blocks = last_message['content'].split('```')
            for i in range(1, len(code_blocks), 2):
                code = code_blocks[i]
                # Check if it's a Python code block or if it doesn't have a language specified
                first_line = code.split('\n')[0].strip()
                if first_line == 'python' or not first_line:
                    # Extract the Python code
                    if first_line == 'python':
                        code = '\n'.join(code.split('\n')[1:])
                    else:
                        code = code.strip()
                    return code, last_message['content']

    return gr.update(), last_ai_message

llm = ChatOpenAI(temperature=1.0, model="github/gpt-4o-mini")

with gr.Blocks() as demo:

    gr.Markdown("## ChatGPT Interface with Custom Chatbot")

    last_ai_message = gr.State("")

    # Add system message textbox
    system_message = gr.Textbox(
        label="System Message",
        placeholder="Enter a system message to set the context for the AI assistant...",
        lines=2
    )

    # Create a custom Chatbot
    custom_chatbot = gr.Chatbot(
        label="AI Assistant",
        bubble_full_width=False,
        type="messages",
        show_copy_button=True,
        layout="bubble"
    )
        
    chat_interface = gr.ChatInterface(
        predict,
        chatbot=custom_chatbot,
        additional_inputs=[system_message],  # Add system_message as an additional input
        examples=[["Tell me a joke",""], ["Explain quantum computing",""], ["What's the weather like?",""]],
        title="AI Assistant",
        description="Ask me anything!",
        theme="soft",
        type="messages",
        retry_btn="Retry ↺",
        undo_btn="Undo ↩",
        clear_btn="Clear 🗑"
    )

    gr.Markdown("# Dynamic Python Code Execution")
    
    with gr.Group():
        code_input = gr.Code(label="Enter your Python code", language="python")
        with gr.Row():
            auto_update = gr.Checkbox(label="Auto Update", value=False)
            update_button = gr.Button("Update")

    with gr.Accordion(open=False):
        code_to_run = gr.Code(label="Code to run", language="python", interactive=False)
        # Add this inside the gr.Blocks() context, before the code_input
        timeout_input = gr.Number(label="Execution Timeout (seconds)", value=10, minimum=10, maximum=180, step=5)

    # Set up event handlers
    code_input.change(auto_update_code, [code_input, auto_update], [code_to_run])
    auto_update.change(auto_update_code, [code_input, auto_update], [code_to_run])
    update_button.click(manual_update_code, [code_input], [code_to_run])

    @gr.render(inputs=[code_to_run, timeout_input])
    def execute_and_render(code, timeout):
        if not code:
            return gr.Markdown("## No input provided")
        
        # Create a StringIO object to capture stdout
        stdout_capture = StringIO()
        
        try:
            local_scope = {}
            
            # Redirect stdout to our StringIO object
            original_stdout = sys.stdout
            sys.stdout = stdout_capture
            
            exec(code, globals(), local_scope)
            
            error = ""
        except Exception as e:
            tb = traceback.format_exc()
            error = f"""
## Error Details
```
<traceback>
{tb}
</traceback>
<error>
{str(e)}
</error>
<code>
{code}
</code>
```
"""
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout
        
        # Get the captured output
        output = stdout_capture.getvalue()
        
        # Display the output
        gr.Textbox(value=output, label="Output", lines=10, interactive=False)
        
        if error:
            gr.Textbox(value=error, label="Errors", lines=10, interactive=False)

    # Add the new event handler for custom_chatbot
    custom_chatbot.change(
        update_code_input,
        [custom_chatbot, last_ai_message],
        [code_input, last_ai_message]
    )

if __name__ == "__main__":
    demo.launch()