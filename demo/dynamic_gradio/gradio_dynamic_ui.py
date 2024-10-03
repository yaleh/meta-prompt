import gradio as gr
import traceback
import sys
from io import StringIO

output_blocks = None

def execute_code(blocks, code):

    if not code.strip():
        return "", gr.Textbox(value="", label="Errors", lines=10)
    
    # Capture stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        with blocks:
            exec(code)
        output = redirected_output.getvalue()
        error = ""
    except Exception as e:
        output = redirected_output.getvalue()
        tb = traceback.format_exc()
        error_message = f"<traceback>\n{tb}\n</traceback>\n<error>\n{e}\n</error>\n<code>\n{code}\n</code>"
        error = gr.Textbox(value=error_message, label="Errors", lines=10)
    finally:
        sys.stdout = old_stdout

    return output, error

def auto_update_code(code, auto_update):
    if auto_update:
        result = execute_code(output_blocks, code)
        return code, result[0], result[1]
    # change nothing
    return gr.update(), gr.update(), gr.update()

def manual_update_code(code):
    result = execute_code(output_blocks, code)
    # prepend the code to the output triple
    return code, result[0], result[1]

with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Python Code Execution")
    
    with gr.Row():
        with gr.Column(scale=2):
            code_input = gr.Code(label="Enter your Python code", language="python")
            auto_update = gr.Checkbox(label="Auto Update", value=True)
            update_button = gr.Button("Update")
            code_to_run = gr.Code(label="Code to run", language="python", interactive=False)
        
        with gr.Column(scale=2):
            output_blocks = gr.Blocks()
            output = gr.Textbox(label="Output", lines=5)
            error = gr.Textbox(label="Errors", lines=10)

    output_blocks = gr.Blocks()

    # Set up event handlers
    code_input.change(auto_update_code, [code_input, auto_update], [code_to_run, output, error])
    auto_update.change(auto_update_code, [code_input, auto_update], [code_to_run, output, error])
    update_button.click(manual_update_code, [code_input], [code_to_run, output, error])

    @gr.render(inputs=[code_to_run])
    def execute_and_render(code):
        if not code:
            return gr.Markdown("## No input provided")
        
        try:
            local_scope = {}
            exec(code, globals(), local_scope)
        except Exception as e:
            tb = traceback.format_exc()
            error_message = f"""
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
            gr.Markdown(error_message)

if __name__ == "__main__":
    demo.launch()