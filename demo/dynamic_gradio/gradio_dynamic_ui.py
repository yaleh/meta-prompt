import gradio as gr
import traceback
import sys
from io import StringIO

# Removed: output_blocks = None

# Removed: execute_code function

def auto_update_code(code, auto_update):
    if auto_update:
        # Simplified: only return the code
        return code
    # change nothing
    return gr.update()

def manual_update_code(code):
    # Simplified: only return the code
    return code

with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Python Code Execution")
    
    with gr.Row():
        with gr.Group():
            code_input = gr.Code(label="Enter your Python code", language="python")
            auto_update = gr.Checkbox(label="Auto Update", value=True)
            update_button = gr.Button("Update")
            code_to_run = gr.Code(label="Code to run", language="python", interactive=False)

    # Set up event handlers
    code_input.change(auto_update_code, [code_input, auto_update], [code_to_run])
    auto_update.change(auto_update_code, [code_input, auto_update], [code_to_run])
    update_button.click(manual_update_code, [code_input], [code_to_run])

    @gr.render(inputs=[code_to_run])
    def execute_and_render(code):
        if not code:
            return gr.Markdown("## No input provided")
        
        try:
            # Capture stdout
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()

            local_scope = {}
            exec(code, globals(), local_scope)

            output = redirected_output.getvalue()
            error = ""
        except Exception as e:
            output = redirected_output.getvalue()
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
            sys.stdout = old_stdout

        if output:
            gr.Textbox(value=output, label="Output", lines=5, interactive=False)
        if error:
            gr.Textbox(value=error, label="Errors", lines=10, interactive=False)

if __name__ == "__main__":
    demo.launch()