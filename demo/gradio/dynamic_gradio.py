import gradio as gr
import inspect
import traceback

from components import embedded_show_dynamic_components

# Remove this line as we'll use the Textbox content instead
# code = inspect.getsource(embedded_show_dynamic_components)

with gr.Blocks() as demo:
    # Add a new Textbox at the top to display and edit the code
    code_textbox = gr.Textbox(
        label="Code to execute",
        value=inspect.getsource(embedded_show_dynamic_components),
        lines=10
    )

    input_text = gr.Textbox(label="Enter some text")
    mode = gr.Radio(["textbox", "button"], value="textbox", label="Display mode")

    def merge_text(*args):
        s = "".join([arg for arg in args])
        return s
    
    output = gr.Markdown()

    merge_btn = gr.Button("Merge Textboxes")

    @gr.render(inputs=[input_text, mode, code_textbox])
    def show_dynamic_components(text, display_mode, code):
        if not text:
            return gr.Markdown("## No input provided")
        
        try:
            local_scope = {}
            exec(code, globals(), local_scope)
            f = local_scope["embedded_show_dynamic_components"]
            
            result = f(text, display_mode)
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
            return gr.Markdown(error_message)

        return result

if __name__ == "__main__":
    demo.launch()