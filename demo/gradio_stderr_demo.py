import gradio as gr
import sys
import io
from contextlib import redirect_stderr

# Create a StringIO object to capture stderr
stderr_capture = io.StringIO()

# # Redirect stderr to our StringIO object
# sys.stderr = stderr_capture

def generate_error(const_param):
    # This function will generate an error message
    try:
        result = 10 / const_param
        return f"Result: {result}"
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        return "An error occurred. Check the error log."

def update_error_log():
    # Get the current content of our stderr capture
    return stderr_capture.getvalue()

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Gradio Demo with STDERR Monitoring")
    
    with gr.Row():
        trigger_button = gr.Button("Trigger Error")
        const_param = gr.Number(value=0, label="Constant Parameter")
    
    output = gr.Textbox(label="Output")
    error_log = gr.Textbox(label="Error Log", lines=5)
    
    # Set up the timer to update the error log every second
    timer = gr.Timer(1)
    timer.tick(update_error_log, outputs=[error_log])
    
    # Bind the button click event
    # trigger_button.click(generate_error, inputs=["const"], outputs=[output])

    @gr.render(inputs=[trigger_button])
    def generate_error(trigger_button):
        btn = gr.Button("Click me")
        try:
            btn.click(generate_error, inputs=["const"], outputs=[output])
        except AttributeError as e:
            # print error to stdout, and ignore it
            # print(f"An error occurred: {str(e)}")
            pass

# Launch the demo
if __name__ == "__main__":
    demo.launch()