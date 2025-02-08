import gradio as gr

component_list = []

def add_new_component(num_components):
    with demo:
        new_textbox = gr.Textbox(label=f"New Textbox {num_components + 1}")
        component_list.append(new_textbox)
    updated_num = num_components + 1
    
    return [
        updated_num,
        gr.Column(component_list)
    ]

with gr.Blocks() as demo:
    num_components = gr.Number(value=0, label="Number of components")
    add_button = gr.Button("Add new component")
    container = gr.Column()
    
    add_button.click(
        add_new_component,
        inputs=[num_components],
        outputs=[num_components, container],
    )

if __name__ == "__main__":
    demo.launch()