import gradio as gr

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Enter some text")
    mode = gr.Radio(["textbox", "button"], value="textbox", label="Display mode")

    def merge_text(*args):
        s = "".join([arg for arg in args])
        return s
    
    output = gr.Markdown()

    merge_btn = gr.Button("Merge Textboxes")

    @gr.render(inputs=[input_text, mode])
    def show_dynamic_components(text, display_mode):
        if not text:
            return gr.Markdown("## No input provided")
        
        components = []
        for letter in text:
            if display_mode == "textbox":
                components.append(gr.Textbox(value=letter, label=f"Letter: {letter}"))
            else:
                components.append(gr.Button(letter))
        
        merge_btn.click(merge_text, inputs=components, outputs=output)

        return components



demo.launch()
