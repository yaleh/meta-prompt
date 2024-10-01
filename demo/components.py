import gradio as gr

# This function is called by the gradio render function, within the scope of the gradio blocks.
def embedded_show_dynamic_components(text, display_mode):
    components = []
    for letter in text:
        if display_mode == "textbox":
            components.append(gr.Textbox(value=letter, label=f"Letter: {letter}"))
        else:
            components.append(gr.Button(letter))

    merge_btn.click(merge_text, inputs=components, outputs=output) # type: ignore