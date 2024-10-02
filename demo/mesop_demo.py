import mesop as me
import inspect

def embedded_show_dynamic_components(text, display_mode):
    if display_mode == "textbox":
        return me.textarea(value=text, label="Output")
    elif display_mode == "button":
        return me.button(text)
    else:
        return me.text("Invalid display mode")

@me.stateclass
class State:
    input_text: str = ""
    display_mode: str = "textbox"
    output: str = ""
    current_input: str = ""

def on_input_change(e: me.InputBlurEvent):
    state = me.state(State)
    state.input_text = e.value
    state.current_input = e.value

def on_mode_change(e: me.RadioChangeEvent):
    me.state(State).display_mode = e.value

def on_merge_click(e: me.ClickEvent):
    state = me.state(State)
    state.output = state.current_input

def on_textbox_change(e: me.InputBlurEvent, index: int):
    state = me.state(State)
    current_input_list = list(state.current_input)
    if index < len(current_input_list):
        current_input_list[index] = e.value
    state.current_input = ''.join(current_input_list)

@me.page(path="/")
def demo():
    state = me.state(State)
    
    me.input(label="Enter some text", value=state.input_text, on_blur=on_input_change)
    me.radio(
        options=[
            me.RadioOption(label="Textbox", value="textbox"),
            me.RadioOption(label="Button", value="button")
        ],
        value=state.display_mode,
        on_change=on_mode_change
    )
    
    # Dynamic component rendering
    embedded_show_dynamic_components(state.input_text, state.display_mode)
    
    # Dynamic Textboxes section
    me.text("Dynamic Textboxes", type="headline-5")
    for i, char in enumerate(state.current_input):
        me.textarea(
            label=f"Char {i+1}",
            value=char,
            on_blur=lambda e, index=i: on_textbox_change(e, index)
        )
    
    me.button("Merge Textboxes", on_click=on_merge_click)
    
    if state.output:
        me.text("Merged Output:", type="headline-6")
        me.text(state.output)
    
    # Display source code
    me.text("Source Code:", type="headline-5")
    code = inspect.getsource(embedded_show_dynamic_components)
    me.code(code, language="python")
