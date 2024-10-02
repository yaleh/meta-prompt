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
    textbox1: str = ""
    textbox2: str = ""
    textbox3: str = ""

def on_input_change(e: me.InputBlurEvent):
    me.state(State).input_text = e.value

def on_mode_change(e: me.RadioChangeEvent):
    me.state(State).display_mode = e.value

def on_merge_click(e: me.ClickEvent):
    state = me.state(State)
    state.output = state.textbox1 + state.textbox2 + state.textbox3

def on_textbox_change(e: me.InputBlurEvent, box_number: int):
    state = me.state(State)
    if box_number == 1:
        state.textbox1 = e.value
    elif box_number == 2:
        state.textbox2 = e.value
    elif box_number == 3:
        state.textbox3 = e.value

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
    
    # Merge Textboxes section
    me.text("Merge Textboxes", type="headline-5")
    me.textarea(label="Textbox 1", value=state.textbox1, on_blur=lambda e: on_textbox_change(e, 1))
    me.textarea(label="Textbox 2", value=state.textbox2, on_blur=lambda e: on_textbox_change(e, 2))
    me.textarea(label="Textbox 3", value=state.textbox3, on_blur=lambda e: on_textbox_change(e, 3))
    me.button("Merge Textboxes", on_click=on_merge_click)
    
    if state.output:
        me.text("Merged Output:", type="headline-6")
        me.text(state.output)
    
    # Display source code
    me.text("Source Code:", type="headline-5")
    code = inspect.getsource(embedded_show_dynamic_components)
    me.code(code, language="python")
