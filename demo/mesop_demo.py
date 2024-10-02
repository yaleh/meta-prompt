import mesop as me
import inspect

def embedded_show_dynamic_components(input_text: str, display_mode: str):
    if display_mode == "textbox":
        me.text(input_text)
    elif display_mode == "button":
        me.button(input_text)

# Initialize code_str with the source of embedded_show_dynamic_components
initial_code = inspect.getsource(embedded_show_dynamic_components)

@me.stateclass
class State:
    input_text: str = ""
    display_mode: str = "textbox"
    output: str = ""
    current_input: str = ""
    code_str: str = ""
    apply_dynamic_code: bool = True  # New field
    dynamic_code_success: bool = True  # New field

def on_input_change(e: me.InputBlurEvent):
    state = me.state(State)
    state.input_text = e.value
    state.current_input = e.value

def on_mode_change(e: me.RadioChangeEvent):
    state = me.state(State)
    state.display_mode = e.value

def on_merge_click(e: me.ClickEvent):
    state = me.state(State)
    state.output = state.current_input

def on_textbox_change(e: me.InputBlurEvent, index: int):
    state = me.state(State)
    current_input_list = list(state.current_input)
    if index < len(current_input_list):
        current_input_list[index] = e.value
    state.current_input = ''.join(current_input_list)

def on_code_change(e: me.InputBlurEvent):
    state = me.state(State)
    state.code_str = e.value

def on_apply_dynamic_code_change(e: me.CheckboxChangeEvent):
    state = me.state(State)
    state.apply_dynamic_code = e.checked

@me.page(path="/")
def demo():
    state = me.state(State)
    
    # Initialize code_str if it's empty
    if not state.code_str:
        state.code_str = initial_code

    # Display source code
    me.text("Source Code:", type="headline-5")
    me.textarea(
        value=state.code_str, 
        on_blur=on_code_change, 
        autosize=False, 
        min_rows=8,
        style=me.Style(width="100%")
    )
    # me.code(state.code_str, language="python")
    
    me.input(label="Enter some text", value=state.input_text, on_blur=on_input_change)
    me.radio(
        options=[
            me.RadioOption(label="Textbox", value="textbox"),
            me.RadioOption(label="Button", value="button")
        ],
        value=state.display_mode,
        on_change=on_mode_change
    )
    
    # Add checkbox for applying dynamic code
    me.checkbox(
        "Apply Dynamic Code",
        checked=state.apply_dynamic_code,
        on_change=on_apply_dynamic_code_change
    )
    
    # Dynamic component rendering
    with me.box():
        me.text("Dynamic Component Rendering", type="headline-5")
        if state.apply_dynamic_code:
            local_scope = {}
            try:
                exec(state.code_str, globals(), local_scope)
                f = local_scope.get("embedded_show_dynamic_components")
                if f:
                    f(state.input_text, state.display_mode)
                    state.dynamic_code_success = True
                else:
                    me.text("Error: Function 'embedded_show_dynamic_components' not found in the code.")
                    state.dynamic_code_success = False
            except Exception as e:
                me.text(f"Error executing code: {str(e)}")
                state.dynamic_code_success = False
        else:
            me.text("Dynamic code execution is disabled.")
            state.dynamic_code_success = True
    
    # Add readonly checkbox to show dynamic_code_success
    me.checkbox(
        "Dynamic Code Execution Success",
        checked=state.dynamic_code_success,
        disabled=True
    )
    
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
    
