import mesop as me
import inspect
import traceback
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

    with me.box(style=me.Style(padding=me.Padding.all(16))):
        me.text("Mesop Dynamic Component Demo", type="headline-4")

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap="16px")):
            # Left column
            with me.box(style=me.Style(display="flex", flex_direction="column", gap="16px")):
                # Source code section
                with me.box():
                    me.text("Source Code:", type="headline-5")
                    me.textarea(
                        value=state.code_str, 
                        on_blur=on_code_change, 
                        autosize=True, 
                        min_rows=10,
                        style=me.Style(width="100%")
                    )

                # Input and controls section
                with me.box():
                    me.input(label="Enter some text", value=state.input_text, on_blur=on_input_change)
                    me.radio(
                        options=[
                            me.RadioOption(label="Textbox", value="textbox"),
                            me.RadioOption(label="Button", value="button")
                        ],
                        value=state.display_mode,
                        on_change=on_mode_change
                    )
                    me.checkbox(
                        "Apply Dynamic Code",
                        checked=state.apply_dynamic_code,
                        on_change=on_apply_dynamic_code_change
                    )

            # Right column
            with me.box(style=me.Style(display="flex", flex_direction="column", gap="16px")):
                # Dynamic component rendering section
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
                            tb = traceback.format_exc()
                            me.textarea(
                                value=f"<traceback>\n{tb}\n</traceback>\n<error>\n{e}\n</error>\n<code>\n{state.code_str}\n</code>",
                                readonly=True,
                                autosize=True,
                                min_rows=10,
                                max_rows=20,
                                style=me.Style(width="100%")
                            )
                            state.dynamic_code_success = False
                    else:
                        me.text("Dynamic code execution is disabled.")
                        state.dynamic_code_success = True

                    me.checkbox(
                        "Dynamic Code Execution Success",
                        checked=state.dynamic_code_success,
                        disabled=True
                    )

                # Dynamic Textboxes section
                with me.box():
                    me.text("Dynamic Textboxes", type="headline-5")
                    with me.box(style=me.Style(display="flex", flex_wrap="wrap", gap="8px")):
                        for i, char in enumerate(state.current_input):
                            me.textarea(
                                label=f"Char {i+1}",
                                value=char,
                                on_blur=lambda e, index=i: on_textbox_change(e, index),
                                style=me.Style(width="60px", height="60px")
                            )

                    me.button("Merge Textboxes", on_click=on_merge_click)

                    if state.output:
                        me.text("Merged Output:", type="headline-6")
                        me.text(state.output)

