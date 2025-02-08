import mesop as me
import traceback

@me.stateclass
class State:
    input_text: str = ""
    auto_update: bool = True
    dynamic_code: str = ""
    dynamic_code_error: str = ""
    needs_execution: bool = False

def update_state(state: State):
    if state.input_text != state.dynamic_code:
        state.dynamic_code = state.input_text
        state.needs_execution = True
    else:
        # If input hasn't changed, don't update output or execute code
        return

def on_input_change(event: me.InputEvent):
    state = me.state(State)
    state.input_text = event.value
    if state.auto_update:
        update_state(state)

def on_checkbox_change(event: me.CheckboxChangeEvent):
    state = me.state(State)
    state.auto_update = event.checked
    if state.auto_update:
        update_state(state)

def on_button_click(event: me.ClickEvent):
    state = me.state(State)
    update_state(state)
    state.needs_execution = True

@me.page(path="/")
def app():
    state = me.state(State)
    
    with me.box(style=me.Style(
        display="flex",
        flex_direction="column",
        gap=24,
        padding=me.Padding.all(32),
        max_width=800,
        margin=me.Margin.symmetric(horizontal="auto"),
        background=me.theme_var("surface"),
        border_radius=8,
        box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)"
    )): 
        with me.box(style=me.Style(display="flex", flex_direction="column", gap=16)):
            me.textarea(
                label="Enter your text",
                on_blur=on_input_change,
                value=state.input_text,
                rows=5,
                style=me.Style(width="100%")
            )
            
            with me.box(style=me.Style(display="flex", justify_content="space-between", align_items="center")):
                me.checkbox(
                    label="Auto Update",
                    checked=state.auto_update,
                    on_change=on_checkbox_change
                )
                me.button(
                    label="Update",
                    on_click=on_button_click,
                    type="raised",
                    color="primary"
                )
        
        with me.box(style=me.Style(display="flex", flex_direction="column", gap=16)):
            me.textarea(
                label="Output",
                value=state.dynamic_code,
                rows=5,
                readonly=True,
                style=me.Style(width="100%")
            )

        with me.box(key="dynamic_code_box", style=me.Style(display="flex", flex_direction="column", gap=16)):
            me.text(f"Dynamic Code: {state.dynamic_code_error}", key="dynamic_code_text")
            # Execute dynamic code and show errors if any
            if state.needs_execution:
                try:
                    exec(state.dynamic_code)
                    state.dynamic_code_error = ""  # Clear any previous errors
                except Exception as e:
                    tb = traceback.format_exc()
                    state.dynamic_code_error = f"<traceback>\n{tb}\n</traceback>\n<error>\n{e}\n</error>\n<code>\n{state.dynamic_code}\n</code>"
                state.needs_execution = False

                me.textarea(
                    key="dynamic_code_error",
                    label="Dynamic Code Error",
                    value=state.dynamic_code_error,
                    readonly=True,
                    autosize=True,
                    min_rows=10 if state.dynamic_code_error else 1,
                    max_rows=20,
                    style=me.Style(width="100%")
                )

            me.text(f"Dynamic Code End: {state.dynamic_code_error}", key="dynamic_code_end")