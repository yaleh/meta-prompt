# Gradio Framework Guideline

## Best Practices

When using Gradio, consider the following best practices:

- Keep your interface simple and intuitive for users to understand and use.
- Provide clear instructions and examples to guide users on how to interact with your interface.
- Validate and sanitize user inputs to prevent unexpected behavior or security vulnerabilities.
- Handle errors gracefully and provide informative error messages to users.
- Optimize the performance of your interface by minimizing the number of components and using caching when appropriate.
- Test your interface thoroughly with various inputs and edge cases to ensure its robustness.
- Use `gr.File` for file uploads and downloads. The file to download should be prepared before being passed into `gr.File`.
  - Example:
    ```python
    json_file = gr.File(
        label="Import/Export JSON", file_types=[".json"], type="filepath"
    )
    ```
- `gr.Dataframe` doesn't support deleting rows. 
- Set `col_count` to `(x, 'fixed')` to make the column width fixed.
  - Example:
    ```python
    input_df = gr.DataFrame(
        label="Input Examples",
        headers=["Input", "Output"],
        datatype=["str", "str"],
        row_count=(1, "dynamic"),
        col_count=(2, "fixed"),
    )
    ```
- Use `gr.Accordion` to hide advanced or optional settings that are not required for basic usage.
  - Example:
    ```python
    with gr.Accordion("Advanced Settings", open=False):
        gr.Slider(0, 100, 50, label="Slider")
    ```
- Use `gr.Tabs` and `gr.Tab` to organize related components into separate views.
  - Example:
    ```python
    with gr.Tabs() as llm_tabs:
        with gr.Tab('Simple') as simple_llm_tab:
            ...
        with gr.Tab('Advanced') as advanced_llm_tab:
            ...
    ```
- Use `gr.Examples` to provide sample inputs for users to test the interface.
  - Example:
    ```python
    gr.Examples(
        examples=[
            ["Hello, world!", "Hello, world!"],
            ["Hello, world!", "Hello, world!"],
        ],
        inputs=["text", "text"],
    )
- Use `gr.Markdown` to display formatted text, including links and images.
- Use `gr.ClearButton` to allow users to clear specific components or the entire interface.
- Use the `@gr.render` decorator to create dynamic apps with a variable number of components and event listeners. 
  - Example:
    ```python
    import gradio as gr

    with gr.Blocks() as demo:
        input_text = gr.Textbox(label="input")

        @gr.render(inputs=input_text)
        def show_split(text):
            if len(text) == 0:
                gr.Markdown("## No Input Provided")
            else:
                for letter in text:
                    gr.Textbox(letter)

    demo.launch()
    ```
- When using `@gr.render`, set the `key` argument for each component to preserve its value between re-renders.
  - Example:
    ```python
    import gradio as gr

    with gr.Blocks() as demo:
        text_count = gr.State(1)
        add_btn = gr.Button("Add Box")
        add_btn.click(lambda x: x + 1, text_count, text_count)

        @gr.render(inputs=text_count)
        def render_count(count):
            boxes = []
            for i in range(count):
                box = gr.Textbox(key=i, label=f"Box {i}")
                boxes.append(box)

            def merge(*args):
                return " ".join(args)

            merge_btn.click(merge, boxes, output)

    merge_btn = gr.Button("Merge")
    output = gr.Textbox(label="Merged Output")

    demo.launch()
    ```
- Update component attributes with `gr.update()` in the return statement of the event callback function.
  - Example:
    ```python
    def update_output(text):
        return gr.update(value=text)
    ```
- Use `gr.State` to maintain the state of your interface between re-renders, and to synchronize the updating of multiple components.
  - Example:
    ```python
    import gradio as gr
    
    def on_state_change(state: gr.State):
        return gr.update(visible=state.value), gr.update(visible=state.value)

    with gr.Blocks() as demo:
        text1 = gr.Textbox(label="Text 1")
        text2 = gr.Textbox(label="Text 2")
        state = gr.State(value=True)
        state.change(on_state_change, state, [text1, text2])

    demo.launch()
    ```
## Common pitfalls to avoid

- Overloading the interface with too many components or complex layouts, leading to a confusing user experience.
- Failing to handle edge cases or unexpected inputs, resulting in errors or incorrect outputs.
- Neglecting to secure your interface against potential security risks, such as unauthorized access or malicious inputs.
- Forgetting to optimize the performance of your interface, leading to slow response times or resource exhaustion.
- When using `@gr.render`, not setting the `key` argument for components, causing their values to be lost between re-renders.

