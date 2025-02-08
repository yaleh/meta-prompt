import gradio as gr
import pandas as pd

# Create a sample data frame
df = pd.DataFrame({
    'Column 1': [1, 2, 3],
    'Column 2': [4, 5, 6],
    'Column 3': [7, 8, 9]
})

# Define the callback function to handle the `select` event
def handle_select_event(selected_row):
    total = sum(selected_row.values())
    return str(total)

# Create a Gradio interface
interface = gr.Interface(
    fn=handle_select_event,
    inputs=[gr.inputs.Dataframe(df, label='Dataframe')],
    outputs='textbox',
    title='Gradio Dataframe Select Event Demo'
)

# Run the interface
interface.launch()