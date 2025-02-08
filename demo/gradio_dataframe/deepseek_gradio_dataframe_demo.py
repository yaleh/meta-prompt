import gradio as gr
import pandas as pd

# Sample data for the DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data)

def on_select(evt: gr.SelectData):
    """Function to handle the select event of the DataFrame."""
    row_index = evt.index[0]  # Get the row index of the selected cell
    row_sum = df.iloc[row_index].sum()  # Sum all cells in the selected row
    return f"Sum of row {row_index}: {row_sum}"

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Gradio DataFrame Select Event Demo")
    dataframe = gr.DataFrame(value=df)
    output_textbox = gr.Textbox(label="Sum of Selected Row")
    
    # Bind the select event to the on_select function
    dataframe.select(on_select, None, output_textbox)

# Launch the demo
demo.launch()