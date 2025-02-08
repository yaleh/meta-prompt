import gradio as gr
import pandas as pd

def on_select(evt: gr.SelectData):
    row_index = evt.index[0]
    row_data = df.iloc[row_index]
    row_sum = row_data.sum()
    return row_sum

with gr.Blocks() as demo:
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
        "col3": [7, 8, 9]
    })
    dataframe = gr.Dataframe(value=df, interactive=True)
    sum_textbox = gr.Textbox(label="Sum of selected row")

    dataframe.select(on_select, None, sum_textbox)

if __name__ == "__main__":
    demo.launch()
