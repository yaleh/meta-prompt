import gradio as gr

def sum_selected_row(selected_data):
    if selected_data is not None:
        row_index = selected_data[0]['row']
        row_data = selected_data[0]['data']
        row_sum = sum(row_data)
        return row_sum
    return ""

data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

demo = gr.Interface(
    fn=sum_selected_row,
    inputs=gr.Dataframe(headers=["Col 1", "Col 2", "Col 3"], datatype=["number", "number", "number"], row_count=3),
    outputs=gr.Textbox(label="Row Sum"),
    examples=[gr.Examples([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], label="Sample Data")],
)

demo.launch()