import streamlit as st
import pandas as pd

# Initialize session state to store the data
if 'data' not in st.session_state:
    # Example data, you can customize this as needed
    st.session_state.data = pd.DataFrame({
        'Column 1': ["1", "2", "3"],
        'Column 2': ['A', 'B', 'C']
    })

# Define a function to update the session state
def update_data_editor():
    st.session_state.data = data1

initial_data = st.session_state.data

# Create the first data editor
data1 = st.data_editor(
    st.session_state.data,
    key='editor1',
    num_rows="dynamic",
    column_config={
        "Column 1": st.column_config.TextColumn("Column 1", width="large"),
        "Column 2": st.column_config.TextColumn("Column 2", width="large"),
    },
    on_change=update_data_editor
)

# Create the second data editor, which will be synced with the first one
data2 = st.data_editor(
    st.session_state.data,
    key='editor2',
    num_rows="dynamic",
    # on_change=lambda: update_data_editor(st.session_state.editor2)
)

# # Display the data
# st.write("Data Editor 1", data1)
# st.write("Data Editor 2", data2)
