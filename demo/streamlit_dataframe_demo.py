import streamlit as st
import pandas as pd

# Initialize the dataframe
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
}

# Initialize session state to store the dataframe
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(data)

# Define the callback function
def on_select():
    with st.sidebar:
        st.write("Selected rows:")
        st.write(st.session_state.selected_rows)

# Create the interactive dataframe
st.dataframe(st.session_state.df, 
               on_select=on_select,
               selection_mode="single-row",
               key='selected_rows')
