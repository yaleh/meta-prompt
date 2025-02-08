import streamlit as st

# Initialize the session state for the click counter
if 'click_count' not in st.session_state:
    st.session_state.click_count = 0

# Define a function to increment the click counter
def increment_counter():
    st.session_state.click_count += 1

# Display the text area with the current click count
st.text_area(
    "Click Count",
    value=f"The button has been clicked {st.session_state.click_count} times.",
    height=100
)

# Button to increment the click counter using the callback function
st.button("Click me!", on_click=increment_counter)
