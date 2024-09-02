import streamlit as st

pg = st.navigation([
    st.Page("app/streamlit_sample_generator.py", title="Sample Generator", icon=":material/text_snippet:"),
    st.Page("app/streamlit_meta_prompt.py", title="Meta Prompt", icon=":material/auto_fix_high:"),
])

pg.run()