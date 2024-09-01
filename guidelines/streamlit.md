# Streamlit Development Guidelines

## Best Practices

- Use `st.session_state` to store and share data across the app, such as input data, output results, and UI state.
- Organize code into functions for better readability and maintainability. 
- Use `st.expander` to group related UI elements and allow users to collapse/expand sections.
- Provide options for users to import/export data, such as using `st.file_uploader` and `st.download_button`.
- Use `st.columns` to create responsive layouts and align UI elements.
- Provide clear labels and instructions for user inputs and interactions.
- Handle exceptions and display user-friendly error messages using `st.warning`.
- Use `st.spinner` to indicate when long-running operations are in progress.
- Allow users to customize and control the app's behavior through widgets like `st.slider`, `st.selectbox`, etc.
- Use `st.dataframe` to display interactive tables, with features like row selection.
- Implement callbacks using `on_click` or `on_change` to respond to user interactions.
- Use `st.sidebar` to display additional information or controls without cluttering the main UI.
- Organize the app's UI elements in a logical order, grouping related functionality together.

## Principles

- Prioritize usability and user experience in the app's design and layout.
- Ensure the app is responsive and works well on different screen sizes.
- Optimize performance by minimizing unnecessary computations and caching results when possible.
- Follow PEP 8 style guidelines for Python code.
- Document the app's purpose, usage instructions, and code to enhance maintainability.
- Test the app thoroughly to identify and fix bugs, edge cases, and performance issues.
- Consider accessibility and ensure the app can be used by people with different abilities.
- Provide clear feedback to users about the app's status and results.
- Allow users to customize the app's behavior through settings and options.
- Design the app to be modular and extensible, allowing for future enhancements and new features.