import dash
import dash_table
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State

# Create a sample data frame
df = pd.DataFrame({
    'Column 1': ['Row 1', 'Row 2', 'Row 3'],
    'Column 2': [1, 2, 3],
    'Column 3': [4, 5, 6]
})

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    dash_table.DataTable(
        id='editable-data-table',
        columns=[{'name': i, 'id': i} for i in df.columns],
        data=df.to_dict('records'),
        editable=True,
        row_deletable=True,  # Add this line to enable row deletion
    ),
    html.Button('Add Row', id='add-row-button', n_clicks=0),  # Add this line to create an "Add Row" button
])

# Add this callback function to handle adding rows
@app.callback(
    Output('editable-data-table', 'data'),
    Input('add-row-button', 'n_clicks'),
    State('editable-data-table', 'data'),
    State('editable-data-table', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)