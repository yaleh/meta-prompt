import solara
import pandas as pd
import plotly

df = plotly.data.iris()

@solara.component
def Page():
    solara.DataFrame(df, items_per_page=5)

# Run the Solara app
if __name__ == "__main__":
    Page()