### Import packages for dataset visualization. Lets use ploty to create dynamy graph.
import plotly as py
from plotly import graph_objs as go
from plotly.offline import plot, iplot, init_notebook_mode

def graph_predictions(df, TARGET):

    ### Graph the daily predictions of the bitcoin closing price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Timestamp, y=df[TARGET], mode="lines", name=TARGET))
    fig.add_trace(go.Scatter(x=df.Timestamp, y=df["predictions"], mode="lines", name="predictions"))

    # Edit the layout
    fig.update_layout(title="Bitcoin Price PredictionSS",
                       xaxis_title='Day',
                       yaxis_title='Price Close',
    )
    return fig
