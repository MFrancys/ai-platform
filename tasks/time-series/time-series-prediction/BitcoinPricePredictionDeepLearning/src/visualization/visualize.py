import plotly.graph_objects as go

def graph_predictions(df, TARGET, predictions):
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[TARGET],
                        mode='lines',
                        name='TARGET'))
    fig.add_trace(go.Scatter(x=df.index, y=df[predictions],
                        mode='lines',
                        name=predictions))
    return fig
