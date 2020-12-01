import numpy as np
import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from sklearn.datasets import load_boston
import plotly.figure_factory as ff
import plotly.graph_objs as go

boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['MEDV'] = boston.target
corr = boston_df.corr()

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph( 
    figure = ff.create_annotated_heatmap(
        z = np.round(corr.values, decimals=2),
        x = boston_df.columns.values.tolist(),
        y = boston_df.columns.values.tolist(),
        colorscale='Magma',showscale=True).update_yaxes(autorange="reversed"
    ))
])
app.run_server(debug=True)