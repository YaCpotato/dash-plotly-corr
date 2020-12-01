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
	id = "heatmap",
	figure = go.Figure( data = [
		go.Heatmap(
			x=boston_df.columns,
			y=boston_df.columns,
			z=corr,
			colorscale='RdBu',
			reversescale=True
		)
	])
)
])
app.run_server(debug=True)