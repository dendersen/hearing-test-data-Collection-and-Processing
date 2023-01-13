from dash import Dash, dcc, html, Input, Output
import pandas as pd
import json

import plotly.express as px

def plotter():
  app = Dash(__name__)
  
  df = pd.read_csv('Data\editResultStorage.csv')
  
  available_plots = df['ID'].unique()
  available_ears = df['Out'].unique()
  
  app.layout = html.Div([
    html.P("Change figure width:"),
    dcc.Slider(id='slider', min=200, max=1900, step=25, value=1900,
              marks={x: str(x) for x in [100,300,500,700,900,1100,1300,1500,1700,1900]}),
    
    dcc.Graph(id='clientside-graph-px'),
    dcc.Store(
      id='clientside-figure-store-px'
    ),
    
    'Plot-version',
    dcc.Dropdown(available_plots, 'Niels', id='clientside-graph-Id-px'),
    
    'Øre',
    dcc.Dropdown(available_ears, 'Both', id='clientside-graph-indicator'),
    
    html.Details([
      html.Summary('Contents of figure storage'),
      dcc.Markdown(
        id='clientside-figure-json-px'
      )
    ])
  ])
  
  @app.callback(
    Output('clientside-figure-store-px', 'data'),
    Input('clientside-graph-Id-px', 'value'),
    Input('slider', 'value'),
    Input('clientside-graph-indicator', 'value')
  )
  
  def update_store_data(Id,width,ear):
    dff = df[df['ID'] == Id]  
    dfff = dff[dff['Response'] == ear]
    dfff["Frekvens"] = dfff["Frekvens"].astype(str)
    fig = px.scatter(dfff, x='Frekvens', y="AnswerTime",color="Response",height=680)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",)
    fig.update_layout(width=int(width))
    fig.update_traces(marker=dict(size=18,
                      line=dict(width=2,
                        color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    return fig
  
  app.clientside_callback(
    """
    function(figure, slider) {
        const fig = Object.assign({}, figure, {
            'layout': {
                ...figure.layout,
                'yaxis': {
                    ...figure.layout.yaxis
                }
              }
        });
        return fig;
    }
    """,
    Output('clientside-graph-px', 'figure'),
    Input('clientside-figure-store-px', 'data'),
  )
  
  @app.callback(
    Output('clientside-figure-json-px', 'children'),
    Input('clientside-figure-store-px', 'data'),)
  
  def generated_px_figure_json(data):
    return '```\n'+json.dumps(data, indent=2)+'\n```'
  
  app.run_server(debug=True)

plotter()