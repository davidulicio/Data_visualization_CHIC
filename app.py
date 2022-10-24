import dash
from dash import html, dcc

app = dash.Dash(__name__, use_pages=True)

# dash.register_page("OPEC",  path='/', layout=html.Div('Home Page'))
# dash.register_page("OPSR", layout=html.Div('Analytics'))

server = app.server

app.layout = html.Div(
    [
        # main app framework
        html.Div("Omora Data Visualization App", style={'fontSize': 50, 'textAlign': 'center'}),
        html.Div([
            dcc.Link(page['name']+"  |  ", href=page['path'])
            for page in dash.page_registry.values()
        ]),
        html.Hr(),

        # content of each page
        dash.page_container
    ]
)


if __name__ == "__main__":
    app.run(debug=False)
