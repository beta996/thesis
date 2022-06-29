import dash_bootstrap_components as dbc
import nltk
from dash import Input, Output, dcc, html

import archive
import feature_extraction
import feature_selection
import load_data
import preprocess
import visualize
import classifier
from app import app, current_job, render_alert

nltk.download('punkt')

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    # "background-color": "#fl8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa"
}

sidebar = html.Div(
    [
        html.H4(children=["Text Sentiment     ", html.I(className="bi bi-spellcheck")]),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Load Data", href="/load-data", active="exact"),
                dbc.NavLink("Preprocess", href="/preprocess", active="exact"),
                dbc.NavLink("Visualize", href="/visualize", active="exact"),
                dbc.NavLink("Feature extraction", href="/feature-extraction", active="exact"),
                dbc.NavLink("Feature selection", href="/feature-selection", active="exact"),
                dbc.NavLink("Classifier", href="/classifier", active="exact"),
                dbc.NavLink("Job History", href="/archive", active="exact")

            ],
            vertical=True,
            pills=True
        ),

    ],
    style=SIDEBAR_STYLE,
)

print(current_job)

alert_bar = html.Div(render_alert("First you need to load the data! Go to", 'Data Load', "danger", '/load-data'),
                     style={"margin-left": "18rem",
                            "margin-right": "2rem"}, id='alert-bar')


@app.callback(Output("alert-bar", "children"), [Input("comb-datasets-button", "n_clicks")])
def update_alert(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        return render_alert("Time to preprocess the data! Go to", 'Preprocessing', "danger", '/preprocess')

    else:
        return render_alert("First you need to load the data! Go to", 'Data Load', "danger", '/load-data')


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, alert_bar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("home page!")
    elif pathname == "/load-data":
        return load_data.page_content()
    elif pathname == "/preprocess":
        return preprocess.page_content()
    elif pathname == "/visualize":
        return visualize.page_content()
    elif pathname == "/feature-extraction":
        return feature_extraction.page_content()
    elif pathname == "/feature-selection":
        return feature_selection.page_content()
    elif pathname == "/classifier":
        return classifier.page_content()
    elif pathname == "/archive":
        return archive.page_content()

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(port=8888)
