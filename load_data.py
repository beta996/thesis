import base64
import io
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html, dcc, Output, Input, State, dash_table

import app


def page_content():
    return html.Div(
        [
            html.H1('Data Load'),
            html.P('The sentiment analysis that we will be conducting is a machine learning-based technique, '
                   'specifically in the area of supervised learning. This means that we need a training dataset, '
                   'so that the model can learn from it.'),
            html.P('In this section you can load the data from the premade datasets coming from the website '
                   'Kaggle.com. You can also load your own dataset if it satisfies the conditions.'),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.H5("Twitter US Airline Sentiment", style={"font-weight": "bold"}),
                            html.P("Analyze how travelers in February 2015 expressed their feelings on Twitter"),
                            html.Span("size = 14845", className="badge rounded-pill bg-warning"),
                            html.Br(),
                            dbc.Button("Add this dataset", style={"margin-top": "10px"}, id="airline btn")
                        ],
                        title="Twitter US Airline Sentiment",
                    ),
                    dbc.AccordionItem(
                        [
                            html.H5("Reddit Sentimental analysis Dataset", style={"font-weight": "bold"}),
                            html.P("This is was a Dataset Created as a part of the university Project On "
                                   "Sentimental Analysis On Multi-Source Social Media Platforms using PySpark."),
                            html.Span("size = 36801", className="badge rounded-pill bg-warning"),
                            html.Br(),
                            dbc.Button("Add this dataset", style={"margin-top": "10px"}, id='reddit btn')
                        ],
                        title="Reddit Sentimental analysis Dataset",
                    ),
                    dbc.AccordionItem(
                        [
                            html.H5("apple_twitter_sentiment_texts", style={"font-weight": "bold"}),
                            html.P(
                                "Dataset is from crowdflower - https://data.world/crowdflower/apple-twitter-sentiment"),
                            html.Span("size = 1624", className="badge rounded-pill bg-warning"),
                            html.Br(),
                            dbc.Button("Add this dataset", style={"margin-top": "10px"}, id='apple btn')
                        ],
                        title="apple_twitter_sentiment_texts",
                    ),
                ],
                start_collapsed=True),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Dataset added")),
                ],
                id="modal",
                is_open=False,
            ),
            dbc.Button("Display Dataset", color="primary", id="comb-datasets-button", n_clicks=0),
            html.Div(id='dataset'),
            html.Div(className='dashed', style={"border-bottom": "1px dashed #000",
                                                "text-align": "center", "height": "10px", "margin-bottom": "10px"
                                                }, children=[html.Span("OR", style={"padding": "0 5px"})]),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-data-upload'),
        ], className="d-grid gap-2")


@app.app.callback(Output("modal", "is_open"),
                  Output("apple btn", "disabled"),
                  Output("apple btn", 'children'),
                  Output("reddit btn", "disabled"),
                  Output("reddit btn", 'children'),
                  Output("airline btn", "disabled"),
                  Output("airline btn", 'children'),
                  [Input("apple btn", "n_clicks"), Input("reddit btn", "n_clicks"), Input("airline btn", "n_clicks")],
                  [State("modal", "is_open")])
def toggle_modal(n1, n2, n3, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if ctx.triggered[0]['value'] > 0:
        if ctx.triggered[0]['prop_id']:
            if button_id == 'apple btn':
                app.current_job.datasets.append("apple dataset")
                # app.current_job.__setattr__('dataset', 'apple dataset')
                app.df_full = pd.concat([app.df_full, app.df3], ignore_index=True)
                return not is_open, True, "Added", False, "Add", False, "Add"
            if button_id == 'reddit btn':
                app.current_job.datasets.append("reddit dataset")
                app.df_full = pd.concat([app.df_full, app.df2], ignore_index=True)
                return not is_open, False, "Add", True, "Added", False, "Add"
            if button_id == 'airline btn':
                app.current_job.datasets.append("airline dataset")
                app.df_full = pd.concat([app.df_full, app.df1], ignore_index=True)
                return not is_open, False, "Add", False, "Add", True, "Added"
    app.df_full.dropna(axis=0, inplace=True)
    app.df_full.sample(frac=1).reset_index(drop=True, inplace=True)
    return is_open, False, None, None, None, None, None


@app.app.callback(Output('output-data-upload', 'children'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            app.df_full = df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.app.callback(Output('dataset', 'children'),
                  Input('comb-datasets-button', 'n_clicks'))
def update_dataset_view(n_clicks):
    if n_clicks >= 1:
        return dbc.Container(
            [dash_table.DataTable(app.df_full[:4].to_dict('records'),
                                  [{"name": i, "id": i} for i in app.df_full.columns],
                                  id='tbl',
                                  style_data={
                                      'whiteSpace': 'normal',
                                      'height': 'auto',
                                  }, editable=True,
                                  tooltip_header={
                                      'clean_comment': 'Content of the post',
                                      'category': f'1 - positive \n 0 - neutral \n -1 - negative',
                                  },
                                  css=[{
                                      'selector': '.dash-table-tooltip',
                                      'rule': 'background-color: grey; font-family: monospace; color: white'
                                  }],
                                  style_header={
                                      'textDecoration': 'underline',
                                      'textDecorationStyle': 'dotted',
                                  }
                                  ),
             dbc.Alert(f"Shape: {app.df_full.shape}"),
             dcc.Slider(1, 20, 1, value=5, marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}, id='slider')])

    return html.Div()


@app.app.callback(Output('tbl', 'data'), Input('slider', 'value'))
def update_df_size(value):
    return app.df_full[:value].to_dict('records')
