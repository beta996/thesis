import dash_bootstrap_components as dbc
import nltk
import pandas as pd
from dash import html, Input, Output, State, dash_table
from nltk.stem import PorterStemmer
import app
import main


def page_content():
    return html.Div([
        html.H1('Processing'),
        html.P('Preprocessing is usually the first step in the pipeline of a '
               'Natural Language Processing (NLP) system. In particular it '
               'assumes usage of following techniques: tokenizing, '
               'lemmatizing, lowercasing etc.'),
        html.Div(dbc.Checklist(
            options=[
                {"label": 'Lowercasing', "value": "Lowercasing", 'label_id': 'lower'},
                {"label": 'symbols removal', "value": "symbols removal", 'label_id': 'symbol'},
                {"label": 'numerical values removal', "value": 'numerical values removal', 'label_id': 'numerical'},
                {"label": 'stopwords removal', "value": 'stopwords removal', 'label_id': 'stopwords'},
                {"label": 'stemming', "value": "stemming", 'label_id': 'stem'},
                {"label": 'users removal', "value": 'users removal', 'label_id': 'users'},
                {"label": 'hashtags removal', "value": 'hashtags removal', 'label_id': 'hashtags'},
                {"label": 'hyperlinks removal', "value": 'hyperlinks removal', 'label_id': 'hyper'}
            ], value=[], id='checklist'), style={}),
        dbc.Button("Submit", color="primary", id="submit-btn", n_clicks=0),
        html.Div(id='compare-data'),
        dbc.Tooltip(
            "Lowercasing: "
            "Converting all your data to lowercase helps in the process of preprocessing and in "
            "later stages in the NLP application.",
            target="lower",
        ),
        dbc.Tooltip(
            "Symbol removal: "
            "This operation removes characters such as: ~, :, ', '+', [, \\, @, ^ etc.",
            target="symbol",
        ),
        dbc.Tooltip(
            "Numerical values removal: "
            "This operation removes all the numerical values from the text, they usually do not "
            "impact the sentiment of the given textual content.",
            target="numerical",
        ),
        dbc.Tooltip(
            "Stopwords removal: "
            "Stop word removal is one of the most commonly used preprocessing steps across different "
            "NLP applications. Typically, articles and pronouns are generally classified as stop "
            "words.",
            target="stopwords",
        ),
        dbc.Tooltip(
            "Stemming: Stemming algorithms work by cutting off the end or the beginning of the word, "
            "taking into account a list of common prefixes and suffixes that can be found in an "
            "inflected word. "
            "This operation removes characters such as: ~, :, ', '+', [, \\, @, ^ etc.",
            target="stem",
        ),
        dbc.Tooltip(
            "Users removal: "
            "This operation works well for social media texts, because its aim is to remove the "
            "occurances of all words starting with the character @.",
            target="users",
        ),
        dbc.Tooltip(
            "Hashtags removal: "
            "This operation removes words starting with a character #",
            target="hashtags",
        ),
        dbc.Tooltip(
            "Hyperlinks removal: "
            "This operation removes hyperlinks that are often part of the posts, but rarely give any significant "
            "value towards the sentiment of the particular text.",
            target="hyper",
        ),
    ])


@app.app.callback(Output('compare-data', 'children'),
                  Output("alert-bar", "children"),
                  Input('submit-btn', 'n_clicks'),
                  State('checklist', 'value'))
def submit_preprocessing(n_clicks, chck_values):
    if n_clicks >= 1:
        app.current_job.preprocessing_steps = chck_values
        return preprocess(chck_values), app.render_alert("First you need to do feature extraction! Go to ",
                                                          'Feature extraction', "danger", '/feature-extraction')


def preprocess(chck_values: []):
    app.df_preprocessed = app.df_full.copy()

    special_chars = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`',
                     '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/', '"']

    if 'Lowercasing' in chck_values:
        app.df_preprocessed['clean_text'] = app.df_preprocessed['clean_text'].str.lower()  # convert to lowercase
    if 'hyperlinks removal' in chck_values:
        app.df_preprocessed['clean_text'] = app.df_preprocessed['clean_text'].str.replace('http?:\/\/[^\s]+|www\.[^\s]',
                                                                                          '',
                                                                                          regex=True)  # remove hyperlinks
    if 'users removal' in chck_values:
        app.df_preprocessed['clean_text'] = app.df_preprocessed['clean_text'].str.replace('@[\\w]+', '',
                                                                                          regex=True)  # remove users
    if 'hashtags removal' in chck_values:
        app.df_preprocessed['clean_text'] = app.df_preprocessed['clean_text'].str.replace('#\\w+', '',
                                                                                          regex=True)  # remove hashtags
    if 'numerical values removal' in chck_values:
        app.df_preprocessed['clean_text'] = app.df_preprocessed['clean_text'].str.replace('\\d+', '',
                                                                                          regex=True)  # remove numbers
    if 'symbols removal' in chck_values:
        for char in special_chars:
            app.df_preprocessed['clean_text'] = app.df_preprocessed['clean_text'].str.replace(char,
                                                                                              '')  # special characters

    # remove stopwords
    # nltk.download()
    stopwords = nltk.corpus.stopwords.words('english')
    if 'stopwords removal' in chck_values:
        app.df_preprocessed['clean_text'] = app.df_preprocessed['clean_text'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    if 'stemming' in chck_values:
        stemming(app.df_preprocessed)
    return dbc.Container(
        [dash_table.DataTable(app.df_preprocessed[:4].to_dict('records'),
                              [{"name": i, "id": i} for i in app.df_preprocessed.columns], id='tbl',
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
                              )])

def stemming(df:pd.DataFrame):
    ps = PorterStemmer()
    app.df_preprocessed['clean_text'] = app.df_preprocessed['clean_text'].apply(
        lambda x: ' '.join([ps.stem(word) for word in x.split()]))