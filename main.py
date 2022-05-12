from collections import defaultdict


from nltk import ngrams
import numpy as np
import plotly.figure_factory as ff
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, dcc, html, dash_table, State
import dash_auth
import plotly.express as px
import matplotlib.pyplot as plt
from nltk import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

import base64
import datetime
import io

import nltk

nltk.download('punkt')

VALID_USERNAME_PASSWORD_PAIRS = {
    'hello': 'world'
}

# df1 = pd.read_csv('./datasets/Reddit_Data.csv')
# df2 = pd.read_csv('./datasets/Twitter_Data.csv')
# df_full = pd.concat([df2, df1])
# df_full.dropna(axis=0, inplace=True)
# df_full = pd.read_csv('./datasets/test.csv')
df_full = pd.DataFrame()
df_vect = pd.DataFrame()

df1 = pd.read_csv('./datasets/Tweets1.csv')
df2 = pd.read_csv('./datasets/Reddit_Data.csv')
df3 = pd.read_csv('./datasets/apple-twitter-sentiment-texts.csv')
df3 = df3[['clean_text', 'category']]
df1.dropna(axis=0, inplace=True)
df2.dropna(axis=0, inplace=True)
df3.dropna(axis=0, inplace=True)

app = dash.Dash(external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP])

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

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
                dbc.NavLink("Load Data", href="/page-1", active="exact"),
                dbc.NavLink("Preprocess", href="/page-2", active="exact"),
                dbc.NavLink("Visualize", href="/page-3", active="exact"),
                dbc.NavLink("Feature extraction", href="/page-4", active="exact"),
                dbc.NavLink("Feature selection", href="/page-5", active="exact")

            ],
            vertical=True,
            pills=True
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("home page!")
    elif pathname == "/page-1":
        return html.Div(
            [
                html.H1('Data Load'),
                html.P('The sentiment analysis that we will be conducting is a machine learning-based technique, '
                       'specifically in the area of supervised learning. This means that we need a training dataset, so that the model can learn from it.'),
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
                        # dbc.ModalBody(f"Total shape: {df_full.shape}")
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
    elif pathname == "/page-2":
        return html.Div([html.H1('Processing'), html.P('Preprocessing is usually the first step in the pipeline of a '
                                                       'Natural Language Processing (NLP) system. In particular it '
                                                       'assumes usage of following techniques: tokenizing, '
                                                       'lemmatizing, lowercasing etc.'), html.Div(dbc.Checklist(
            options=[
                {"label": 'Lowercasing', "value": "Lowercasing", 'label_id': 'lower'},
                {"label": 'symbols removal', "value": "symbols removal", 'label_id': 'symbol'},
                {"label": 'numerical values removal', "value": 'numerical values removal', 'label_id': 'numerical'},
                {"label": 'stopwords removal', "value": 'stopwords removal', 'label_id': 'stopwords'},
                {"label": 'stemming', "value": "stemming", 'label_id': 'stem'},
                {"label": 'users removal', "value": 'users removal', 'label_id': 'users'},
                {"label": 'hashtags removal', "value": 'hashtags removal', 'label_id': 'hashtags'},
                {"label": 'hyperlinks removal', "value": 'hyperlinks removal', 'label_id': 'hyper'}
            ], value=[],
            id='checklist'
        ), style={}),
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
                             "This operation removes characters such as: ~, :, ', '+', [, \, @, ^ etc.",
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
                             "This operation removes hyperlinks that are often part of the posts, but rarely give any significant value towards the sentiment of the particular text.",
                             target="hyper",
                         ),
                         ])

    # If the user tries to reach a different page, return a 404 message
    elif pathname == "/page-3":
        return html.Div([
            html.H2('Visualization'),
            dcc.Tabs(id="tabs-graph", value='tab-1-graph', children=[
                dcc.Tab(label='Pie chart', value='tab-1-graph'),
                dcc.Tab(label='Bar charts', value='tab-2-graph'),
                dcc.Tab(label='Histograms', value='tab-3-graph'),
                dcc.Tab(label='Tab Four', value='tab-4-graph')
            ]),
            html.Div(id='freq dropdown'),
            dcc.Graph(id='tabs-content-graph'),
            html.Div(id='wordcloud')
        ])
    elif pathname == "/page-4":
        return html.Div([html.H2('Feature extraction'),
                         html.P('The main problem in working with language processing is that machine learning '
                                'algorithms cannot work on the raw text directly. So, we need some feature extraction '
                                'techniques to convert text into a matrix(or vector) of features. Some of the most '
                                'popular methods of feature extraction are :'),
                         html.Ul([html.Li('Bag-of-words'), html.Li('TF-IDF')]),
                         dbc.Accordion(
                             [
                                 dbc.AccordionItem(
                                     [
                                         html.H5("Bag-of-words", style={"font-weight": "bold"}),
                                         html.P(
                                             "Bag-of-Words is one of the most fundamental methods to transform tokens "
                                             "into a set of features. The BoW model is used in document "
                                             "classification, where each word is used as a feature for training the "
                                             "classifier."),
                                         dbc.Button("Try it!", style={"margin-top": "10px"}, id='try bow')

                                     ],
                                     title="Bag-of-words",
                                 ),

                                 dbc.AccordionItem(
                                     [
                                         html.H5("TF-IDF", style={"font-weight": "bold"}),
                                         html.P("This is the most popular way to represent documents as feature "
                                                "vectors. TF-IDF stands for Term Frequency, Inverse Document "
                                                "Frequency. TF-IDF measures how important a particular word is with "
                                                "respect to a document and the entire corpus. "
                                                ),
                                         dbc.Button("Try it!", style={"margin-top": "10px"}, id='try tfidf')

                                     ],
                                     title="TF-IDF",
                                 ),
                             ]),
                         html.P("Our "),
                         html.H5("Choose feature extraction method: "),
                         dbc.Select(
                             id="select feature extraction",
                             options=[
                                 {"label": "Bag-of-Words", "value": "1"},
                                 {"label": "TF-IDF", "value": "2"}
                             ],
                             value="2"
                         ),
                         html.H4("First 4 records from your data:"),
                         html.Div(id='feature extraction df'),
                         html.P(),
                         dbc.Modal(
                             [
                                 dbc.ModalHeader(dbc.ModalTitle("BOW")),
                                 html.Div([html.P('Lets take 2 examplary posts:'),
                                           html.P(df_full['clean_text'][1], style={"font-weight": "bold"}),
                                           html.P(df_full['clean_text'][2], style={"font-weight": "bold"})
                                           ]),
                                 html.Div(id='explanation div bow'),
                                 dbc.Button("Next step", id='next step bow')
                             ],
                             id="modal bow",
                             is_open=False,
                         ),
                         dbc.Modal(
                             [
                                 dbc.ModalHeader(dbc.ModalTitle("TF-IDF")),
                                 html.Div([html.P('Lets assume we want to calculate the TF-IDF of the word "people", '
                                                  'which occurs 4 times in the post of 100 words. Furthermore, '
                                                  'we have in total 1000 posts and "people" is in 100 of them.'),
                                           ]),
                                 html.Div(id='explanation div tfidf'),
                                 dbc.Button("Next step", id='next step tfidf')
                             ],
                             id="modal tfidf",
                             is_open=False,
                         )
                         ])
    elif pathname == "/page-5":
        return html.Div([html.H1("Feature Selection"),
                         html.P(f"Congratulations! Your vectorized dataframe has {df_vect.shape[1]} columns!"
                                f" That's a lot! However, A large number of irrelevant features increases the training time exponentially and increase the risk of overfitting."),
                         html.P("Feature selection is s a process of extracting the most relevant features from the dataset."
                                " It can help with a problem of too many columns in the vectorized dataframe."),
                         html.Ul(["Statistical tests: ", html.Li("Chi square Test")]),
                         dcc.Slider(1, 20, 1, value=5, marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True}, id='slider feature selection'),
                         selectKBestChi(df_vect)
                         ])
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

def selectKBestChi(vect_df):
    select = SelectKBest(chi2, k=100)
    X_new = select.fit_transform(vect_df,df_full['category'])
    # df_selected = pd.DataFrame(X_new.loc[:,:].toarray())
    df_selected = pd.DataFrame(X_new)
    return dbc.Container(
            [dash_table.DataTable(df_selected[:5].to_dict('records'), [{"name": i, "id": j} for i,j in zip(select.get_feature_names_out(), df_selected.columns)], style_table={'overflowX': 'auto'}),
             html.P(f"Shape: {df_selected.shape}")])

@app.callback(Output('explanation div bow', 'children'),
              Input('next step bow', 'n_clicks'))
def next_step_submit_bow(n_clicks):
    first_text = df_full['clean_text'][1]
    second_text = df_full['clean_text'][2]
    # distinct_tokens = [word for word in (first_text + second_text).split()]
    l_doc1 = first_text.split()
    l_doc2 = second_text.split()
    distinct_tokens = np.union1d(l_doc1, l_doc2)
    bow1 = calculateBOW(distinct_tokens, l_doc1)
    bow2 = calculateBOW(distinct_tokens, l_doc2)
    df_bow = pd.DataFrame([bow1, bow2])
    if n_clicks==1:
        return html.Div([html.P('2. We consider all the unique words from the above set of reviews to create a '
                               'vocabulary, which is going to be as follows :'),
                         html.P(f'[{distinct_tokens}]')
                         ])
    elif n_clicks==2:
        return html.Div([html.P("3. In the third step, we create a matrix of features by assigning a separate column for "
                                "each word, while each row corresponds to a review. This process is known as Text "
                                "Vectorization. Each entry in the matrix signifies the presence(or absence) of the "
                                "word in the review. We put 1 if the word is present in the review, and 0 if it is "
                                "not present."),
                         dash_table.DataTable(
                             df_bow.to_dict('records'),
                             [{'name': i, 'id': i} for i in df_bow.columns],
                             style_table={'overflowX': 'auto'},
                         ),
                         ])


@app.callback(Output('explanation div tfidf', 'children'),
              Input('next step tfidf', 'n_clicks'))
def next_step_submit_tfidf(n_clicks):

    if n_clicks==1:
        return html.Div([html.P('TF-IDF is a combination of Term Frequency and Inverse Document Frequency.'),
                         html.P(f'First, let us calculate the Term Frequency: '),
                         html.P(f'TF(w) = (number of times word w appears in a document) / (total number of words in the document)'),
                         html.P(f'TF(people) = 4 / 100 = 0.04')

                         ])
    elif n_clicks==2:
        return html.Div([html.P("Now we measure Inverse Document Frequency, It measures how important a word is for the corpus."),
                         html.P("IDF(w) = log(total number of documents / number of documents with w in it)"),
                         html.P("IDF(people) = log(1000 / 100) = 1"),
                         ])
    elif n_clicks==3:
        return html.Div([
            html.P("Finally, to calculate TF-IDF, we multiply these two factors – TF and IDF."),
            html.P("TF-IDF(people) = TF(people) x IDF(people) = 0.04 * 1 = 0.04"),
        ])


def calculateBOW(wordset,l_doc):
  tf_diz = dict.fromkeys(wordset,0)
  for word in l_doc:
      tf_diz[word]=l_doc.count(word)
  return tf_diz


@app.callback(Output("modal bow", "is_open"),
              [Input("try bow", "n_clicks")],
              [State("modal bow", "is_open")],
              )
def toggle_modal_bow(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(Output("modal tfidf", "is_open"),
              [Input("try tfidf", "n_clicks")],
              [State("modal tfidf", "is_open")],
              )
def toggle_modal_tfidf(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output("feature extraction df", "children"),
    Input("select feature extraction", "value")
)
def display_feature_extraction_df(choice):
    if choice == "1":
        return vectoization(df_full, "bow")
    else:
        return vectoization(df_full, "tfidf")

def vectoization(df, type):
    if type == 'tfidf':
        vectorizer = TfidfVectorizer()
    elif type == 'bow':
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df.loc[:,'clean_text'])
    global df_vect
    df_vect = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    return dbc.Container(
            [dash_table.DataTable(df_vect[:5].to_dict('records'), [{"name": i, "id": j} for i,j in zip(vectorizer.get_feature_names_out(), df_vect.columns)], style_table={'overflowX': 'auto'}),
             html.P(f"Shape: {df_vect.shape}")])




@app.callback(Output("modal", "is_open"),
              Output("apple btn", "disabled"),
              Output("apple btn", 'children'),
              Output("reddit btn", "disabled"),
              Output("reddit btn", 'children'),
              Output("airline btn", "disabled"),
              Output("airline btn", 'children'),
              [Input("apple btn", "n_clicks"), Input("reddit btn", "n_clicks"), Input("airline btn", "n_clicks")],
              [State("modal", "is_open")],
              )
def toggle_modal(n1, n2, n3, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    global df_full
    if ctx.triggered[0]['value'] > 0:
        if ctx.triggered[0]['prop_id']:
            if button_id == 'apple btn':
                df_full = df_full.append(df3)
                return not is_open, True, "Added", False, None, False, None
            if button_id == 'reddit btn':
                df_full = df_full.append(df2)
                return not is_open, False, None, True, "Added", False, None
            if button_id == 'airline btn':
                df_full = df_full.append(df1)
                return not is_open, False, None, False, None, True, "Added"
    df_full.dropna(axis=0, inplace=True)
    df_full = df_full.sample(frac=1).reset_index(drop=True)
    return is_open, False, None, None, None, None, None


@app.callback(Output('freq dropdown', 'children'),
              Output('tabs-content-graph', 'figure'),
              Output('wordcloud', 'children'),
              Input('tabs-graph', 'value'))
def render_content(tab):
    if tab == 'tab-1-graph':
        categories = df_full.groupby('category', as_index=False).count().rename(columns={'clean_text': 'Total_Numbers'})
        return None, px.pie(categories, values='Total_Numbers', names='category'), None

    if tab == 'tab-2-graph':
        freq_df = draw_frequebcy_bars(df_full[df_full['category'] == 1]['clean_text'])
        # return px.bar(freq_df, x='word', y='count'
        #               )
        return dbc.DropdownMenu(label="Choose sentiment to display...", children=[
            dbc.DropdownMenuItem("Positive"),
            dbc.DropdownMenuItem("Negative"),
            dbc.DropdownMenuItem("Neutral"),
        ]), px.bar(freq_df,
                   x='word',
                   y='count'), None
    if tab == 'tab-3-graph':
        tweets_stats = pd.DataFrame()
        tweets_stats["clean_text"] = df_full["clean_text"].astype(str)
        tweets_stats["category"] = df_full["category"]
        tweets_stats['length'] = tweets_stats["clean_text"].apply(len)
        hist_data = [tweets_stats[tweets_stats['category'] == -1]['length'],
                     tweets_stats[tweets_stats['category'] == 0]['length'],
                     tweets_stats[tweets_stats['category'] == 1]['length']]

        group_labels = ['-1', '0', '1']
        colors = ['#A56CC1', '#A6ACEC', '#63F5EF']
        fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                                 bin_size=2, show_rug=False)
        fig.update_layout(
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="Blue",
            legend_title_font_color="green",
            title_text="Histogram of the lengths of the posts"
        )

        return None, fig.update_xaxes(range=[0, 450]), None
    if tab == 'tab-4-graph':
        img = io.BytesIO()
        plot_wordcloud().save(img, format='PNG')
        src = 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
        return None, None, html.Img(src=src)


def plot_wordcloud():
    wc = WordCloud(collocations=False, background_color="white", width=500, height=400,
                   max_words=100).generate(' '.join(df_full[df_full['category'] == 1]['clean_text']))
    return wc.to_image()


# @app.callback(
#     Output('tabs-content-graph', 'figure'),
#     Input('freq dropdown', 'value')
# )
# def update_output(value):
#     if value == 'Negative':
#         freq_df = draw_frequebcy_bars(df_full[df_full['category'] == -1]['clean_text'])
#         return px.bar(freq_df, x='word', y='count')

def draw_frequebcy_bars(series):
    freq_dict = defaultdict(int)
    for sample in series:
        for word in word_tokenize(sample):
            freq_dict[word] += 1
    patrial = dict(sorted(freq_dict.items(), key=lambda pair: pair[1], reverse=True)[:20])
    keys = tuple(patrial.keys())
    values = patrial.values()
    freq_df = pd.DataFrame()
    freq_df['word'] = keys
    freq_df['count'] = values
    return freq_df


@app.callback(Output('compare-data', 'children'),
              Input('submit-btn', 'n_clicks'),
              State('checklist', 'value'))
def submit_preprocessing(n_clicks, chck_values):
    if n_clicks >= 1:
        return preprocess(chck_values)


def preprocess(chck_values: []):
    df = df_full

    special_chars = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`',
                     '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/', '"']

    if 'Lowercasing' in chck_values:
        df['clean_text'] = df['clean_text'].str.lower()  # convert to lowercase
    if 'hyperlinks removal' in chck_values:
        df['clean_text'] = df['clean_text'].str.replace('http?:\/\/[^\s]+|www\.[^\s]', '',
                                                        regex=True)  # remove hyperlinks
    if 'users removal' in chck_values:
        df['clean_text'] = df['clean_text'].str.replace('@[\w]+', '', regex=True)  # remove users
    if 'hashtags removal' in chck_values:
        df['clean_text'] = df['clean_text'].str.replace('#\w+', '', regex=True)  # remove hashtags
    if 'numerical values removal' in chck_values:
        df['clean_text'] = df['clean_text'].str.replace('\d+', '', regex=True)  # remove numbers
    if 'symbols removal' in chck_values:
        for char in special_chars:
            df['clean_text'] = df['clean_text'].str.replace(char, '')  # special characters

    # remove stopwords
    # nltk.download()
    stopwords = nltk.corpus.stopwords.words('english')
    print(chck_values)
    if 'stopwords removal' in chck_values:
        df['clean_text'] = df['clean_text'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    # df = stemming(df)
    return dbc.Container(
        [dash_table.DataTable(df[:4].to_dict('records'), [{"name": i, "id": i} for i in df.columns], id='tbl',
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


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            global df_full
            df_full = df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

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


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('tbl', 'data'), Input('slider', 'value'))
def update_df_size(value):
    return df_full[:value].to_dict('records')


@app.callback(Output('dataset', 'children'),
              Input('comb-datasets-button', 'n_clicks'))
def update_dataset_view(n_clicks):
    if n_clicks >= 1:
        return dbc.Container(
            [dash_table.DataTable(df_full[:4].to_dict('records'), [{"name": i, "id": i} for i in df_full.columns],
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
             dbc.Alert(f"Shape: {df_full.shape}"),
             dcc.Slider(1, 20, 1, value=5, marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}, id='slider')])

    return html.Div()


if __name__ == "__main__":
    app.run_server(port=8888)
