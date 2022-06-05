import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import html, Output, Input, dash_table, State, dcc
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import app


def page_content():
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
                     dcc.Loading(html.Div(id='feature extraction df')),
                     html.P(),
                     dbc.Modal(
                         [
                             dbc.ModalHeader(dbc.ModalTitle("BOW")),
                             html.Div([html.P('Lets take 2 examplary posts:'),
                                       html.P(app.df_preprocessed['clean_text'][1], style={"font-weight": "bold"}),
                                       html.P(app.df_preprocessed['clean_text'][2], style={"font-weight": "bold"})
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


@app.app.callback(Output('explanation div bow', 'children'),
              Input('next step bow', 'n_clicks'))
def next_step_submit_bow(n_clicks):
    first_text = app.df_preprocessed['clean_text'][1]
    second_text = app.df_preprocessed['clean_text'][2]
    # distinct_tokens = [word for word in (first_text + second_text).split()]
    l_doc1 = first_text.split()
    l_doc2 = second_text.split()
    distinct_tokens = np.union1d(l_doc1, l_doc2)
    bow1 = calculate_bow(distinct_tokens, l_doc1)
    bow2 = calculate_bow(distinct_tokens, l_doc2)
    df_bow = pd.DataFrame([bow1, bow2])
    if n_clicks == 1:
        return html.Div([html.P('2. We consider all the unique words from the above set of reviews to create a '
                                'vocabulary, which is going to be as follows :'),
                         html.P(f'[{distinct_tokens}]')
                         ])
    elif n_clicks == 2:
        return html.Div(
            [html.P("3. In the third step, we create a matrix of features by assigning a separate column for "
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


def calculate_bow(wordset, l_doc):
    tf_diz = dict.fromkeys(wordset, 0)
    for word in l_doc:
        tf_diz[word] = l_doc.count(word)
    return tf_diz


@app.app.callback(Output('explanation div tfidf', 'children'),
              Input('next step tfidf', 'n_clicks'))
def next_step_submit_tfidf(n_clicks):
    if n_clicks == 1:
        return html.Div([html.P('TF-IDF is a combination of Term Frequency and Inverse Document Frequency.'),
                         html.P(f'First, let us calculate the Term Frequency: '),
                         html.P(
                             f'TF(w) = (number of times word w appears in a document) / (total number of words in the '
                             f'document)'),
                         html.P(f'TF(people) = 4 / 100 = 0.04')

                         ])
    elif n_clicks == 2:
        return html.Div(
            [html.P("Now we measure Inverse Document Frequency, It measures how important a word is for the corpus."),
             html.P("IDF(w) = log(total number of documents / number of documents with w in it)"),
             html.P("IDF(people) = log(1000 / 100) = 1"),
             ])
    elif n_clicks == 3:
        return html.Div([
            html.P("Finally, to calculate TF-IDF, we multiply these two factors – TF and IDF."),
            html.P("TF-IDF(people) = TF(people) x IDF(people) = 0.04 * 1 = 0.04"),
        ])


@app.app.callback(Output("modal bow", "is_open"),
              [Input("try bow", "n_clicks")],
              [State("modal bow", "is_open")])
def toggle_modal_bow(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.app.callback(Output("modal tfidf", "is_open"),
              [Input("try tfidf", "n_clicks")],
              [State("modal tfidf", "is_open")])
def toggle_modal_tfidf(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.app.callback(
    Output("feature extraction df", "children"),
    Input("select feature extraction", "value")
)
def display_feature_extraction_df(choice):
    if choice == "1":
        return vectoization(app.df_preprocessed, "bow")
    else:
        return vectoization(app.df_preprocessed, "tfidf")


def vectoization(df, type):
    if type == 'tfidf':
        vectorizer = TfidfVectorizer()
    elif type == 'bow':
        vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(df.loc[:, 'clean_text'])
    app.df_feature_extraction = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

    return dbc.Container(
        [dash_table.DataTable(app.df_feature_extraction[:5].to_dict('records'), [{"name": i, "id": j} for i, j in
                                                                   zip(vectorizer.get_feature_names_out(),
                                                                       app.df_feature_extraction.columns)],
                              style_table={'overflowX': 'auto'}),
         html.P(f"Shape: {app.df_feature_extraction.shape}")])
