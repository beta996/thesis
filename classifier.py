import time

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import html, Output, Input, dash_table, State, dcc, Dash
from sklearn.pipeline import Pipeline

import app

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV


def page_content():
    return html.Div([html.H2('Classifier'),
                     dcc.Dropdown(['Naive Bayes', 'SVM', 'Decision Tree'], ['Naive Bayes', 'SVM', 'Decision Tree'],
                                  id='algorithms chosen',
                                  multi=True),
                     html.Div(id='choice'),

                     dbc.Button("Submit chosen classifiers and configs", color="primary", id='submit params'),
                     dcc.Loading(html.Div(id='all params')),
                     ])


@app.app.callback(Output('choice', 'children'), Input('algorithms chosen', 'value'))
def take_user_classifiers(value):
    returns = []
    if 'Naive Bayes' in value:
        returns.append(html.Div([
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(
                        [
                            html.Div(
                                [
                                    html.H5("NB configuration parameters", className="text-success"),
                                    html.Small("NB!", className="text-success"),
                                ],
                                className="d-flex w-100 justify-content-between",
                            ),
                            html.P("Alpha", className="text-success"),
                            html.Small("Alpha Laplace smoothing -  a smoothing technique that helps "
                                       "tackle the problem of zero probability in the Naïve "
                                       "Bayes machine learning algorithm",
                                       className="text-muted"),
                            dcc.RangeSlider(0, 20, 1, value=[5, 7], id='nb slider'),
                        ]
                    ), ])

        ]))
    if 'SVM' in value:
        returns.append(html.Div([
            dbc.ListGroup(
                [

                    dbc.ListGroupItem(
                        [
                            html.Div(
                                [
                                    html.H5("SVM configuration parameters", className="text-success"),
                                    html.Small("SVM!", className="text-success"),
                                ],
                                className="d-flex w-100 justify-content-between",
                            ),
                            html.P("C", className="text-success"),
                            html.Small("C -  a regularization parameter. The strength of the regularization is "
                                       "inversely proportional to C. Must be strictly positive. ",
                                       className="text-muted"),
                            dcc.RangeSlider(0, 20, 1, value=[5, 7], id='c slider'),
                            html.P("Kernel", className="text-success"),
                            html.Small("Kernel - Specifies the kernel type to be used in the algorithm. If none is "
                                       "given, ‘rbf’ will be used.", className="text-muted"),
                            dbc.Checklist(
                                options=[
                                    {"label": "linear", "value": 'linear'},
                                    {"label": "poly", "value": 'poly'},
                                    {"label": "rbf", "value": 'rbf'},
                                ],
                                value=['linear'],
                                id="kernel-input",
                                switch=True,
                                className="text-muted"
                            ),
                            html.Div(id='degree div')

                        ]
                    ),
                ])

        ]))
    if 'Decision Tree' in value:
        returns.append(html.Div([
            dbc.ListGroup(
                [

                    dbc.ListGroupItem(
                        [
                            html.Div(
                                [
                                    html.H5("Decision Tree configuration parameters", className="text-success"),
                                    html.Small("DT!", className="text-success"),
                                ],
                                className="d-flex w-100 justify-content-between",
                            ),
                            html.P("Splitter", className="text-success"),
                            html.Small("Splitter -  The strategy used to choose the split at each node. Supported "
                                       "strategies are “best” to choose the best split and 'random' to choose the "
                                       "best random split. ",
                                       className="text-muted"),
                            dbc.Checklist(
                                options=[
                                    {"label": "best", "value": 'best'},
                                    {"label": "random", "value": 'random'},
                                ],
                                value=['best'],
                                switch=True,
                                id='splitter checklist',
                                className="text-muted"
                            ),
                            html.P("max_depth", className="text-success"),
                            html.Small("max_depth - The maximum depth of the tree.", className="text-muted"),
                            dcc.RangeSlider(0, 2000, 100, value=[100, 200], id='max_depth slider'),
                            html.P("max_features", className="text-success"),
                            html.Small("max_features -  The number of features to consider when looking for the best split",
                                       className="text-muted"),
                            dbc.Checklist(
                                options=[
                                    {"label": "auto", "value": 'auto'},
                                    {"label": "sqrt", "value": 'sqrt'},
                                    {"label": "log2", "value": 'log2'},
                                ],
                                value=['auto'],
                                switch=True,
                                id='max_features checklist',
                                className="text-muted"
                            ),

                        ]
                    ),
                ])
        ]))

    return html.Div(returns)


@app.app.callback(Output('degree div', 'children'), Input('kernel-input', 'value'))
def display_degree_if_poly_chosen(value):
    if 'poly' in value:
        return [
            html.P("Degree", className="text-success"),
            html.Small("Degree -  Degree of the polynomial kernel function (‘poly’). Ignored by "
                       "all other kernels.",
                       className="text-muted"),
            dcc.RangeSlider(0, 10, 1, value=[5, 7], id='degree slider'),

        ]
    else:
        return html.Div(dcc.RangeSlider(0, 10, 1, value=[5, 7], id='degree slider'), style={"visibility": "hidden"})


@app.app.callback(Output('all params', 'children'), Input('submit params', 'n_clicks'), State('nb slider', 'value'),
                  State('c slider', 'value'), State('kernel-input', 'value'), State('degree slider', 'value'),
                  State('splitter checklist', 'value'), State('max_depth slider', 'value'), State('max_features checklist', 'value'))
def display_and_train(n_clicks, nb_alpha, c, kernel, degree, splitter, max_depth, max_features):
    df_feature_selection_full = app.df_feature_selection.copy()
    df_feature_selection_full['category'] = app.df_full['category']
    training_df, test_df = train_test_split(df_feature_selection_full, test_size=0.2)
    if n_clicks > 0:
        # time.sleep(5)

        best_params_nb, best_score_nb = train_nb(nb_alpha, training_df)
        best_params_svm, best_score_svm = train_SVM(c, kernel, degree, training_df)
        best_params_dt, best_score_dt = train_dt(splitter, max_depth, max_features, training_df)
        return f"NB with {best_params_nb} finished! Score = {best_score_nb} \n" \
               f"SVM with {best_params_svm} finished! Score = {best_score_svm} \n" \
               f"DT with {best_params_dt} finished! Score = {best_score_dt} \n"


def train_nb(nb_alpha, training_df):
    pipeline = Pipeline(
        [
            ("clf", MultinomialNB())
        ]
    )

    parameters = {
        "clf__alpha": nb_alpha,
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)  # defaut k=5
    grid_search.fit(training_df.iloc[:, :-1], training_df['category'])
    return grid_search.best_estimator_.get_params(), grid_search.best_score_


def train_SVM(c, kernel, degree, training_df):
    pipeline = Pipeline(
        [
            ("clf", SVC())
        ]

    )
    parameters = {
        "clf__C": c,
        "clf__kernel": kernel,
        "clf__degree": degree
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)  # defaut k=5
    grid_search.fit(training_df.iloc[:, :-1], training_df['category'])
    return grid_search.best_estimator_.get_params(), grid_search.best_score_


def train_dt(splitter, max_depth, max_features, training_df):
    pipeline = Pipeline(
        [
            ("clf", DecisionTreeClassifier())
        ]

    )
    parameters = {
        "clf__splitter": splitter,
        "clf__max_depth": max_depth,
        "clf__max_features": max_features
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)  # defaut k=5
    grid_search.fit(training_df.iloc[:, :-1], training_df['category'])
    return grid_search.best_estimator_.get_params(), grid_search.best_score_