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

import db


def page_content():
    q1 = """
    SELECT * FROM history.historical_jobs;
    """
    # db.read_query()
    result_dataFrame = pd.read_sql(q1, app.connection)
    return dbc.Table.from_dataframe(result_dataFrame, striped=True, bordered=True, hover=True)
