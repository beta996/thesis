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
    return dbc.Table.from_dataframe(app.run_jobs, striped=True, bordered=True, hover=True)
