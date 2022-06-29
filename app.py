import uuid

import dash_auth
import pandas as pd
from dash import dash, html, dcc
import dash_bootstrap_components as dbc
from collections import defaultdict
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform

import db
import job

app = DashProxy(external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP], transforms=[MultiplexerTransform()])


VALID_USERNAME_PASSWORD_PAIRS = {
    'hello': 'world'
}

# globals
df_full = pd.DataFrame()
df_preprocessed = pd.DataFrame()
df_feature_extraction = pd.DataFrame()
df_feature_selection = pd.DataFrame()
run_jobs = pd.DataFrame(columns=["algorithm", "config", "best_score", "time"])
connection = db.connection
current_job = job.Job(datasets=[], preprocessing_steps=[], feature_extraction_method='',
                      feature_selection_percent=0)

# from sklearn import datasets
# df_feature_selection = datasets.load_iris()
# df_feature_selection['category'] = [1 for _ in range df_feature_selection.sh]
df1 = pd.read_csv('./datasets/Tweets1.csv')
df2 = pd.read_csv('./datasets/Reddit_Data.csv')
df3 = pd.read_csv('./datasets/apple-twitter-sentiment-texts.csv')
df3 = df3[['clean_text', 'category']]
df1.dropna(axis=0, inplace=True)
df2.dropna(axis=0, inplace=True)
df3.dropna(axis=0, inplace=True)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)


def render_alert(message: str, link: str, color: str, url: str):
    return dbc.Alert(
        [
            html.I(className="bi bi-x-octagon-fill me-2"),
            html.P(message), dcc.Link(link, url)
        ],
        color=color,
        className="d-flex align-items-center",
    )