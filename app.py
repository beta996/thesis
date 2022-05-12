import dash_auth
import pandas as pd
from dash import dash
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP])

VALID_USERNAME_PASSWORD_PAIRS = {
    'hello': 'world'
}

df_full = pd.DataFrame()
df_vect = pd.DataFrame()

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