import pandas as pd
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from sklearn.feature_selection import SelectKBest, chi2
import app


def page_content():
    return html.Div([html.H1("Feature Selection"),
                     html.P(f"Congratulations! Your vectorized dataframe has {app.df_vect.shape[1]} columns!"
                            f" That's a lot! However, A large number of irrelevant features increases the training "
                            f"time exponentially and increase the risk of overfitting."),
                     html.P(
                         "Feature selection is s a process of extracting the most relevant features from the dataset."
                         " It can help with a problem of too many columns in the vectorized dataframe."),
                     html.Ul(["Statistical tests: ", html.Li("Chi square Test")]),
                     dcc.Slider(1, 20, 1, value=5, marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                                id='slider feature selection'),
                     selectKBestChi(app.df_vect)
                     ])


def selectKBestChi(vect_df):
    select = SelectKBest(chi2, k=100)
    X_new = select.fit_transform(vect_df, app.df_full['category'])
    df_selected = pd.DataFrame(X_new)
    return dbc.Container(
        [dash_table.DataTable(df_selected[:5].to_dict('records'), [{"name": i, "id": j} for i, j in
                                                                   zip(select.get_feature_names_out(),
                                                                       df_selected.columns)],
                              style_table={'overflowX': 'auto'}),
         html.P(f"Shape: {df_selected.shape}")])
