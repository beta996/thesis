import pandas as pd
from dash import html, dcc, dash_table, Output, Input
import dash_bootstrap_components as dbc
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest
import app
import plotly.express as px


def page_content():
    bar_plot = selectKBest(app.df_feature_extraction,10)
    return dcc.Loading(html.Div([html.H1("Feature Selection"),
                     html.P(f"Congratulations! Your vectorized dataframe has {app.df_feature_extraction.shape[1]} columns!"
                            f" That's a lot! However, A large number of irrelevant features increases the training "
                            f"time exponentially and increase the risk of overfitting."),
                     html.P(
                         "Feature selection is s a process of extracting the most relevant features from the dataset."
                         " It can help with a problem of too many columns in the vectorized dataframe."),
                     html.Ul(["Statistical tests: ", html.Li("Chi square Test")]),
                     dcc.Slider(1, 100, 1, value=50, marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                                id='slider feature selection'),
                     html.Div(selectPercentileChi(app.df_feature_extraction, 50), id='selected df'),
                     html.Div(dcc.Graph(figure=bar_plot))
                     ]))


@app.app.callback(Output('selected df', 'children'), Input('slider feature selection', 'value'))
def update_df_size(value):
    return selectPercentileChi(app.df_feature_extraction, value)


def selectPercentileChi(vect_df, percentile):
    select = SelectPercentile(chi2, percentile=percentile)
    X_new = select.fit_transform(vect_df, app.df_full['category'])
    app.df_feature_selection = pd.DataFrame(X_new)
    return dbc.Container(
        [dash_table.DataTable(app.df_feature_selection[:5].to_dict('records'), [{"name": i, "id": j} for i, j in
                                                                   zip(select.get_feature_names_out(),
                                                                       app.df_feature_selection.columns)],
                              style_table={'overflowX': 'auto'}, id='tbl feature selection'),
         html.P(f"Shape: {app.df_feature_selection.shape}")])


def selectKBest(vect_df, k):
    chi2score, _ = chi2(vect_df, app.df_full['category'])
    x = sorted(list(zip(chi2score, vect_df.columns)), reverse=True)[:k]
    bar_df = pd.DataFrame(x, columns=['score', 'column'])
    return px.bar(bar_df,
                  x='column',
                  y='score')