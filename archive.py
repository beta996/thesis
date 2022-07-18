import time

import dash.html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import html, Output, Input, dash_table, State, dcc, Dash
from sklearn.pipeline import Pipeline
import plotly.figure_factory as ff


import app

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

import db


def page_content():
    q1 = """
    SELECT execution_time, algorithm, best_score, config, duration FROM history.historical_jobs order by execution_time;
    """
    # db.read_query()

    result_dataFrame = pd.read_sql(q1, app.connection)
    lst = []
    dfg = dash_table.DataTable(result_dataFrame.to_dict('records'), [{"name": i, "id": i} for i in result_dataFrame.columns], id='archive-table')
    dfg.columns.append({'name':'buttons', 'id':'buttons'})
    for row in dfg.data:
        row['buttons'] = "View"
    return html.Div([dfg, html.Div(id="details-div")])
    # return dash_table.DataTable(result_dataFrame.to_dict('records'), [{"name": i, "id": i} for i in result_dataFrame.columns], id='tbl')


@app.app.callback(
    Output('details-div', 'children'),
    Input('archive-table', 'active_cell'),
    State('archive-table', 'data')
)
def getActiveCell(active_cell, data):
    if active_cell and active_cell['column_id'] == "buttons":
        col = active_cell['column_id']
        row = active_cell['row']
        cellData = data[row][col]
        q2 = """    
            SELECT *,   
        ROW_NUMBER() OVER(order by execution_time) - 1 row_num  
    FROM history.historical_jobs order by execution_time;  
        """
        # db.read_query()


        result_dataFrame = pd.read_sql(q2, app.connection)
        cm = result_dataFrame.iloc[row,-2]
        cm = cm.split("\n")
        for i,ele in enumerate(cm):
            cm[i] = ele.replace('[', '').replace(']', '').split(" ")
            cm[i] = [int(m) for m in cm[i] if m]
        z_text = [[str(y) for y in x] for x in cm]
        x = [-1, 0, 1]
        y = [1, 0, -1]
        fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

        # add colorbar
        fig['data'][0]['showscale'] = True
        return dcc.Graph(figure=fig)