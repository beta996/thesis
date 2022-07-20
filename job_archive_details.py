import dash
import pandas as pd
from dash import html, dcc
import plotly.figure_factory as ff


import app

dash.register_page(__name__, path_template="/job_archive_details/<job_id>")


def layout(report_id=None):
    # print(dash.page_registry)
    # q2 = """
    #             SELECT *
    #     FROM history.historical_jobs order by execution_time;
    #         """
    # db.read_query()

    #result_dataFrame = pd.read_sql(q2, app.connection)
    result_dataFrame = pd.read_sql_query("SELECT * FROM historical_jobs hj left join jobs j on hj.id=j.id order by execution_time;",
                                         app.connection)

    cm = result_dataFrame.iloc[int(report_id), -6]
    cm = cm.split("\n")
    for i, ele in enumerate(cm):
        cm[i] = ele.replace('[', '').replace(']', '').split(" ")
        cm[i] = [int(m) for m in cm[i] if m]
    z_text = [[str(y) for y in x] for x in cm]
    x = [-1, 0, 1]
    y = [1, 0, -1]
    fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add colorbar
    fig['data'][0]['showscale'] = True


    return html.Div([dcc.Graph(figure=fig), html.Div(
        f"Details: {result_dataFrame.iloc[int(report_id), -4:]}."
    )])