import base64
import io
from collections import defaultdict

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from dash import html, dcc, Output, Input
from nltk import word_tokenize
from wordcloud import WordCloud

import app


def page_content():
    return html.Div([
        html.H2('Visualization'),
        dcc.Tabs(id="tabs-graph", value='tab-1-graph', children=[
            dcc.Tab(label='Pie chart', value='tab-1-graph'),
            dcc.Tab(label='Bar charts', value='tab-2-graph'),
            dcc.Tab(label='Histograms', value='tab-3-graph'),
            dcc.Tab(label='Tab Four', value='tab-4-graph')
        ]),
        html.Div(id='freq dropdown'),
        dcc.Graph(id='tabs-content-graph'),
        html.Div(id='wordcloud')
    ])


@app.app.callback(Output('freq dropdown', 'children'),
              Output('tabs-content-graph', 'figure'),
              Output('wordcloud', 'children'),
              Input('tabs-graph', 'value'))
def render_content(tab):
    if tab == 'tab-1-graph':
        categories = app.df_preprocessed.groupby('category', as_index=False).count().rename(columns={'clean_text': 'Total_Numbers'})
        return None, px.pie(categories, values='Total_Numbers', names='category'), None
    if tab == 'tab-2-graph':
        freq_df = draw_frequebcy_bars(app.df_preprocessed[app.df_preprocessed['category'] == 1]['clean_text'])
        # return px.bar(freq_df, x='word', y='count'
        #               )
        return dbc.DropdownMenu(label="Choose sentiment to display...", children=[
            dbc.DropdownMenuItem("Positive"),
            dbc.DropdownMenuItem("Negative"),
            dbc.DropdownMenuItem("Neutral"),
        ]), px.bar(freq_df,
                   x='word',
                   y='count'), None
    if tab == 'tab-3-graph':
        tweets_stats = pd.DataFrame()
        tweets_stats["clean_text"] = app.df_preprocessed["clean_text"].astype(str)
        tweets_stats["category"] = app.df_preprocessed["category"]
        tweets_stats['length'] = tweets_stats["clean_text"].apply(len)
        hist_data = [tweets_stats[tweets_stats['category'] == -1]['length'],
                     tweets_stats[tweets_stats['category'] == 0]['length'],
                     tweets_stats[tweets_stats['category'] == 1]['length']]

        group_labels = ['-1', '0', '1']
        colors = ['#A56CC1', '#A6ACEC', '#63F5EF']
        fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                                 bin_size=2, show_rug=False)
        fig.update_layout(
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="Blue",
            legend_title_font_color="green",
            title_text="Histogram of the lengths of the posts"
        )

        return None, fig.update_xaxes(range=[0, 450]), None
    if tab == 'tab-4-graph':
        img = io.BytesIO()
        plot_wordcloud().save(img, format='PNG')
        src = 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
        return None, None, html.Img(src=src)


def plot_wordcloud():
    wc = WordCloud(collocations=False, background_color="white", width=500, height=400,
                   max_words=100).generate(' '.join(app.df_preprocessed[app.df_preprocessed['category'] == 1]['clean_text']))
    return wc.to_image()


def draw_frequebcy_bars(series):
    freq_dict = defaultdict(int)
    for sample in series:
        for word in word_tokenize(sample):
            freq_dict[word] += 1
    patrial = dict(sorted(freq_dict.items(), key=lambda pair: pair[1], reverse=True)[:20])
    keys = tuple(patrial.keys())
    values = patrial.values()
    freq_df = pd.DataFrame()
    freq_df['word'] = keys
    freq_df['count'] = values
    return freq_df
