import altair as alt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
pio.renderers.default = "browser"

filepath_dummy_data = '../../dummy_data/hp_10m_2023.csv'


def altair_friendly(df, col1='kWh', col2='use'):
    """Convert conventional dataframe format to altair chart format

    :param df:
    :return:
    """
    new_df = pd.DataFrame(columns=[col1, col2])
    for col in df.items():
        col[1].name = col1
        df_app = pd.DataFrame(col[1])
        new_df = pd.concat([new_df, df_app])
        new_df[col2].fillna(col[0], inplace=True)

    new_df['timestamp'] = new_df.index

    return new_df

# dummy data
df_hp_history = pd.read_csv(filepath_dummy_data, index_col=0, parse_dates=True)
time_frame = (df_hp_history.index[0].day, df_hp_history.index[-1].day)
df_monthly = df_hp_history.resample('M').sum()

eui_instantaneous = 22.5


# subplots
fig = make_subplots(
    rows=1,
    cols=1,
    specs=[[{'type': 'indicator'}]] #{'type': 'indicator'}]]
)

fig.add_trace(
    go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=eui_instantaneous,
        mode="gauge+number+delta",
        title={'text': "Current EUI"},
        delta={'reference': 7},
        gauge={'axis': {'range': [None, 50], 'tickwidth': 1,'tickcolor': "black"},
            'bar': {'color': "MidnightBlue"},
                 'steps' : [
                     {'range': [0, 5], 'color': '#f8991d'},
                     {'range': [5, 10], 'color': '#f8a334'},
                     {'range': [10, 15], 'color': "#f9ad4a"},
                     {'range': [15, 20], 'color': '#fab760'},
                     {'range': [20, 25], 'color': '#fac176'},
                     {'range': [25, 30], 'color': '#fbcb8d'},
                     {'range': [30, 35], 'color': '#fcd6a3'},
                     {'range': [35, 40], 'color': '#fde0ba'},
                     {'range': [40, 45], 'color': '#fdead1'},
                     {'range': [45, 50], 'color': '#fef5e8'}
                 ],
                 'threshold' : {'line': {'color': "navy", 'width': 4}, 'thickness': 0.75, 'value': 7}}
    ), row=1, col=1)

fig.update_layout(
    # font_family='IBM Plex Sans',
    width=330,
    height=330
)

# this launches the graph in a web browser
fig.show()


### Monthly Summaries
plot = alt.Chart(df_monthly).mark_bar().encode(
    column=alt.Column(
        'month:N',
        title='',
        header=alt.Header(labelOrient='bottom'),
        sort=['January']
    ),
    # labelAlign='right')),
    x=alt.X('category', title='', axis=alt.Axis(labels=False)),
    y='kWh',
    color=alt.Color('category', title='')
)#.properties(title=alt.TitleParams('Monthly Comparison to Baseline', anchor='middle'))

plot.show()

pass

