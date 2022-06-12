import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from st_aggrid import AgGrid
from tools.read import get_data
from review_bombing import *

import plotly.express as px
import plotly.graph_objects as go

DEFAULT_GAME = 'Elden Ring'


def main_page(df):
    st.title('Global statistics')

    col1, col2 = st.columns(2)

    with col1:
        st.header('Number of games')
        st.subheader(f"{len(np.unique(df['name']))} ({len(df['name'])} with platforms)")

    with col2:
        st.header('Number of comments')
        st.subheader(sum([len(comments) for comments in df['comments']]))

    platforms_count = df['platform'].value_counts().to_frame().reset_index() \
        .rename(columns={'platform': 'count', 'index': 'platform'})

    st.subheader('Number of games per platform')

    fig_platform = px.bar(platforms_count, x='platform', y='count')
    st.plotly_chart(fig_platform, use_container_width=True)

    genres_exploded = df.explode('genres')
    genres_count = genres_exploded['genres'].value_counts().to_frame().reset_index() \
        .rename(columns={'genres': 'count', 'index': 'genres'})
    st.subheader('Number of games per genre')
    fig_genres = px.bar(genres_count, x='genres', y='count')
    st.plotly_chart(fig_genres, use_container_width=True)

    release_date_bymonth = df.resample('M', on='release_date')['name'].count().reset_index() \
        .rename(columns={'name': 'count'})
    release_date_bymonth['release_date'] = release_date_bymonth['release_date'].dt.strftime('%b %Y')
    fig_bymonth = px.bar(release_date_bymonth, x='release_date', y='count')
    st.subheader('Number of games released per month')
    st.plotly_chart(fig_bymonth, use_container_width=True)


def game_page(df, positive_bombing_table, negative_bombing_table):
    with st.sidebar:
        unique_games = np.unique(df['name'])
        game = st.selectbox('Select the games', unique_games, index=int(np.where(unique_games == DEFAULT_GAME)[0][0]))
        review_bombing = st.checkbox('Review Bombing Comment')
        if review_bombing:
            confident = st.select_slider('Select the confidence of Review Bombing Detection',
                                         options=['LOW', 'MEDIUM', 'HIGH'], value='HIGH')

    st.title(game)

    game_df = df.loc[df['name'] == game].copy()
    game_df['n_comments'] = [len(comments) for comments in game_df['comments']]
    platforms = game_df['platform'].to_list()
    genres = game_df.iloc[0]['genres']

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    with col1:
        st.write(f"**Platforms**: {' | '.join(platforms)}")
    with col2:
        st.write(f"**Release date**: {game_df.iloc[0]['release_date'].strftime('%d %b %Y')}")
    with col3:
        n_comments = sum(game_df['n_comments'])
        st.write(f"**Number of comments**: {n_comments}")
    with col4:
        st.write(f"**Genres**: {' | '.join(genres)}")

    st.write(f"**Editorial Grade**:{game_df.iloc[0]['editorial_grade']}")
    st.write("**Users grade**:")

    cols_size = [1] * len(platforms)
    cols_grades = st.columns(cols_size)

    with_comments = True

    df_platforms_comments = {}
    for platform, col in zip(platforms, cols_grades):
        comments_platform = game_df[game_df['platform'] == platform].iloc[0]['comments']
        grade_platform = sum([comment['grade'][0] for comment in comments_platform]) / len(comments_platform)
        try:
            comments_platform = pd.DataFrame([{'date': comment['date'][0], 'grade': comment['grade'][0],
                                               'comment': comment['comment'][0], 'username': comment['username'][0]}
                                              for comment in comments_platform])
        except KeyError:
            with_comments = False

        if with_comments:
            comments_platform['date'] = pd.to_datetime(comments_platform['date'])
            df_platforms_comments[platform] = comments_platform
            # print(comments_platform['grade'])
            n_comments_0to5 = comments_platform['grade'].between(left=0, right=5).sum()
            n_comments_6to10 = comments_platform['grade'].between(left=6, right=10).sum()
            n_comments_11to15 = comments_platform['grade'].between(left=11, right=15).sum()
            n_comments_16to20 = comments_platform['grade'].between(left=16, right=20).sum()

            df_grades_range = pd.DataFrame(
                {'n_comments': [n_comments_0to5, n_comments_6to10, n_comments_11to15, n_comments_16to20],
                 'label': ['0 to 5', '6 to 10', '11 to 15', '16 to 20'],
                 'color': ['red', 'yellow', 'yellowgreen', 'green']})
            # fig_grades_range = px.bar(df_grades_range,x='n_comments',y='label',text_auto=True)
            fig_grades_range = go.Figure(data=[go.Bar(
                y=df_grades_range['label'],
                x=df_grades_range['n_comments'],
                marker_color=df_grades_range['color'],
                orientation='h',
                text=df_grades_range['n_comments'],
                textposition='inside'
            )])

            fig_grades_range.update_layout(
                yaxis_title='',
                yaxis_visible=True,
                yaxis_showticklabels=True,
                xaxis_visible=False,
                xaxis_showticklabels=False,
                margin=dict(l=5, r=5, b=5),
                title=f"{platform}: {grade_platform:.2f}",
                title_x=0.5,
                height=200)

            with col:
                st.plotly_chart(fig_grades_range, use_container_width=True, config=dict(displayModeBar=False))

    st.write("**Synopsis**:")
    st.write(game_df.iloc[0]['synopsis'])

    if with_comments:
        st.subheader('Number of comments per platform')
        fig_comments_byplatform = px.bar(game_df.sort_values('n_comments'), x='platform', y='n_comments')
        st.plotly_chart(fig_comments_byplatform, use_container_width=True)

        comments = [{'date': comment['date'][0], 'grade': comment['grade'][0], 'comment': comment['comment'][0],
                     'username': comment['username'][0]} for comments in game_df['comments'] for comment in comments]
        df_comments = pd.DataFrame(comments)
        # print(df_comments)
        df_comments['date'] = pd.to_datetime(df_comments['date'])

        date_comments_byday = df_comments.resample('D', on='date')['username'].count().reset_index() \
            .rename(columns={'username': 'count'})
        date_comments_byday['date'] = date_comments_byday['date'].dt.strftime('%d %b %Y')

        fig_byday = px.bar(date_comments_byday, x='date', y='count')
        st.subheader('Number of comments per day')
        st.plotly_chart(fig_byday, use_container_width=True)

        fig_mean_time = go.Figure()

        average_per_days = 4

        for platform, comments in df_platforms_comments.items():
            date_comments_byday_mean = comments.resample(f'{average_per_days}D', on='date')['grade'].mean() \
                .reset_index().rename(columns={'grade': 'mean'})
            date_comments_byday_mean['mean'] = date_comments_byday_mean['mean'].fillna(method='ffill')
            fig_mean_time.add_trace(go.Scatter(x=date_comments_byday_mean['date'], y=date_comments_byday_mean['mean'],
                                               mode='lines', name=platform))

        st.subheader(f'Average grades every {average_per_days} days per platform')
        st.plotly_chart(fig_mean_time, use_container_width=True)
        if review_bombing:
            positive_bombing = positive_bombing_table[(positive_bombing_table['game'] == game)
                                                      & (positive_bombing_table['confidence'] == confident)]
            negative_bombing = negative_bombing_table[(negative_bombing_table['game'] == game)
                                                      & (negative_bombing_table['confidence'] == confident)]
            try:
                st.subheader('Positive Review Bombing Example')
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    comm_num = st.number_input('Comment number', min_value=0,
                                               max_value=len(positive_bombing) - 1,
                                               value=0, step=1)
                with col2:
                    st.text_input('Username', positive_bombing.iloc[comm_num]['username'])
                comm = positive_bombing.iloc[comm_num]['comment']
                st.markdown(comm)
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    if st.button('Not Positive Bombing'):
                        positive_bombing_table.iloc[positive_bombing.index[comm_num]]['confidence'] = None
                        positive_bombing_table.to_csv('data/positive_bombing.csv')

            except:
                st.markdown('No positive comment considered as Review Bombing')
            try:
                st.subheader('Negative Review Bombing Example')
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    comm_num = st.number_input('Comment number', min_value=0,
                                               max_value=len(negative_bombing) - 1,
                                               value=0, step=1)
                with col2:
                    st.text_input('Username', negative_bombing.iloc[comm_num]['username'])
                comm = negative_bombing.iloc[comm_num]['comment']
                st.markdown(comm)
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    if st.button('Not Negative Bombing'):
                        negative_bombing_table.iloc[negative_bombing.index[comm_num]]['confidence'] = None
                        negative_bombing_table.to_csv('data/negative_bombing.csv')
            except:
                st.markdown('No negative comment considered as Review Bombing')
    else:
        st.subheader("No comments")


def comment_page(df):
    pass


@st.cache()
def load_data():
    df = get_data('data/dataset500.csv')
    return df


@st.cache(allow_output_mutation=True)
def load_review_bombing():
    pos = pd.read_csv('data/positive_bombing.csv')
    neg = pd.read_csv('data/negative_bombing.csv')
    return pos, neg


if __name__ == "__main__":
    st.set_page_config(page_title='JVC analytics', layout='wide')

    df = load_data()
    positive_bombing_table, negative_bombing_table = load_review_bombing()

    with st.sidebar:
        st.subheader('navigation')
        page_selected = st.radio('Go To', ['Main', 'Game', 'Comments'])

    if page_selected == 'Main':
        main_page(df)
    elif page_selected == 'Game':
        game_page(df, positive_bombing_table, negative_bombing_table)
    elif page_selected == 'Comments':
        comment_page(df)
