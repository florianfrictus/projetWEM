import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from st_aggrid import AgGrid
from tools.read import get_data

import plotly.express as px

# Main page:
# Number of games
# Number of comments
# Number of games per platform
# Number of games per genre
# Number of games by release_date (per month for example)

# Games (one by one):
# Number of comments
# Global info of the game (platform, release date, genre)
# Number of comments per platform
# Number of comments per release_date (with bins)


# Comments:

def main_page():
    st.title('Global statistics')

    col1, col2 = st.columns(2)

    with col1:
        st.header('Number of games')
        st.subheader(f"{len(np.unique(df['name']))} ({len(df['name'])} with platforms)")

    with col2:
        st.header('Number of comments')
        st.subheader(sum([len(comments) for comments in df['comments']]))



    platforms_count = df['platform'].value_counts().to_frame().reset_index().rename(columns={'platform':'count','index':'platform'})

    st.subheader('Number of games per platform')

    fig_platform = px.bar(platforms_count,x='platform',y='count')
    st.plotly_chart(fig_platform,use_container_width=True)

    genres_exploded = df.explode('genres')
    genres_count = genres_exploded['genres'].value_counts().to_frame().reset_index().rename(columns={'genres':'count','index':'genres'})
    st.subheader('Number of games per genre')
    fig_genres = px.bar(genres_count,x='genres',y='count')
    st.plotly_chart(fig_genres,use_container_width=True)

    release_date_bymonth = df.resample('M', on='release_date')['name'].count().reset_index().rename(columns={'name':'count'})
    release_date_bymonth['release_date'] = release_date_bymonth['release_date'].dt.strftime('%b %Y')
    fig_bymonth = px.bar(release_date_bymonth,x='release_date',y='count')
    st.subheader('Number of games released per month')
    st.plotly_chart(fig_bymonth,use_container_width=True)



def game_page():
    with st.sidebar:
        game = st.selectbox('Select the games',np.unique(df['name']))
    st.title('yo')
    AgGrid(df[df['name']==game])

def comment_page():
    pass


@st.cache()
def load_data():
    df = get_data('data/dataset500.csv')
    return df

if __name__ == "__main__":
    st.set_page_config(page_title='JVC analytics',layout='wide')

    df = load_data()

    with st.sidebar:
        st.subheader('navigation')
        page_selected = st.radio('Go To',['Main','Game','Comments'])

    if page_selected == 'Main':
        main_page()
    elif page_selected == 'Game':
        game_page()
    elif page_selected == 'Comments':
        comment_page()