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


DEFAULT_GAME='Elden Ring'


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
        unique_games = np.unique(df['name'])
        game = st.selectbox('Select the games',unique_games,index=int(np.where(unique_games==DEFAULT_GAME)[0][0]))
    st.title(game)

    game_df = df.loc[df['name']==game].copy()
    game_df['n_comments'] = [len(comments) for comments in game_df['comments']]
    platforms = game_df['platform'].to_list()
    genres = game_df.iloc[0]['genres']
    
    col1,col2=st.columns(2)
    col3,col4=st.columns(2)
    with col1:
        st.write(f"**Platforms**: {' | '.join(platforms)}")
    with col2:
        st.write(f"**Release date**: {game_df.iloc[0]['release_date'].strftime('%d %b %Y')}")
    with col3:
        n_comments = sum(game_df['n_comments'])
        st.write(f"**Number of comments**: {n_comments}")
    with col4:
        st.write(f"**Genres**: {' | '.join(genres)}")

    cols_size = [3] + [1] + [1]*len(platforms)
    cols_grades = st.columns(cols_size)
    with cols_grades[0]:
        st.write(f"**Editorial Grade**:{game_df.iloc[0]['editorial_grade']}")
    with cols_grades[1]:
        st.write("**Users grade**:")

    for platform, col in zip(platforms,cols_grades[2:]):
        comments_platform = game_df[game_df['platform']==platform].iloc[0]['comments']
        grade_platform = sum([comment['grade'][0] for comment in comments_platform])/len(comments_platform)
        with col:
           st.write(f"{platform}: {grade_platform:.2f}")

    st.write("**Synopsis**:")
    st.write(game_df.iloc[0]['synopsis'])

    st.subheader('Number of comments per platform')
    fig_comments_byplatform = px.bar(game_df.sort_values('n_comments'),x='platform',y='n_comments')
    st.plotly_chart(fig_comments_byplatform,use_container_width=True)


    comments = [{'date':comment['date'][0],'grade':comment['grade'][0],'comment':comment['comment'][0],'username':comment['username'][0]} for comments in game_df['comments'] for comment in comments]
    df_comments = pd.DataFrame(comments)
    df_comments['date'] =pd.to_datetime(df_comments['date'])

    date_comments_bymonth = df_comments.resample('D', on='date')['username'].count().reset_index().rename(columns={'username':'count'})
    date_comments_bymonth['date'] = date_comments_bymonth['date'].dt.strftime('%d %b %Y')
    
    fig_bymonth = px.bar(date_comments_bymonth,x='date',y='count')
    st.subheader('Number of comments per day')
    st.plotly_chart(fig_bymonth,use_container_width=True)


    AgGrid(game_df)

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