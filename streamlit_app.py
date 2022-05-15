import streamlit as st
import pandas as pd
import numpy as np

from st_aggrid import AgGrid
from tools.read import get_data


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



@st.cache()
def load_data():
    df = get_data('data/dataset500.csv')
    return df

if __name__ == "__main__":
    st.set_page_config(page_title='JVC analytics')

    df = load_data()
    with st.sidebar:
        st.subheader('navigation')
        page_selected = st.radio('Go To',['Main','Game','Comments'])

    if page_selected == 'Main':
        pass
    elif page_selected == 'Game':
        with st.sidebar:
            game = st.selectbox('Select the games',np.unique(df['name']))
        st.title('yo')
        AgGrid(df[df['name']==game])
    elif page_selected == 'Comments':
        pass