import pandas as pd
import ast


def get_data(filename):
    df = pd.read_csv(filename)
    df.loc[:, "comments"] = df.apply(lambda x: ast.literal_eval(x['comments']), axis=1)
    df.loc[:,'comments'] = df.apply(lambda x: (x['comments'],) if isinstance(x['comments'],dict) else x['comments'],axis=1)
    df.loc[:,'genres'] = df.apply(lambda x: [genre.strip() for genre in x['genres'].split(',')],axis=1)
    df['release_date'] = pd.to_datetime(df['release_date'],format='%b %d, %Y @ %H:%M:%S.%f')
    return df

