import pandas as pd
import ast


def get_data(filename):
    df = pd.read_csv(filename)
    df.loc[:, "comments"] = df.apply(lambda x: ast.literal_eval(x['comments']), axis=1)
    df.loc[:,'comments'] = df.apply(lambda x: (x['comments'],) if isinstance(x['comments'],dict) else x['comments'],axis=1)
    return df

