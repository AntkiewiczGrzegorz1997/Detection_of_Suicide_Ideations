import json_lines
import pandas as pd
import json

import pandas as pd
import json
import ast
split_datasets = ["xaa", "xab", "xac", "xad", "xae"]

"""Especially SMHD for the bigger files need to be split into smaller in terminal with the following command
    The datasets after the split will probalby have the following names: ["xaa", "xab", "xac", "xad", "xae"]

    split files in the terminal: split -l 20000 /Users/grzegorzantkiewicz/Downloads/SMHDv1.1/SMHD_train.jl

    tutorial here: https://softhints.com/python-read-huge-json-file-pandas/


# Each train, test and validation dataset needs to be preprocessed separately
"""

#Specify if SMHD or RSDD
SMHD = True
split_datasets = ["/Users/grzegorzantkiewicz/PycharmProjects/MasterThesis/SMHD/train test or validation"]
df = pd.DataFrame()
import numpy as np
#df = pd.json_normalize(df['X_0'])


if SMHD:

    for item in split_datasets:

        with open(item) as f:
            df1 = pd.DataFrame(json.loads(line) for line in f)
        print(df1)
        df1.label = df1.apply(lambda x: x["label"][0], axis = 1 )
        df1.posts = df1.apply(lambda x: x["posts"][0], axis = 1 )

        df2 = pd.json_normalize(df1['posts'])
        result = pd.merge(df1[["id", "label"]], df2, left_index=True, right_index=True)

        df = df.append(result)

    df.to_csv("SMHD_train.csv")

else:
    for item in split_datasets:

        with open(item) as f:
            df = pd.DataFrame(json.loads(line) for line in f)

        df = df.add_prefix('X_')

        #df = pd.read_csv("RSDD_train.csv")
        df["posts"] = ""
        df["label"] = ""

        #df.X_0 = df.apply(lambda x: ast.literal_eval(x["X_0"]), axis=1)
        df.posts = df.apply(lambda x: x["X_0"]["posts"], axis=1)


        df.label = df.apply(lambda x: x["X_0"]["label"], axis=1)
        dataTypeSeries = df.posts.dtypes

        df.posts = df.apply(lambda x: [el[1] for el in x["posts"]], axis=1)
        df.posts = df.apply(lambda x: sorted(x["posts"], key=len, reverse=True)[0:3], axis=1)

        lst_col = 'posts'

        r = pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_col].str.len())
            for col in df.columns.drop(lst_col)}
        ).assign(**{lst_col: np.concatenate(df[lst_col].values)})[df.columns]
        df.posts = df.apply(lambda x: x["posts"][0], axis=1)
        df = pd.concat([df, r], axis=0)

        df.drop(['X_0'], axis=1, inplace=True)
        df.to_csv("RSDD_valid.csv")
