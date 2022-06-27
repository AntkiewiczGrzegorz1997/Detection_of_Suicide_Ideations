import pandas as pd
import numpy as np


import os
import glob

class Data_Preparator:
    def __init__(self,  path_out, file_out, path_in=False, google_query=False):
        self.path_in = path_in
        self.path_out = path_out
        self.file_out = file_out
        self.google_query = google_query

    def load_and_merge_data(self):
        print(self.path_in)
        if(self.path_in == False):
            self.path_in = os.getcwd()
            csv_files = glob.glob(os.path.join(self.path_in, "*.csv"))
        elif(self.google_query==True):
            csv_files = glob.glob(os.path.join(self.path_in, "*.csv"))
        else:
            csv_files = [self.path_in]

        df_empty = pd.DataFrame()

        for f in csv_files:
            # read the csv file
            df = pd.read_csv(f)

            df_empty = df_empty.append(df)


        return df_empty

    def drop_duplicates(self, df, duplicate_columns):

        df = df.drop_duplicates(subset=duplicate_columns, keep="first")

        return df


    def save_the_data(self):

        df_merged = self.load_and_merge_data()

        print("hi", df_merged)

        if(self.google_query==True):
            df_merged.rename(columns={"selftext":"body", "created_utc":"created"}, inplace=True)


        df_merged = self.drop_duplicates(df = df_merged, duplicate_columns = ["title", "id", "subreddit", "body"])

        df_merged = df_merged[df_merged['body'].notna()]

        print(df_merged)

        df_merged.to_csv(self.path_out + self.file_out)

        return df_merged

