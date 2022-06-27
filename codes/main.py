from scraping import RedditScraper
from data_loader import Data_Preparator
import pandas as pd
import ast
import praw
import openpyxl
import os
import glob
import numpy as np
import ast
import pickle
import re
from bs4 import BeautifulSoup
import re, unicodedata
import contractions
from preprocess_data import Data_preprocessor

from psaw import PushshiftAPI

#state your goal

scraping = False

prepare_data = False

preprocess_data = False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #Enter your credentials
    if scraping:
        Scraper = RedditScraper(client_id='',
                             client_secret='',
                             user_agent='',  # your user agent
                             username="",  # your reddit username
                             password="")

        #decide which subreddits you want to scrape
        subreddit_list = ["cripplingalcoholism", "disorder", "whatsbotheringyou"]


        limit_posts = 250000
        Scraper.run_scraper(subreddit_list, limit_posts, include_comments=False)

    with open('Emoji_Dict.p', 'rb') as fp:
        Emoji_Dict = pickle.load(fp)

    with open('Emoticon_Dict.p', 'rb') as fp:
        Emoticon_Dict = pickle.load(fp)

    Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

    print(Emoji_Dict)

    # adjust the parameters for data preparator. Data preparator can work as well with data obtaned from google big query

    if prepare_data:
        Loader = Data_Preparator(path_in='/Users/grzegorzantkiewicz/PycharmProjects/MasterThesis/googlequeryDS/raw', path_out='/Users/grzegorzantkiewicz/PycharmProjects/MasterThesis/googlequeryDS/', file_out="/psych_bert_google_merged.csv", google_query=True)

        df = Loader.save_the_data()

    if preprocess_data:


        df = pd.read_csv("/Users/grzegorzantkiewicz/PycharmProjects/MasterThesis/psych_bert_merged.csv")[1700000:]
        print(df)


        Data = Data_preprocessor(df, Emoticon_Dict, google_query=False)

        Data.preprocess_data(df, "/Users/grzegorzantkiewicz/PycharmProjects/MasterThesis/500_Reddit_users_posts_labels_preprocessed.csv")
