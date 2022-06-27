import pandas as pd
import numpy as np
import ast

import os
import glob

import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import re, unicodedata
import string as improvedstring
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
import re
import pickle
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import emoji



def stopword(data):

  nltk.download('stopwords')
  clean = []
  for i in data:
    if i not in stopwords.words('english'):
      clean.append(i)
  return clean

def stemming(data):
  stemmer = LancasterStemmer()
  stemmed = []
  for i in data:
    stem = stemmer.stem(i)
    stemmed.append(stem)
  return stemmed

def lemmatization(data):
  nltk.download('wordnet')
  lemma = WordNetLemmatizer()
  lemmas = []
  for i in data:
    lem = lemma.lemmatize(i, pos='v')
    lemmas.append(lem)
  return lemmas






class Data_preprocessor:
    def __init__(self, file_in, Emoticon_Dict, google_query=False):
        self.file_in = file_in
        self.Emoticon_Dict = Emoticon_Dict
        self.google_query = google_query



    #df_merged = pd.read_csv("/Users/grzegorzantkiewicz/PycharmProjects/MasterThesis/depression_help_50000.csv")



    def delete_bots(self, column_with_bots):

        comments_text = column_with_bots

        comments_text = ast.literal_eval(comments_text)

        comments_text = [x for x in comments_text if " bot," not in x and " bot." not in x and " bot " not in x
                         and " Bot," not in x and " Bot." not in x and " Bot " not in x
                         and " BOT," not in x and " BOT." not in x and " BOT " not in x and "Thank you for submitting" not in x]

        if len(comments_text) == 0:
            comments_text.append("empty")

        return comments_text

    def comments_to_text(self, df, comment_column):

        r = pd.DataFrame({
            col: np.repeat(df[col].values, df[comment_column].str.len())
            for col in df.columns.drop(comment_column)}
        ).assign(**{comment_column: np.concatenate(df[comment_column].values)})[df.columns]

        r = r[
            (r[comment_column] != "empty") & (r[comment_column] != "[entfernt]") & (r[comment_column] != "[gelöscht]")]

        r["body"] = r[comment_column]
        print(len(r))

        df = pd.concat([df, r], axis=0)

        del r

        df[comment_column] = ''

        return df



    def titles_to_text(self, df, title_column):

        title_df = df.copy()

        title_df["body"] = title_df["title"]

        return title_df



    def convert_emojis_to_word(self, text):

        text = emoji.demojize(text, delimiters=("", ""))

        return text


    # Function for converting emoticons into word
    def convert_emoticons_to_word(self, text):
        for emot in EMOTICONS:

            text = re.sub(u'(' + emot + ')', "_".join(EMOTICONS[emot].replace(",", "").split()), text)
        return text

    def clean_text(self, text):

        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)

        text = re.sub(r"\-", " ", text)

        text = re.sub(r"\=", " ", text)


        # text = re.sub(r"\'s", " ", text)

        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r" ve ", " have ", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " d ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"y'", "you ", text)

        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " usa ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)


        #text = re.sub(r"[^A-Za-z0-9^,!.]", " ", text)
        text = re.sub(r"[^A-Za-z0-9^,!.\/+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)

        ## Remove html content
        text = BeautifulSoup(text, "html.parser").text

        ## Remove contractions
        text = contractions.fix(text)

        return text

    def preprocess_data_pred(self, file_in, file_out):

        def preprocess_column(file_in, column):


            file_in = file_in[file_in[column].notna()]

            file_in[column] = file_in.apply(lambda x: self.convert_emojis_to_word(x[column]), axis=1)

            file_in[column] = file_in.apply(lambda x: self.convert_emoticons_to_word(x[column]), axis=1)

            file_in[column] = file_in.apply(lambda x: re.sub(r'^https?:\/\/.*[\r\n]*', '', x[column], flags=re.MULTILINE), axis=1)

            file_in[column] = file_in.apply(lambda x: x[column].lower(), axis=1)

            file_in[column] = file_in.apply(lambda x: self.clean_text(x[column]), axis=1)

            return file_in[column]

        file_in["selftext"] = preprocess_column(file_in, "selftext")
        file_in["title"] = preprocess_column(file_in, "title")

        file_in.to_csv(file_out)

    def preprocess_data_pred_2(self, file_in, file_out):

        colname = "Post"

        file_in.Post = file_in.apply(lambda x:  x["Post"] + "']" if not x["Post"].endswith("']") and not x["Post"].endswith('"]')  else x["Post"], axis=1)

        for i in range(len(file_in)):
            file_in.Post.iloc[i] = ast.literal_eval(file_in.Post.iloc[i])
        file_in = pd.DataFrame({
            col: np.repeat(file_in[col].values, file_in[colname].str.len())
            for col in file_in.columns.drop(colname)}
        ).assign(**{colname: np.concatenate(file_in[colname].values)})[file_in.columns]


        file_in = file_in[file_in["Post"].notna()]

        file_in.Post = file_in.apply(lambda x: self.convert_emojis_to_word(x["Post"]), axis=1)

        file_in.Post = file_in.apply(lambda x: self.convert_emoticons_to_word(x["Post"]), axis=1)

        file_in.Post = file_in.apply(lambda x: re.sub(r'^https?:\/\/.*[\r\n]*', '', x["Post"], flags=re.MULTILINE), axis=1)

        file_in.Post = file_in.apply(lambda x: x["Post"].lower(), axis=1)

        file_in.Post = file_in.apply(lambda x: self.clean_text(x["Post"]), axis=1)

        file_in.rename(columns={"Post": "selftext", "User":"name"}, inplace=True)
        print(file_in)



        file_in.to_csv(file_out)




    def preprocess_data(self, file_in, file_out):


        lst_col = 'comments_text'
        title_col = "title"

        if(self.google_query == False):
            file_in["comments_text"] = file_in["comments_text"].replace(np.nan, '[]', regex=True)
            file_in.comments_text = file_in.apply(lambda x: self.delete_bots(x['comments_text']), axis=1)


        file_in_title = self.titles_to_text(file_in, title_col)

        if (self.google_query == False):
            file_in = self.comments_to_text(file_in, lst_col)

        file_in = pd.concat([file_in_title, file_in], axis=0)

        del file_in_title

        #sort on two columns
        file_in = file_in.sort_values(["id", "created"], ascending = (None, True))

        if (self.google_query == False):
            file_in.dropna(subset=[lst_col], inplace=True)

        file_in = file_in[file_in['body'].notna()]

        file_in[
            (file_in['body'] != "empty") & (file_in['body'] != "[removed]") & (file_in['body'] != "[deleted]")]

        file_in.body = file_in.apply(lambda x: self.convert_emojis_to_word(x['body']), axis=1)

        file_in.body = file_in.apply(lambda x: self.convert_emoticons_to_word(x['body']), axis=1)

        #delete urls from text

        file_in.body = file_in.apply(lambda x: re.sub(r'^https?:\/\/.*[\r\n]*', '', x["body"], flags=re.MULTILINE), axis=1)

        #lower text
        file_in.body = file_in.apply(lambda x: x['body'].lower(), axis=1)

        # clean text
        file_in.body = file_in.apply(lambda x: self.clean_text(x['body']), axis=1)

        #file_in.to_csv(file_out)

        with open(file_out, 'w') as f:
            f.write(file_in['body'].str.cat(sep='\n'))

        '''

        file = open(file_out, 'r')
        read_data = file.read()
        per_word = read_data.split()

        print('Total Words:', len(per_word))
        '''
        return file_in



