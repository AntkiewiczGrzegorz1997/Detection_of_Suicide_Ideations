import praw
import pandas as pd
import openpyxl

from psaw import PushshiftAPI
import datetime as dt

class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent, username, password):
        self.client_id = client_id
        self.client_secret =client_secret
        self.user_agent = user_agent
        self.username = username
        self.password = password


    def establish_connection(self):

        reddit = praw.Reddit(client_id = self.client_id,
                             client_secret = self.client_secret,
                             user_agent = self.user_agent ,  # your user agent
                             username = self.username,  # your reddit username
                             password = self.password)

        api = PushshiftAPI(reddit)

        return api, reddit

    def add_comments(self, url_column, comments_text, id, bot_present, reddit):

        comments = []
        url = url_column
        comments_text = comments_text

        submission = reddit.submission(id=id)

        for top_level_comment in submission.comments:
            comments.append(top_level_comment.body)

        if bot_present:
            del comments[0]

        comments_text = comments

        return comments_text



    def run_scraper(self, subreddits, limit_posts, include_comments):

        api, reddit = self.establish_connection()

        for subreddit in subreddits:
            submissions = list(api.search_submissions(subreddit=subreddit,
                                                      filter=['title', 'score', 'id', 'subreddit', 'url',
                                                              'num_comments', 'comments', 'body', 'created', 'selftext',
                                                              'limit', 'created'],
                                                      limit=limit_posts))

            posts = []
            for post in submissions:
                posts.append(
                    [post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.comment_limit,
                     post.selftext, post.created])

            posts = pd.DataFrame(posts, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'comments',
                                                 'body', 'created'])

            # delete posts that are not text
            posts = posts.drop(
                posts[(posts['body'] == '') | (posts['body'] == '[gelöscht]') | (posts['body'] == '[entfernt]')].index)
            print(len(posts))
            # If each post hast more than 0 comments there probably is a bot comment which we have to delete
            bot_present = (posts.num_comments > 0).all()

            # create new column
            posts['comments_text'] = ''
            print(posts)

            if (include_comments == True):

                posts.comments_text = posts.apply(
                    lambda x: self.add_comments(x['url'], x['comments_text'], x['id'], bot_present=bot_present, reddit=reddit), axis=1)

                path = f"/Users/grzegorzantkiewicz/PycharmProjects/MasterThesis/{subreddit}_{limit_posts}.csv"

            else:
                path = f"/Users/grzegorzantkiewicz/PycharmProjects/MasterThesis/{subreddit}_{limit_posts}_no_comments.csv"

            posts.to_csv(path)















