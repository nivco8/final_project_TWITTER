import config
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from pickle import *

os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")


# -------------------pre processing - concatenating tweets from each user ---------------------------

# df = pd.read_csv('full_df.csv').drop('Unnamed: 0', axis=1)

df = pd.read_pickle('full_df_new_pickle.pickle')


author_id_list = []
text_concat = []

for authorId in df['author_id'].unique(): #loop for all author_id
    df_temp = df[df['author_id']==authorId]
    all_user_tweets = ''
    for idx, row in df_temp.iterrows():
        all_user_tweets = all_user_tweets + row['text']

    author_id_list.append(authorId)
    text_concat.append(all_user_tweets)

text_rep_df = pd.DataFrame()

text_rep_df['author_id'] = author_id_list
text_rep_df['text_concat'] = text_concat




# regular expressions used to clean up the tweet data
http_re = re.compile(r'\s*http://[^\s]*')
https_re = re.compile(r'\s*https://[^\s]*')
remove_ellipsis_re = re.compile(r'\.\.\.')
at_sign_re = re.compile(r'\@\S+')
punct_re = re.compile(r"[\"'\[\],.:;()\-&!]")
price_re = re.compile(r"\d+\.\d\d")
number_re = re.compile(r"\d+")

mystopwords = ["&amp;", 'amp', 'let', 'll', 've', 'rt']  # added &amp; to avoid the sign & as a word

stop_words = text.ENGLISH_STOP_WORDS.union(mystopwords)


# converts to lower case and clean up the text
def normalize_tweet(tweet):
    tweet = str(tweet)
    t = tweet.lower()
    t = re.sub(price_re, 'PRICE', t)
    t = re.sub(remove_ellipsis_re, '', t)
    t = re.sub(http_re, ' LINK', t)
    t = re.sub(https_re, ' LINK', t)
    t = re.sub(punct_re, '', t)
    t = re.sub(at_sign_re, '@', t)
    t = re.sub(number_re, 'NUM', t)
    return t


def Count_Vectorizer(User_Data_train, col='Bio'):
    word_vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word', max_df=0.5, min_df=3)
    sparse_matrix = word_vectorizer.fit_transform(User_Data_train[col])
    clf = TruncatedSVD(10)
    Xpca = clf.fit_transform(sparse_matrix)
    svd = pd.DataFrame(Xpca, index=User_Data_train.index.values)
    return [User_Data_train, svd]


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))



def Latent_Dirichlet_Allocation(User_Data_train, col):
    tf_vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word', max_df=0.5, min_df=3, stop_words='english')
    tf = tf_vectorizer.fit_transform(User_Data_train[col])
    tf_feature_names = tf_vectorizer.get_feature_names()
    model = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',
                                      learning_offset=50., random_state=1)
    lda = model.fit(tf)
    no_top_words = 10
    print(display_topics(lda, tf_feature_names, no_top_words))
    ldatf = lda.transform(tf)
    svd = pd.DataFrame(ldatf, index=User_Data_train.index.values)
    return [User_Data_train, svd]


def tfidf(User_Data_train, col='Bio'):
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, strip_accents='unicode', analyzer='word',
                                 token_pattern=r'\w{2,}', ngram_range=(1, 2), max_features=10)
    vectorizer.fit(User_Data_train[col])
    tfidf = vectorizer.transform(User_Data_train[col])
    svd = pd.DataFrame(tfidf.toarray(), index=User_Data_train.index.values)

    return [User_Data_train, svd]




text_rep_df['text_concat'] = text_rep_df['text_concat'].apply(lambda x: normalize_tweet(x))



frames = Count_Vectorizer(text_rep_df, 'text_concat')

LDA = Latent_Dirichlet_Allocation(text_rep_df,'text_concat')

TFIDT = tfidf(text_rep_df, 'text_concat')

# User_Data_train = pd.concat(TFIDT, axis=1)


text_rep_df = pd.concat([text_rep_df, frames[1], LDA[1], TFIDT[1]], axis=1)
text_rep_df.to_csv('text_rep_df.csv')
test = pd.read_csv('text_rep_df.csv')