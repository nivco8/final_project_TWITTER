import math
import xlrd
import os
import glob
import pandas as pd
import numpy as np
import datetime
import xlsxwriter
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from textblob import TextBlob
# import emoji
from profanity import profanity
import re
import nltk
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
# from Full_archive_search import Left_join
import copy



# set directory
os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")


CurseWords = ["fuck", "bitch", "suck", "shit", "damn", "bich", "sht", "fckin", "nigga"]
PunctuationPatterns = []
first = ["i", "me", "my", "myself", "mine"]
commercial = ['bot', 'tips', 'supplements', 'consultant', 'therapist', 'coach', 'platform', 'marketer', 'clinic']


text_concat_userId = pd.read_pickle('text_concat_userId.pickle')
users = pd.read_pickle('users.pickle')

# bio = pd.read_csv('users_author_test.csv')

users = users.drop_duplicates(subset=['id'])
Left_join = Left_join.drop_duplicates(subset=['author_id'])

Left_join['author_id'] = Left_join['author_id'].apply(str)
text_concat_userId['author_id'] = text_concat_userId['author_id'].apply(str)



bio_prep = pd.merge(text_concat_userId,
                     users,
                     left_on='author_id',
                     right_on='id',
                     how='left')

bio_prep = bio_prep.dropna(subset = ['public_metrics'])
bio_prep = bio_prep.dropna(subset = ['index'])

bio_prep.reset_index(inplace=True)




commercial_bio = []
URL_bio = []
URL_bio_count = []
Curse_bio = []
Interjections = []
Person_bio = []
Polarity_bio = []
PolarityPos_bio = []
PolarityNeg_bio = []
Subjectivity_bio = []
Hashtags_bio = []
Mention_bio = []
Length_bio = []
Words_bio = []
like_count = []
reply_count = []
quote_count = []
Source_categorical = []
followers_count = []
following_count = []
listed_count = []
tweet_count = []




def isNaN(num):
    return num!= num


# df.drop(labels='in_reply_to_user_id', axis=1)

# nltk.download()
for i in range(0, len(bio_prep["description"])):
    description = str(bio_prep["description"][i])
    #--- URL ---#
    if description.__contains__("http"):
        URL_bio.append(1)
        URL_bio_count.append(description.count('http'))
    else:
        URL_bio_count.append(0)
        URL_bio.append(0)

    # --- commercial list of words----
    if any(x in description.lower() for x in commercial):
        commercial_bio.append(1)
    else:
        commercial_bio.append(0)

    #--- Curse ---#
    if any(x in description.lower() for x in CurseWords):
        Curse_bio.append(1)
    else:
        Curse_bio.append(0)

    #--- Person & Interjections ---#
    FirstPerson_bio = False
    UH_bio = 0
    tok_text = nltk.word_tokenize(description)
    # pos = part of speech
    tagged = nltk.pos_tag(tok_text)
    for word, tag in tagged:
        # personal pronoun (hers, herself, him, himself) possessive pronoun (her, his, mine, my, our )
        if tag in ("PRP", "PRP$"):
            if word.lower() in first:
                FirstPerson_bio = True
        # UH = interjection (goodbye)
        if tag in ("UH"):
            UH_bio = UH_bio + 1
    if FirstPerson_bio:
        Person_bio.append(1)
    else:
        Person_bio.append(0)
    # Interjections.append(UH_bio)

    #--- Sentiment ---#
    analysis_bio = TextBlob(description)
    pol_bio = analysis_bio.sentiment.polarity
    Polarity_bio.append(pol_bio)
    if pol_bio>0:
        PolarityPos_bio.append(1)
        PolarityNeg_bio.append(0)
    else:
        if pol_bio<0:
            PolarityPos_bio.append(0)
            PolarityNeg_bio.append(1)
        else:
            PolarityPos_bio.append(0)
            PolarityNeg_bio.append(0)

    # source
    # if str(bio_prep["source"][i]) == 'Twitter for iPhone' or str(bio_prep["source"][i]) == 'Twitter for Android' or str(bio_prep["source"][i]) == 'Twitter for iPad':
    #     Source_categorical.append(1)
    # elif str(bio_prep["source"][i]) == 'Twitter Web App':
    #     Source_categorical.append(2)
    # else:
    #     Source_categorical.append(3)
    Subjectivity_bio.append(analysis_bio.sentiment.subjectivity)
    # --- Counters ---#
    Hashtags_bio.append(description.count("#"))
    Mention_bio.append(description.count("@"))
    Length_bio.append(len(description))
    Words_bio.append(len(re.findall(r'\S+', description)))

    # changing 'public_metrics' into a dictionary and creating feilds for each public metric
    # dict_tweet = bio_prep['public_metrics_x'][i]
    # quote_re = re.compile(r'\'')
    # dict_tweet = re.sub(quote_re, '"', dict_tweet)
    # dict_tweet = json.loads(dict_tweet)
    # retweet_count.append(dict_tweet['retweet_count'])
    # reply_count.append(dict_tweet['reply_count'])
    # like_count.append(dict_tweet['like_count'])
    # quote_count.append(dict_tweet['quote_count'])
    quote_re = re.compile(r'\'')
    dict_bio = bio_prep['public_metrics'][i]
    # print(dict_bio)
    # dict_bio = re.sub(quote_re, '"', dict_bio)
    # dict_bio = json.loads(dict_bio)
    followers_count.append(dict_bio['followers_count'])
    following_count.append(dict_bio['following_count'])
    listed_count.append(dict_bio['listed_count'])
    tweet_count.append(dict_bio['tweet_count'])






bio_prep["URL_bio"] = URL_bio
bio_prep["URL_bio_count"] = URL_bio_count
bio_prep["Curse_bio"] = Curse_bio
bio_prep["Person_bio"] = Person_bio
# df["Interjections"] = Interjections
bio_prep["Polarity_bio"] = Polarity_bio
bio_prep["PolarityPos_bio"] = PolarityPos_bio
bio_prep["PolarityNeg_bio"] = PolarityNeg_bio
bio_prep["Subjectivity_bio"] = Subjectivity_bio
bio_prep["Hashtags_bio"] = Hashtags_bio
bio_prep["Mention_bio"] = Mention_bio
bio_prep["Length_bio"] = Length_bio
bio_prep["Words_bio"] = Words_bio



bio_prep["followers_count"] = followers_count
bio_prep["following_count"] = following_count
bio_prep["tweet_count"] = tweet_count
bio_prep["listed_count"] = listed_count



bio_prep = bio_prep.drop(columns=['author_id', 'text_concat', 'Unnamed: 0', 'created_at_x', 'source',
       'public_metrics_x', 'referenced_tweets', 'lang', 'reply_settings', 'id',
       'conversation_id', 'name_y', 'in_reply_to_user_id', 'text', 'geo',
       'created_at_y', 'username', 'name', 'public_metrics_y', 'description',
       'URL_tweet','URL_tweet_count', 'Curse_tweet', 'Person_tweet', 'Polarity_tweet',
       'PolarityPos_tweet', 'PolarityNeg_tweet', 'Subjectivity_tweet',
       'Hashtags_tweet', 'Mention_tweet', 'Length_tweet', 'Words_tweet',
       'like_count'])

# no one curses in their bio:
bio_prep = bio_prep.drop(columns=['Curse_bio'])



#-----------------------------------follow rate----------------------------------------




bio_prep['follow_rate'] = bio_prep['following_count']/bio_prep['followers_count']

bio_prep.describe()

follow_rate_cond = [
    (bio_prep['follow_rate'] == 0),
    (bio_prep['follow_rate'] > 0) & (bio_prep['follow_rate'] < 0.25),
    (bio_prep['follow_rate'] >= 0.25) & (bio_prep['follow_rate'] < 0.75),
    (bio_prep['follow_rate'] >= 0.75) & (bio_prep['follow_rate'] < 79),
    (bio_prep['follow_rate'] >= 79)]

follow_rate_cat = [0,1,2,3,4]
bio_prep['follow_rate_cat'] = np.select(follow_rate_cond, follow_rate_cat)


bio_prep = bio_prep.drop(columns=['follow_rate', 'following_count', 'followers_count'])


# -------------- add tags --------------


users_to_merge = pd.read_csv('df_with_tag_no_nulls.csv')
users['id'] = users['id'].apply(str)

users_to_merge['author_id'] = users['id']
users_to_merge.drop('id', inplace=True, axis=1)

users_to_merge = users_to_merge.drop_duplicates(subset=['author_id'])

users_to_merge = users_to_merge.drop_duplicates(subset=['username'])



Left_join = pd.merge(bio_prep,
                    users_to_merge,
                     on='username',
                     how='left')

Left_join = Left_join.drop(columns=['index', 'text_concat', 'id', 'description',
       'public_metrics', 'created_at', 'name', 'description2'])

