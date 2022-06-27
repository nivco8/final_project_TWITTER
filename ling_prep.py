import math
import xlrd
import os
import glob
import pandas as pd
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






def ling_prep(lingustic_df):
    CurseWords = ["fuck", "bitch", "suck", "shit", "damn", "bich", "sht", "fckin", "nigga"]
    PunctuationPatterns = []
    first = ["i", "me", "my", "myself", "mine"]
    commercial = ['bot', 'tips', 'supplements', 'consultant', 'therapist', 'coach', 'platform', 'marketer', 'clinic']

    commercial_tweet = []
    URL_tweet = []
    URL_tweet_count = []
    Curse_tweet = []
    Interjections = []
    Person_tweet = []
    Polarity_tweet = []
    PolarityPos_tweet = []
    PolarityNeg_tweet = []
    Subjectivity_tweet = []
    Hashtags_tweet = []
    Mention_tweet = []
    Length_tweet = []
    Words_tweet = []
    retweet_count = []
    like_count = []
    reply_count = []
    quote_count = []
    Source_categorical = []



    # df.drop(labels='in_reply_to_user_id', axis=1)

    # nltk.download()
    for i in range(0, len(lingustic_df["text_concat"])):
        tweet = str(lingustic_df["text_concat"][i])
        # --- URL ---#
        if tweet.__contains__("http"):
            URL_tweet.append(1)
            URL_tweet_count.append(tweet.count('http'))
        else:
            URL_tweet_count.append(0)
            URL_tweet.append(0)

        # --- commercial list of words----
        if any(x in tweet.lower() for x in commercial):
            commercial_tweet.append(1)
        else:
            commercial_tweet.append(0)

        # --- Curse ---#
        if any(x in tweet.lower() for x in CurseWords):
            Curse_tweet.append(1)
        else:
            Curse_tweet.append(0)

        # --- Person & Interjections ---#
        FirstPerson_bio = False
        FirstPerson_tweet = False
        UH_bio = 0
        UH_tweet = 0
        # pos = part of speech
        tok_text = nltk.word_tokenize(tweet)
        tagged = nltk.pos_tag(tok_text)
        for word, tag in tagged:
            # personal pronoun (hers, herself, him, himself) possessive pronoun (her, his, mine, my, our )
            if tag in ("PRP", "PRP$"):
                if word.lower() in first:
                    FirstPerson_tweet = True
            # UH = interjection (goodbye)
            if tag in ("UH"):
                UH_tweet = UH_tweet + 1
        if FirstPerson_tweet:
            Person_tweet.append(1)
        else:
            Person_tweet.append(0)
        # Interjections.append(UH_bio)
        # Interjections.append(UH_tweet)
        # --- Sentiment ---#
        analysis_tweet = TextBlob(tweet)
        pol_tweet = analysis_tweet.sentiment.polarity
        Polarity_tweet.append(pol_tweet)
        if pol_tweet > 0:
            PolarityPos_tweet.append(1)
            PolarityNeg_tweet.append(0)
        else:
            if pol_tweet < 0:
                PolarityPos_tweet.append(0)
                PolarityNeg_tweet.append(1)
            else:
                PolarityPos_tweet.append(0)
                PolarityNeg_tweet.append(0)
        # source
        # if str(lingustic_df["source"][i]) == 'Twitter for iPhone' or str(lingustic_df["source"][i]) == 'Twitter for Android' or str(
        #         lingustic_df["source"][i]) == 'Twitter for iPad':
        #     Source_categorical.append(1)
        # elif str(lingustic_df["source"][i]) == 'Twitter Web App':
        #     Source_categorical.append(2)
        # else:
        #     Source_categorical.append(3)
        Subjectivity_tweet.append(analysis_tweet.sentiment.subjectivity)
        # --- Counters ---#
        Hashtags_tweet.append(tweet.count("#"))
        Mention_tweet.append(tweet.count("@"))
        Length_tweet.append(len(tweet))
        Words_tweet.append(len(re.findall(r'\S+', tweet)))
        print(i)
        # changing 'public_metrics' into a dictionary and creating feilds for each public metric
        # dict_tweet = lingustic_df['public_metrics'][i]
        # quote_re = re.compile(r'\'')
        # dict_tweet = re.sub(quote_re, '"', dict_tweet)
        # dict_tweet = json.loads(dict_tweet)
        # retweet_count.append(dict_tweet['retweet_count'])
        # reply_count.append(dict_tweet['reply_count'])
        # like_count.append(dict_tweet['like_count'])
        # quote_count.append(dict_tweet['quote_count'])



    lingustic_df["URL_tweet"] = URL_tweet
    lingustic_df["URL_tweet_count"] = URL_tweet_count
    lingustic_df["Curse_tweet"] = Curse_tweet
    lingustic_df["Person_tweet"] = Person_tweet
    # df["Interjections"] = Interjections
    lingustic_df["Polarity_tweet"] = Polarity_tweet
    lingustic_df["PolarityPos_tweet"] = PolarityPos_tweet
    lingustic_df["PolarityNeg_tweet"] = PolarityNeg_tweet
    lingustic_df["Subjectivity_tweet"] = Subjectivity_tweet
    lingustic_df["Hashtags_tweet"] = Hashtags_tweet
    lingustic_df["Mention_tweet"] = Mention_tweet
    lingustic_df["Length_tweet"] = Length_tweet
    lingustic_df["Words_tweet"] = Words_tweet
    # lingustic_df["source_categorical"] = Source_categorical
    lingustic_df["commercial_tweet"] = commercial_tweet
    # lingustic_df["like_count"] = like_count



    return lingustic_df


# --------------------  avaraging values for each user -------------------- #

text_concat_userId = pd.read_pickle('text_concat_userId.pickle')


ling_df = ling_prep(text_concat_userId)




lingustic_df_AVG = pd.DataFrame()

author_id_list = []
number_of_tweets = []
Commercial_tweet_AVG = []
URL_tweet_AVG = []
URL_tweet_count_AVG = []
Curse_tweet_AVG = []
Interjections_AVG = []
Person_tweet_AVG = []
Polarity_tweet_AVG = []
Subjectivity_tweet_AVG = []
Hashtags_tweet_AVG = []
Mention_tweet_AVG = []
Length_tweet_AVG = []
Words_tweet_AVG = []
Source_categorical_MAX = []
# retweet_count_AVG = []
# like_count_AVG = []
# reply_count_AVG = []
# quote_count_AVG = []



for authorId in lingustic_df['author_id'].unique():
    df_temp = lingustic_df[lingustic_df['author_id']==authorId]
    all_user_tweets = len(df_temp)

    #---source categorical
    dict = df_temp['source_categorical'].value_counts()
    source_counting_df = pd.DataFrame({'source': dict.index, 'value_count': dict.values})

    max_count = 0
    index = 0

    for idx, row in source_counting_df.iterrows():
        print(row['value_count'])
        if row['value_count'] > max_count:  # if true we will mark it as rt
            max_count = row['value_count']
            index = row['source']
            print('row[\'source\']' + str(row['source']))
            print('row[\'value_count\']' + str(row['value_count']))

    Source_categorical_MAX.append(index)

    #---words tweet
    sum = df_temp['Words_tweet'].sum()
    avg = sum/all_user_tweets
    Words_tweet_AVG.append(avg)

    #---length tweet
    sum = df_temp['Length_tweet'].sum()
    avg = sum/all_user_tweets
    Length_tweet_AVG.append(avg)


    #---mention tweet
    sum = df_temp['Mention_tweet'].sum()
    avg = sum/all_user_tweets
    Mention_tweet_AVG.append(avg)

    #---hashtags tweet
    sum = df_temp['Hashtags_tweet'].sum()
    avg = sum/all_user_tweets
    Hashtags_tweet_AVG.append(avg)

    #---Subjectivity tweet
    sum = df_temp['Subjectivity_tweet'].sum()
    avg = sum/all_user_tweets
    Subjectivity_tweet_AVG.append(avg)

    # ---polarity tweet
    sum = df_temp['Polarity_tweet'].sum()
    avg = sum / all_user_tweets
    if avg > 0:
        Polarity_tweet_AVG.append(1)
    else:
        Polarity_tweet_AVG.append(0)

    # ---commercial tweet
    sum = df_temp['commercial_tweet'].sum()
    avg = sum / all_user_tweets
    if avg > 0.5:
        Commercial_tweet_AVG.append(1)
    else:
        Commercial_tweet_AVG.append(0)

    # ---person tweet
    sum = df_temp['Person_tweet'].sum()
    avg = sum / all_user_tweets
    if avg > 0.5:
        Person_tweet_AVG.append(1)
    else:
        Person_tweet_AVG.append(0)

    # ---curse tweet
    sum = df_temp['Curse_tweet'].sum()
    avg = sum / all_user_tweets
    if avg > 0.5:
        Curse_tweet_AVG.append(1)
    else:
        Curse_tweet_AVG.append(0)

    # ---url tweet
    sum = df_temp['URL_tweet'].sum()
    avg = sum / all_user_tweets
    if avg > 0.5:
        URL_tweet_AVG.append(1)
    else:
        URL_tweet_AVG.append(0)

    # ---url count tweet
    sum = df_temp['URL_tweet_count'].sum()
    avg = sum / all_user_tweets
    URL_tweet_count_AVG.append(avg)

    author_id_list.append(authorId)



lingustic_df_AVG['author_id'] = author_id_list
lingustic_df_AVG['commercial_tweet_AVG'] = Commercial_tweet_AVG
lingustic_df_AVG['URL_tweet_AVG'] = URL_tweet_AVG
lingustic_df_AVG['URL_tweet_count_AVG'] = URL_tweet_count_AVG
lingustic_df_AVG['Curse_tweet_AVG'] = Curse_tweet_AVG
lingustic_df_AVG['Polarity_tweet_AVG'] = Polarity_tweet_AVG
lingustic_df_AVG['Subjectivity_tweet_AVG'] = Subjectivity_tweet_AVG
lingustic_df_AVG['Hashtags_tweet_AVG'] = Hashtags_tweet_AVG
lingustic_df_AVG['Mention_tweet_AVG'] = Mention_tweet_AVG
lingustic_df_AVG['Words_tweet_AVG'] = Words_tweet_AVG
lingustic_df_AVG['Source_categorical_MAX'] = Source_categorical_MAX






# ------------------------------------- categorizaiton ------------------------------------- #



lingustic_df_AVG.describe()

follow_rate_cond = [
    (lingustic_df_AVG['follow_rate'] == 0),
    (lingustic_df_AVG['follow_rate'] > 0) & (lingustic_df_AVG['follow_rate'] < 0.11),
    (lingustic_df_AVG['follow_rate'] >= 0.11) & (lingustic_df_AVG['follow_rate'] < 0.81),
    (lingustic_df_AVG['follow_rate'] >= 0.81) & (lingustic_df_AVG['follow_rate'] < 2.18),
    (lingustic_df_AVG['follow_rate'] >= 2.18)]
follow_rate_cat = [0,1,2,3,4]
df_train['follow_rate_cat'] = np.select(follow_rate_cond, follow_rate_cat)
