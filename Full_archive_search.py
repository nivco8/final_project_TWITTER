import requests
import os
import json
import pandas as pd
import csv
import datetime
import dateutil.parser
import unicodedata
import time
import copy

os.environ[
    'TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAH%2BqVwEAAAAA109XqRhzWkEAIaGxKVHemrOhRug%3DokS8leyFFdRepJK9n2le1Xzirhi8Vo9I9VJyivOPhL1XqphHsB'


def auth():
    return os.getenv('TOKEN')


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def create_url(keyword, start_date, end_date, max_results=10):
    search_url = "https://api.twitter.com/2/tweets/search/all"  # Change to the endpoint you want to collect data from

    # change params based on the endpoint you are using
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id',
                    # in_reply_to_user_id,geo.place_id, attachments.poll_ids'
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)


def connect_to_endpoint(url, headers, params, next_token=None):
    params['next_token'] = next_token  # params object received from create_url function
    response = requests.request("GET", url, headers=headers, params=params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json(), response


# Inputs for the request
bearer_token = auth()
headers = create_headers(bearer_token)
start_time = "2022-03-20T00:00:00.000Z"
end_time = "2022-03-21T00:00:00.000Z"
max_results = 500

# first request to set a dataframe
url = create_url('DIABETES', start_time, end_time, max_results)
json_response, response = connect_to_endpoint(url[0], headers, url[1])
df = pd.DataFrame(json_response['data'])
users = pd.DataFrame(json_response['includes']['users'])

for keyword in ["DIABETES", "T1D", "T2D", "INSULIN"]:
    url = create_url(keyword, start_time, end_time, max_results)
    # request
    json_response, response = connect_to_endpoint(url[0], headers, url[1])
    if keyword != "DIABETES":
        df_temp = pd.DataFrame(json_response['data'])
        users_temp = pd.DataFrame(json_response['includes']['users'])
        df = df.append(df_temp)
        users = users.append(users_temp)
        ## to retrieve all of the results query
    while True:
        try:
            if json_response['meta']['next_token']:
                json_response, response = connect_to_endpoint(url[0], headers, url[1],
                                                              next_token=json_response['meta']['next_token'])
                df_temp = pd.DataFrame(json_response['data'])
                users_temp = pd.DataFrame(json_response['includes']['users'])
                df = df.append(df_temp)
                users = users.append(users_temp)
        except:
            break

# df_joined = df.merge(right = users,how='inner', left_on='author_id', right_on='id')
# df_reduced= df.sample(frac=0.01, random_state = 1)


# set directory
os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")
os.chdir("C:\\Users\\Niv\\PycharmProjects\\final_project\\Final_project\\pickles")
archive_search= pd.read_pickle('archive_search.pickle')


# ## export to excel files
# df.to_excel("df4.xlsx", header=True, index=False)
# users.to_excel("users4.xlsx", header=True, index=False)


users_to_merge = pd.read_csv('df_with_tag_no_nulls.csv')
users['id'] = users['id'].apply(str)

users_to_merge['author_id'] = users['id']
users_to_merge.drop('id', inplace=True, axis=1)

users_to_merge = users_to_merge.drop_duplicates(subset=['author_id'])

users = users.drop_duplicates(subset=['id'])

Left_join = pd.merge(df,
                     users,
                     left_on='author_id',
                     right_on='id',
                     how='left')

Left_join.to_excel("Left_join2.xlsx", header=True, index=False)
users_list = Left_join.id_y.values.tolist()

archive_search = pd.merge(df,
                          users,
                          left_on='author_id',
                          right_on='id',
                          how='left')

archive_search.to_pickle('archive_search.pickle')
users.to_pickle('users.pickle')



# -------------------------------- creating dataframe for tagging --------------------------------


def isNaN(num):
    return num != num


Left_join = pd.merge(df,
                     users,
                     left_on='author_id',
                     right_on='id',
                     how='left')

author_id_list = []
number_of_tweets = []
tweets = []
description = []
user_name = []

new_users_taggign = pd.DataFrame()

for authorId in Left_join['author_id'].unique():  # loop for all author_id
    df_temp = Left_join[Left_join['author_id'] == authorId]
    all_user_tweets = len(df_temp)

    user_tweets = []

    for idx, row in df_temp.iterrows():
        # filtering RT QT and
        if isNaN(row['referenced_tweets']) and row['lang'] == 'en':
            user_tweets.append(row['text'])
    if len(user_tweets) != 0:
        tweets.append(user_tweets)
    else:
        tweets.append(None)
    description.append(row['description'])
    description.append(row['description'])
    author_id_list.append(authorId)
    number_of_tweets.append(all_user_tweets)

new_users_taggign['author_id_list'] = author_id_list
new_users_taggign['number_of_tweets'] = number_of_tweets
new_users_taggign['tweets'] = tweets
new_users_taggign['description'] = description

new_users_taggign = new_users_taggign.dropna()

# -------------------------------- take users that we didn't tag yet --------------------------------

new_users_taggign= pd.read_pickle('new_users_taggign.pickle')
all_features_df = pd.read_pickle('all_features_df.pickle')

already_tagged_users = all_features_df.drop(columns=['Polarity_bio'])
already_tagged_users = all_features_df.drop(columns=['URL_bio', 'URL_bio_count',
                                                     'Curse_bio', 'Person_bio',
                                                     'Polarity_bio', 'PolarityPos_bio',
                                                     'PolarityNeg_bio', 'Subjectivity_bio',
                                                     'Hashtags_bio', 'Mention_bio',
                                                     'Length_bio', 'Words_bio',
                                                     'tweet_count', 'listed_count',
                                                     'follow_rate_cat',
                                                     'number_of_tweets', 'rt_to_notRT',
                                                     'rt_to_numOfTweets', 'include_to_notInclude',
                                                     'include_to_numOfTweets', 'URL_tweet',
                                                     'URL_tweet_count', 'Curse_tweet',
                                                     'Person_tweet', 'Polarity_tweet',
                                                     'PolarityPos_tweet', 'PolarityNeg_tweet',
                                                     'Subjectivity_tweet', 'Hashtags_tweet',
                                                     'Mention_tweet', 'Length_tweet',
                                                     'Words_tweet', 'commercial_tweet',
                                                     0, 1,
                                                     2, 3,
                                                     4, 5,
                                                     6, 7,
                                                     8, 9,
                                                     0, 1,
                                                     2, 3,
                                                     4, 5,
                                                     6, 7,
                                                     8, 9,
                                                     0, 1,
                                                     2, 3,
                                                     4, 5,
                                                     6, 7,
                                                     8, 9], axis=1)


users_to_tag = pd.merge(new_users_taggign,
                     already_tagged_users,
                     left_on='author_id_list',
                     right_on='author_id',
                     how='left')

from extendeed__user_search import *


def auth():
    return os.getenv('TOKEN')


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers



def create_url(user,start_date , end_date, max_results):
    query_params = {'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics',
                    'pagination_token': {}}
    # url = "https://api.twitter.com/2/users/by?usernames=" + user + "&user.fields=description,public_metrics"
    url = "https://api.twitter.com/2/users/" + user + "/tweets"
    return (url, query_params)



def connect_to_endpoint(url, headers, params, pagination_token=None):
    params['pagination_token'] = pagination_token  # params object received from create_url function
    response = requests.request("GET", url,  headers=headers, params = params)
    # print(response.status_code)
    # print(response.text)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()



bearer_token = auth()
headers = create_headers(bearer_token)
start_time = "2022-03-21T00:00:00.000Z"
end_time = "2022-03-28T00:00:00.000Z"
max_results = 100

accessible = []

users_to_tag['author_id_list'] = users_to_tag['author_id_list'].apply(str)
for user in users_to_tag['author_id_list'].to_list():
    url = create_url(user, start_time, end_time, max_results)
    json_response = connect_to_endpoint(url[0], headers, url[1])
    if len(json_response) == 2 and json_response['meta']['result_count'] != 0:
        accessible.append(True)
        print('accessible')
    else:
        accessible.append(None)
        print('not accessible')

users_to_tag['accessible'] = accessible

users_to_tag_only_accessible = users_to_tag.dropna(subset=['accessible'])
users_to_tag_only_accessible.describe()
users_to_tag.describe()


test = pd.read_pickle('users.pickle')