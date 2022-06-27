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
import openpyxl
from users_list import users_string
# import sqlalchemy
# import pyodbc
# engine = sqlalchemy.create_engine("mssql+pyodbc://<username>:<password>@<dsnname>")
# engine = create_engine('sqlite://', echo=False)

os.environ[
    'TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAH%2BqVwEAAAAA109XqRhzWkEAIaGxKVHemrOhRug%3DokS8leyFFdRepJK9n2le1Xzirhi8Vo9I9VJyivOPhL1XqphHsB'


# set directory
os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")



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


#
# ## updated files are users_author_test
# users = pd.read_csv('df_with_tag_no_nulls.csv')
# users['author_id'] = users['author_id'].apply(str)
# users2 = pd.read_excel('users_author.xlsx')
# users2 = users.drop_duplicates(subset=['author_id'])
# users2 = users2[users2['author_id'].notna()]
# users2['author_id'] = users['author_id'].apply(str)
# users_list = users2.author_id.values.tolist()
# print(users_list)
#


all_users_tagged= pd.read_pickle('all_users_tagged.pickle')
all_users_tagged = all_users_tagged.dropna(subset = ['author_id'])
full_df = pd.DataFrame()
users_skipped = []

for user in all_users_tagged['author_id'].unique():
    print(user)
    user_df = pd.DataFrame()
    url = create_url(user, start_time, end_time, max_results)
    json_response = connect_to_endpoint(url[0], headers, url[1])
    if len(json_response) == 2 and json_response['meta']['result_count'] != 0:
        temp_df = pd.DataFrame(json_response['data'])
        user_df = user_df.append(temp_df)
        while True:
            try:
                if json_response['meta']['next_token']:
                    print(json_response['meta']['next_token'])
                    json_response = connect_to_endpoint(url[0], headers, url[1],
                                                                  pagination_token=json_response['meta']['next_token'])
                    temp_df = pd.DataFrame(json_response['data'])
                    user_df = user_df.append(temp_df)
            except:
                break
        full_df = full_df.append(user_df)
    else:
        users_skipped.append(user)
        print(str(user) + ' skipped')


full_df_new = pd.read_pickle('full_df_new.pickle')
test = full_df_new.describe()
#
#
# url = create_url('705021534067273729', start_time , end_time, 50)
# json_response = connect_to_endpoint(url[0], headers, url[1])
# print(json_response['meta']['next_token'])
# json_response = connect_to_endpoint(url[0], headers, url[1],
#                                     pagination_token='7140dibdnow9c7btw420jwai1ntjk8eumihg2ltdlj510')
#
#
#
# while True:
#     try:
#         if json_response['meta']['next_token']:
#             print(json_response['meta']['next_token'])
#             json_response = connect_to_endpoint(url[0], headers, url[1],
#                                                 next_token=json_response['meta']['next_token'])
#             temp_df = pd.DataFrame(json_response['data'])
#             user_df = user_df.append(temp_df)
#     except:
#         break
#
#
#
#
#
# full_df = full_df.append(user_df)
#
# url = create_url('LanchasterDaisy', start_time, end_time, max_results)
#
#
# i = 0
# while i < len(users_list):
#     start = users_list[i]
#     if i+100 <= len(users_list):
#         user_list_100 = ",".join(users_list[i:i+100])
#     else:
#         user_list_100 = ",".join(users_list[i:])
#     print(user_list_100)
#     url = create_url(user_list_100)
#     json_response = connect_to_endpoint(url, headers)
#     # print(json.dumps(json_response, indent=4, sort_keys=True))
#     NumberOfUsers = len(json_response['data'])
#     print(NumberOfUsers, "users were found")
#     # print(NumberOfUsers, "users were found")
#     # print(json.dumps(json_response, indent=4, sort_keys=True))
#     # export_to_excel(json_response, NumberOfUsers, start)
#     i = i + 100
#
#
# json_response = connect_to_endpoint(url, headers, params)
#
#
# test = behavioral_df.describe()
# test = all_features_df.describe()
#
#
