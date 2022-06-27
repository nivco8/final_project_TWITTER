import requests
import os
import json
import pandas as pd


os.environ[
    'TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAH%2BqVwEAAAAA109XqRhzWkEAIaGxKVHemrOhRug%3DokS8leyFFdRepJK9n2le1Xzirhi8Vo9I9VJyivOPhL1XqphHsB'


# set directory
os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")



def auth():
    return os.getenv('TOKEN')


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def create_url_followers(user_id):
    # Replace with user ID below
    user_id = user_id
    return "https://api.twitter.com/2/users/{}/followers".format(user_id)


def create_url_followings(user_id):
    # Replace with user ID below
    user_id = user_id
    return "https://api.twitter.com/2/users/{}/following".format(user_id)


def get_params():
    return {"user.fields": "created_at,description,public_metrics"}


def connect_to_endpoint(url, headers, params, next_token=None):
    params['next_cursor'] = next_token  # params object received from create_url function
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



df = pd.read_csv('full_df.csv').drop('Unnamed: 0', axis=1)
bearer_token = auth()
params = get_params()
headers = create_headers(bearer_token)

for authorId in df['author_id'].unique(): #loop for all author_id
    url_followers = create_url_followers(authorId)
    url_followings = create_url_followings(authorId)
    json_response_followers = connect_to_endpoint(url_followers, headers, params)
    json_response_followings = connect_to_endpoint(url_followings, headers, params)
    followers = pd.DataFrame(json_response_followers['data'])
    following = pd.DataFrame(json_response_followings['data'])

temp_df = pd.DataFrame()


url_followers = create_url_followers(359393647)




for authorId in df['author_id'].unique():  # loop for all author_id
    print(authorId)
    following_df = pd.DataFrame()
    followers_df = pd.DataFrame()
    user_df = pd.DataFrame()
    url_followers = create_url_followers(authorId)
    url_followings = create_url_followings(authorId)
    json_response_followers = connect_to_endpoint(url_followers, headers, params)
    if len(json_response_followers) == 2 and json_response_followers['meta']['result_count'] != 0:
        temp_df = pd.DataFrame(json_response_followers['data'])
        user_df = user_df.append(temp_df)
        followers_df = followers_df.append(temp_df)
        while True:
            try:
                if json_response_followers['meta']['next_token']:
                    print(json_response_followers['meta']['next_token'])
                    json_response_followers = connect_to_endpoint(url_followers, headers, params, next_token=json_response_followers['meta']['next_token'])
                    temp_df = pd.DataFrame(json_response_followers['data'])
                    user_df = user_df.append(temp_df)
            except Exception:
                print(Exception.args)
                break
    followers_df = followers_df.append(user_df)
    json_response_followings = connect_to_endpoint(url_followings, headers, params)
    followers = pd.DataFrame(json_response_followers['data'])
    following = pd.DataFrame(json_response_followings['data'])



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



