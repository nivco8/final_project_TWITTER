import os
import pandas as pd



# set directory
os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")

# df = pd.read_csv('full_df.csv').drop('Unnamed: 0', axis=1)
# df_check = pd.read_csv('users_author_test.csv').drop('Unnamed: 0', axis=1)
df = pd.read_pickle('full_df_new.pickle')
#count how many values each author_id have
# df_tweet = pd.DataFrame({'author_id':df['author_id'].value_counts().index, 'num_of_tweet':df['author_id'].value_counts().values})

behavioral_df = pd.DataFrame()

author_id_list = []
number_of_tweets = []
rt_to_notRT = []
rt_to_numOfTweets = []
include_to_notInclude = []
include_to_numOfTweets = []


#---------all of tweets + RT to not RT ratio + RT to all of tweets ratio

for authorId in df['author_id'].unique(): #loop for all author_id
    df_temp = df[df['author_id']==authorId]
    all_user_tweets = len(df_temp)
    text_RT = df_temp.text.str.contains(r'^[RT]').value_counts() #search text start with RT or not
    df_tmp_RT = pd.DataFrame({'RT':text_RT.index, 'value_count':text_RT.values})
    rt=0
    not_rt=0
    for idx, row in df_tmp_RT.iterrows():
        if row['RT']==True: #if true we will mark it as rt
            rt=row['value_count']
        if row['RT']==False: #if false we will mark it as not rt
            not_rt=row['value_count']
    if not_rt == 0:
        ratio = 1000
    else:
        ratio = rt/not_rt
    author_id_list.append(authorId)
    number_of_tweets.append(all_user_tweets)
    rt_to_notRT.append(ratio)
    rt_to_numOfTweets.append(rt/all_user_tweets)

#---------match to not match ratio + match to all of tweets ratio

match_list = ['diabetes', 't1d', 't2d', 'insulin'] # diabetes OR t1d OR....
for authorId in df['author_id'].unique():
    df_temp = df[df['author_id']==authorId]
    all_user_tweets = len(df_temp)
    text_match = df_temp.text.str.lower().str.contains('|'.join(match_list)).value_counts() #search text has one of word in list or not
    df_tmp_match = pd.DataFrame({'match':text_match.index.to_list(), 'value_count':text_match.values})
    match=0
    not_match=0
    for idx, row in df_tmp_match.iterrows():
        if row['match']==True:
            match=row['value_count']
        if row['match']==False:
            not_match=row['value_count']
    if not_match == 0:
        ratio = 1000
    else:
        ratio = match/not_match
    include_to_notInclude.append(ratio)
    include_to_numOfTweets.append(match/all_user_tweets)




behavioral_df['author_id'] = author_id_list
behavioral_df['number_of_tweets'] = number_of_tweets
behavioral_df['rt_to_notRT'] = rt_to_notRT
behavioral_df['rt_to_numOfTweets'] = rt_to_numOfTweets
behavioral_df['include_to_notInclude'] = include_to_notInclude
behavioral_df['include_to_numOfTweets'] = include_to_numOfTweets




test_RT = df_temp.lang.str.contains(r'en').value_counts()


#---------RT to num of tweets ratio

# for authorId in df['author_id'].unique():
#     df_temp = df[df['author_id']==authorId]
#     text_RT = df_temp.text.str.contains(r'^[RT]').value_counts() #search text start with RT or not
#     df_tmp_RT = pd.DataFrame({'RT':text_RT.index, 'value_count':text_RT.values})
#     all_user_tweets = len(df_temp)
#     rt=0
#     not_rt=0
#     for idx, row in df_tmp_RT.iterrows():
#         if row['RT']==True:
#             rt=row['value_count']
#         if row['RT']==False:
#             not_rt=row['value_count']
#     print("Ratio RT/number of tweets of {}: {}/{}".format(authorId, rt, rt+not_rt)) #ratio rt + not_rt is total tweets
#
#































#
#
#
#
#
# # users_df = pd.read_csv('users_author_test.csv')
# # users_df = pd.read_csv('full_df.csv')
#
# # counting instances
# users = pd.DataFrame(data= users_string)
#
# users_summary = users_author_test.csv
# user_df = pd.DataFrame()
#
# for user in users_list:
#
# number_of_tweets = []
# rt_to_no_rt = []
# rt_to_all = []
#
# for user in users_string:
#     num_of_rt = 0
#     num_of_not_rt = 0
#     rslt_df = full_df[full_df['author_id'] == user]
#     result = len(rslt_df[rslt_df['text'].str.contains('RT')])
#     # print(len(rslt_df[rslt_df['text'].str.contains('RT')]))
#     count_tweets = rslt_df.shape[0]
#     number_of_tweets.append(count_tweets)
#     rt_df = rslt_df[rslt_df['text'].startwith('RT')]
#
#
# users['num_of_tweets'] = number_of_tweets
# users['rt_to_no_rt'] = rt_to_no_rt
# users['rt_to_all'] = rt_to_all
#
# rt_df = full_df[full_df['text'].startwith('RT')]
#
#
# i in len(full_df[reply_settings]):
#
# test = full_df.groupby(by='author_id').count()
# print('a')
#
#
#
#
#




