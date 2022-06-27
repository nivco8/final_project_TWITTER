from pickle import *
#-----------loadin objects-----------------
import pandas as pd
import os
os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")

file_to_store = open("test.pickle", "wb")
pickle.dump(users_string, file_to_store)
file_to_store.close()

test = full_df.to_pickle('my_beautiful_pickle_test.pickle')
behavioral_df = behavioral_df.to_pickle('behavioral_df.pickle')
ling_full_df = lingustic_df.to_pickle('ling_full_df.pickle')
lingustic_df_AVG = lingustic_df_AVG.to_pickle('lingustic_df_AVG.pickle')
text_rep_df = text_rep_df.to_pickle('text_rep_df.pickle')
text_concat_userId = text_rep_df.to_pickle('text_concat_userId.pickle')
full_df_new_pickle = df.to_pickle('full_df_new_pickle.pickle')
ling_df = ling_df.to_pickle('ling_df.pickle')
all_features_df.to_pickle('all_features_df.pickle')
Left_join.to_pickle('bio_prep.pickle')
archive_search.to_pickle('archive_search.pickle')
new_users_taggign.to_pickle('new_users_taggign.pickle')
users_to_tag_only_accessible.to_pickle('users_to_tag_only_accessible.pickle')
full_df.to_pickle('full_df_new.pickle')



lingustic_df = pd.read_pickle('ling_df.pickle')
full_df = pd.read_pickle('full_df.pickle')
behavioral_df = pd.read_pickle('behavioral_df.pickle')
lingustic_df_AVG = pd.read_pickle('lingustic_df_AVG.pickle')
text_rep_df = pd.read_pickle('text_rep_df.pickle')
text_concat_userId = pd.read_pickle('text_concat_userId.pickle')
full_df_new_pickle = pd.read_pickle('full_df_new_pickle.pickle')
all_features_df= pd.read_pickle('all_features_df.pickle')
bio_prep= pd.read_pickle('bio_prep.pickle')
test= pd.read_pickle('bio_prep.pickle')
archive_search= pd.read_pickle('archive_search.pickle')
new_users_taggign= pd.read_pickle('new_users_taggign.pickle')
users= pd.read_pickle('users.pickle')
all_users_tagged= pd.read_pickle('all_users_tagged.pickle')




lingustic_df = lingustic_df.drop(columns=['author_id', 'text_concat'])
text_rep_df = text_rep_df.drop(columns=['author_id', 'text_concat'])

all_features_no_bio = pd.concat([behavioral_df, lingustic_df, text_rep_df], axis=1)
all_features_no_bio['author_id'] = all_features_no_bio['author_id'].apply(str)

all_features_df = pd.merge(bio_prep,
                    all_features_no_bio,
                     on='author_id',
                     how='left')


all_features_df.to_csv('all_features_df.csv')



# -------------------------- all users tagged prep ----------------------------------

archive_search= pd.read_pickle('archive_search.pickle')
all_users_tagged = pd.read_csv('all_users_tagged.csv')

archive_search = archive_search.drop_duplicates(subset=['username'])


all_users_tagged = pd.merge(all_users_tagged,
                    archive_search,
                    on='username',
                     how='left')

all_users_tagged = all_users_tagged.drop(columns=['Unnamed: 0', 'author_id_list',
       'Unnamed: 9', 'text', 'created_at_x',
       'in_reply_to_user_id', 'reply_settings',
       'referenced_tweets', 'lang', 'id_x', 'conversation_id', 'geo',
       'created_at_y', 'id_y', 'description_y', 'name_y'])

all_users_tagged.to_pickle('all_users_tagged.pickle')



# --------------------------- new files (9.6) ---------------------------

behavioral_df.to_pickle('behavioral_df_new.pickle')
text_concat_df.to_pickle('text_concat_df_new.pickle')
ling_df.to_pickle('ling_df_new.pickle')
text_rep_df.to_pickle('text_rep_df_new.pickle')
bio_prep.to_pickle('bio_prep_new.pickle')

lingustic_df = pd.read_pickle('ling_df_new.pickle')
text_rep_df = pd.read_pickle('text_rep_df_new.pickle')
behavioral_df = pd.read_pickle('behavioral_df_new.pickle')
bio_prep = pd.read_pickle('bio_prep_new.pickle')


lingustic_df = lingustic_df.drop(columns=['author_id', 'text_concat'])
text_rep_df = text_rep_df.drop(columns=['author_id', 'text_concat'])

all_features_no_bio = pd.concat([behavioral_df, lingustic_df, text_rep_df], axis=1)
all_features_no_bio['author_id'] = all_features_no_bio['author_id'].apply(str)

all_features_df = pd.merge(bio_prep,
                    all_features_no_bio,
                     on='author_id',
                     how='left')


all_features_df.to_pickle('all_features_df_new.pickle')


