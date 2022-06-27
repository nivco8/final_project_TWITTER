


df = df.drop(df[df.in_reply_to_user_id > 0].index)
df = df[df['referenced_tweets'].isna()]



df = df.drop(columns=['in_reply_to_user_id', 'referenced_tweets', 'public_metrics_x', 'public_metrics_y', 'id', 'Unnamed: 0', 'Unnamed: 0.1',
                      'reply_settings', 'conversation_id', 'author_id', 'geo', 'text', 'description', 'lang', 'source',
                      'Name', 'User name'])



now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d_%H%M")

print(df['URL'])

df.to_excel("bio_df.xlsx" ,sheet_name = "Sheet1", header=True, index=False)


plt.plot(df['like_count'], linestyle='--', linewidth=5)
plt.show()



sns.heatmap(df.drop('tag', 1).corr(), annot=True, cmap='coolwarm')
plt.show()



plt.figure()
df = df.drop(columns=['created_at', 'URL_bio', 'Curse_bio', 'Person_bio', 'PolarityPos_bio', 'PolarityNeg_bio',
                      'URL_tweet', 'Curse_tweet', 'Person_tweet','PolarityPos_tweet', 'PolarityNeg_tweet'])

fisher_mat = np.zeros((len(df.columns) - 1))
for col_i, col_1_name in enumerate(df.columns[:-1]):
    col_pos = df[col_1_name][df[tag][] == 1]
    col_neg = df[col_1_name][df.tag == 0]
    fisher = (np.abs(col_pos.dropna().mean() - col_neg.dropna().mean())) / (col_pos.dropna().std() + col_neg.dropna().std())
    fisher_mat[col_i] = fisher

corr_df = pd.DataFrame(fisher_mat, index=df.columns[:-1], columns=["tag"]).sort_values('tag')
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title("fisher")
plt.tight_layout()
plt.show()



sns.catplot(x="Curse_tweet", y="tag", data=df)
sns.catplot(x="URL_tweet", y="Person_tweet", hue="tag", kind="swarm", data=df)
plt.figure()
sns.scatterplot(x='URL_tweet', y='Curse_tweet', data=df, hue='tag', s=50)
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title('URL_tweet VS Curse_tweet')
plt.show()
plt.boxplot(df["Polarity_bio"])
sns.boxplot(x = 'tag', y='Subjectivity_tweet', data=df).set(title='Subjectivity score in tweets box plot')




df = df.drop(columns=['created_at'])

df = df.drop(columns=['created_at', 'URL_bio', 'Curse_bio', 'Person_bio', 'PolarityPos_bio', 'PolarityNeg_bio',
                      'URL_tweet', 'Curse_tweet', 'Person_tweet','PolarityPos_tweet', 'PolarityNeg_tweet'])
plt.figure()
pd.plotting.scatter_matrix(df,alpha=0.1,figsize=(14,8),diagonal='kde')



#----------------------------scatter----------------------------------------------------


fig = plt.figure()
df = df.drop(columns=['Polarity_bio', 'created_at', 'Subjectivity_bio', 'Hashtags_bio',
       'Mention_bio', 'Length_bio', 'Words_bio', 'Polarity_tweet',
       'Subjectivity_tweet', 'Hashtags_tweet', 'Mention_tweet', 'Length_tweet',
       'Words_tweet', 'like_count'])
corr_mat = np.zeros((len(df.columns), len(df.columns)))
for col_i, col_1_name in enumerate(df.columns):
    for col_j, col_2_name in enumerate(df.columns):
        col1 = df[col_1_name]
        col2 = df[col_2_name]
        cols = pd.concat([col1, col2], axis=1)
        corr_mat[col_i, col_j] = cols.dropna().corr().iloc[0, 1]
corr_df = pd.DataFrame(np.abs(corr_mat), index=df.columns, columns=df.columns) # .sort_values('target')
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.tight_layout()
plt.show()

df = df.drop(columns=['Curse_bio'])