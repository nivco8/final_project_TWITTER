
commercial_bio = []
URL_bio = []
URL_bio_count = []
Curse_bio = []
Person_bio = []
Polarity_bio = []
PolarityPos_bio = []
PolarityNeg_bio = []
Subjectivity_bio = []
Hashtags_bio = []
Mention_bio = []
Length_bio = []
Words_bio = []
followers_count = []
following_count = []
tweet_count = []
listed_count = []




# df.drop(labels='in_reply_to_user_id', axis=1)

# nltk.download()
for i in range(0, len(df["text"])):
    description = str(df["description"][i])
    tweet = str(df["text"][i])
    #--- URL ---#
    if description.__contains__("http"):
        URL_bio.append(1)
        URL_bio_count.append(description.count('http'))
    else:
        URL_bio_count.append(0)
        URL_bio.append(0)
    if tweet.__contains__("http"):
        URL_tweet.append(1)
        URL_tweet_count.append(tweet.count('http'))
    else:
        URL_tweet_count.append(0)
        URL_tweet.append(0)

    # --- commercial list of words----
    if any(x in description.lower() for x in commercial):
        commercial_bio.append(1)
    else:
        commercial_bio.append(0)
    if any(x in tweet.lower() for x in commercial):
        commercial_tweet.append(1)
    else:
        commercial_tweet.append(0)

    #--- Curse ---#
    if any(x in description.lower() for x in CurseWords):
        Curse_bio.append(1)
    else:
        Curse_bio.append(0)
    if any(x in tweet.lower() for x in CurseWords):
        Curse_tweet.append(1)
    else:
        Curse_tweet.append(0)

    #--- Person & Interjections ---#
    FirstPerson_bio = False
    FirstPerson_tweet = False
    UH_bio = 0
    UH_tweet = 0
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
    #--- Sentiment ---#
    analysis_bio = TextBlob(description)
    pol_bio = analysis_bio.sentiment.polarity
    Polarity_bio.append(pol_bio)
    analysis_tweet = TextBlob(tweet)
    pol_tweet = analysis_tweet.sentiment.polarity
    Polarity_tweet.append(pol_tweet)
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
    if pol_tweet>0:
        PolarityPos_tweet.append(1)
        PolarityNeg_tweet.append(0)
    else:
        if pol_bio<0:
            PolarityPos_tweet.append(0)
            PolarityNeg_tweet.append(1)
        else:
            PolarityPos_tweet.append(0)
            PolarityNeg_tweet.append(0)
    # source
    if str(df["source"][i]) == 'Twitter for iPhone' or str(df["source"][i]) == 'Twitter for Android' or str(df["source"][i]) == 'Twitter for iPad':
        Source_categorical.append(1)
    elif str(df["source"][i]) == 'Twitter Web App':
        Source_categorical.append(2)
    else:
        Source_categorical.append(3)
    Subjectivity_bio.append(analysis_bio.sentiment.subjectivity)
    Subjectivity_tweet.append(analysis_tweet.sentiment.subjectivity)
    # --- Counters ---#
    Hashtags_bio.append(description.count("#"))
    Mention_bio.append(description.count("@"))
    Length_bio.append(len(description))
    Words_bio.append(len(re.findall(r'\S+', description)))
    Hashtags_tweet.append(tweet.count("#"))
    Mention_tweet.append(tweet.count("@"))
    Length_tweet.append(len(tweet))
    Words_tweet.append(len(re.findall(r'\S+', tweet)))
    # changing 'public_metrics' into a dictionary and creating feilds for each public metric
    dict_tweet = df['public_metrics_x'][i]
    quote_re = re.compile(r'\'')
    dict_tweet = re.sub(quote_re, '"', dict_tweet)
    dict_tweet = json.loads(dict_tweet)
    retweet_count.append(dict_tweet['retweet_count'])
    reply_count.append(dict_tweet['reply_count'])
    like_count.append(dict_tweet['like_count'])
    quote_count.append(dict_tweet['quote_count'])
    dict_bio = df['public_metrics_y'][i]
    dict_bio = re.sub(quote_re, '"', dict_bio)
    dict_bio = json.loads(dict_bio)
    followers_count.append(dict_bio['followers_count'])
    following_count.append(dict_bio['following_count'])
    tweet_count.append(dict_bio['tweet_count'])
    listed_count.append(dict_bio['listed_count'])




df["URL_bio"] = URL_bio
df["URL_bio_count"] = URL_bio_count
df["Curse_bio"] = Curse_bio
df["Person_bio"] = Person_bio
# df["Interjections"] = Interjections
df["Polarity_bio"] = Polarity_bio
df["PolarityPos_bio"] = PolarityPos_bio
df["PolarityNeg_bio"] = PolarityNeg_bio
df["Subjectivity_bio"] = Subjectivity_bio
df["Hashtags_bio"] = Hashtags_bio
df["Mention_bio"] = Mention_bio
df["Length_bio"] = Length_bio
df["Words_bio"] = Words_bio

df["URL_tweet"] = URL_tweet
df["URL_tweet_count"] = URL_tweet_count
df["Curse_tweet"] = Curse_tweet
df["Person_tweet"] = Person_tweet
# df["Interjections"] = Interjections
df["Polarity_tweet"] = Polarity_tweet
df["PolarityPos_tweet"] = PolarityPos_tweet
df["PolarityNeg_tweet"] = PolarityNeg_tweet
df["Subjectivity_tweet"] = Subjectivity_tweet
df["Hashtags_tweet"] = Hashtags_tweet
df["Mention_tweet"] = Mention_tweet
df["Length_tweet"] = Length_tweet
df["Words_tweet"] = Words_tweet
df["like_count"] = like_count
df["followers_count"] = followers_count
df["following_count"] = following_count
df["tweet_count"] = tweet_count
df["listed_count"] = listed_count
