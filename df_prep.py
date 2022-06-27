import xlrd
import os
import glob
import pandas as pd
import datetime
import xlsxwriter
from textblob import TextBlob
# import emoji
from profanity import profanity
import re
import nltk
import json

# set directory
os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")
excel_names = glob.glob("df.xlsx")
df = pd.read_csv("df.csv")

CurseWords = ["fuck", "bitch", "suck", "shit", "damn"]
PunctuationPatterns = []
first = ["i", "me", "my", "myself", "mine"]
care_giver = ['bot', 'Tips', 'Supplements', 'consultant', 'Therapist', 'coach']


URL = []
URL_count = []
Curse = []
Interjections = []
Person = []
Polarity = []
PolarityPos = []
PolarityNeg = []
Subjectivity = []
Hashtags = []
Mention = []
Length = []
Words = []
Emoji = []
Source_categorical = []
is_english = []
retweet_count = []
like_count = []
reply_count = []
quote_count = []



for i in range(0, len(df["text"])):
    text = str(df["text"][i])
    #--- URL_tweet ---#
    if text.__contains__("http"):
        URL.append(1)
        URL_count.append(text.count('http'))
    else:
        URL_count.append(0)
        URL.append(0)
    #--- Curse ---#
    if any(x in text.lower() for x in CurseWords):
        Curse.append(1)
    else:
        Curse.append(0)
    #--- Person ---#
    #if any(x in text.lower() for x in first):
    #    Person.append(1)
    #else:
    #    Person.append(0)
    # --- Disfluencies ---#
    #if any(x in text.lower() for x in DisfluencyWords):
    #    Disfluencies.append(1)
    #else:
    #    Disfluencies.append(0)
    #--- Person & Interjections ---#
    FirstPerson = False
    UH = 0
    tok_text = nltk.word_tokenize(text)
    # pos = part of speech
    tagged = nltk.pos_tag(tok_text)
    for word, tag in tagged:
        # personal pronoun (hers, herself, him, himself) possessive pronoun (her, his, mine, my, our )
        if tag in ("PRP", "PRP$"):
            if word.lower() in first:
                FirstPerson = True
        # UH = interjection (goodbye)
        if tag in ("UH"):
            UH = UH + 1
    if FirstPerson:
        Person.append(1)
    else:
        Person.append(0)
    Interjections.append(UH)
    #--- Sentiment ---#
    analysis = TextBlob(text)
    pol = analysis.sentiment.polarity
    Polarity.append(pol)
    if pol>0:
        PolarityPos.append(1)
        PolarityNeg.append(0)
    else:
        if pol<0:
            PolarityPos.append(0)
            PolarityNeg.append(1)
        else:
            PolarityPos.append(0)
            PolarityNeg.append(0)
    Subjectivity.append(analysis.sentiment.subjectivity)
    # --- Counters ---#
    Hashtags.append(text.count("#"))
    Mention.append(text.count("@"))
    Length.append(len(text))
    Words.append(len(re.findall(r'\S+', text)))
    if str(df["source"][i]) == 'Twitter for iPhone' or str(df["source"][i]) == 'Twitter for Android' or str(df["source"][i]) == 'Twitter for iPad':
        Source_categorical.append(1)
    elif str(df["source"][i]) == 'Twitter Web App':
        Source_categorical.append(2)
    else:
        Source_categorical.append(3)
    if str(df["lang"][i]) == 'en':
        is_english.append(1)
    else:
        is_english.append(0)

    # changing 'public_metrics' into a dictionary and creating feilds for each public metric
    dict = df['public_metrics'][i]
    quote_re = re.compile(r'\'')
    dict = re.sub(quote_re, '"', dict)
    dict = json.loads(dict)
    retweet_count.append(dict['retweet_count'])
    reply_count.append(dict['reply_count'])
    like_count.append(dict['like_count'])
    quote_count.append(dict['quote_count'])


