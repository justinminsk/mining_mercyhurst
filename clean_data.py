import fastparquet
import datetime
import re
import pandas as pd
import numpy as np
from dateutil.parser import parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("read in file")

df = pd.read_parquet("mm.pq")

print(df.shape)

print("create and delete varibles")

df.date_created = df.date_created.apply(parse)
df.date_created = df.date_created.map(lambda x: x.replace(tzinfo=None))

df["interaction"] = df.favorites + df.retweets
df["hour"] = pd.to_numeric(df.date_created.dt.hour)
df["day"] = pd.to_numeric(df.date_created.dt.dayofyear)
df["day_of_week"] = pd.to_numeric(df.date_created.dt.dayofweek)
df["time_since"] = pd.to_numeric((datetime.datetime.now() - df.date_created).dt.total_seconds())
# get hours since seconds is too large
df["time_since"] = df["time_since"]/3600

df = df.drop(["tweet_id", "favorites", "retweets"], axis=1)
df = df.set_index("date_created")

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1,pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def pre_processing(row):
    first_process = re.sub(combined_pat, '', row)
    second_process = re.sub(www_pat, '', first_process)
    third_process = second_process.lower()
    fourth_process = neg_pattern.sub(lambda x: negations_dic[x.group()], third_process)
    result = re.sub(r'[^A-Za-z ]','',fourth_process)
    return result.strip()

print("precess tweets")

df.body = df.body.apply(pre_processing)

word_grams = TfidfVectorizer(analyzer = "word", ngram_range = (2, 4), stop_words="english")

word_vector = word_grams.fit_transform(df.body)

word_df = pd.DataFrame()

for i, col in enumerate(word_grams.get_feature_names()):
    word_df[col] = pd.Series(word_vector[:, i].toarray().ravel())

del word_df

df = df.drop(["body"], axis=1)

print("write parquet")

fastparquet.write("processed_tweets.parquet", df)

print(df.shape)
