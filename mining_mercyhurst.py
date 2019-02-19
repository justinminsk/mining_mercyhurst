import json
import tweepy
import pandas
from fastparquet import write

# CONSUMER_KEY = 'GoESGoge59AzCGKfrR044B8sn'
# CONSUMER_SECRET = ''
# OAUTH_TOKEN = '742052488761749504-dmZVKlX4JZO07PwagCd4nj8r2bjinjX'
# OAUTH_TOKEN_SECRET = ''
#
# auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
# auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
# api = tweepy.API(auth)
#
# # Setting username of interest and creating data structure
# user_name = 'mercyhurstu'
# all_tweets = []

# Grabbing maximum amount of tweets possible (3200)
# for tweet in tweepy.Cursor(api.user_timeline, id=user_name).items(10):
#     df.append(tweet)
#

# # Exporting all collected tweets to json
# with open('mm.json', 'w') as outfile:
#     json.dump(all_tweets, outfile)

pd_list = []
with open('mm.json') as file:
    tweets = json.load(file)
    for x in tweets:
        tweet_id = x['id']
        text = x['text']
        date = x['created_at']
        favs = x['favorite_count']
        rts = x['retweet_count']
        pd_list.append({'tweet_id': str(tweet_id),
                        'body': str(text),
                        'favorites': int(favs),
                        'retweets': int(rts),
                        'date_created': date
                        })
        df = pandas.DataFrame(pd_list, columns=['tweet_id', 'body', 'favorites', 'retweets', 'date_created'])
print(df)

write('mm.pq', df)
open()
