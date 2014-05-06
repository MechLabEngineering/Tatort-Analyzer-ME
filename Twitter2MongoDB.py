# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Twitter 2 MongoDB

# <markdowncell>

# Thanks to [this great Tutorial](http://www.danielforsyth.me/analyzing-a-nhl-playoff-game-with-twitter/), we made it for German #Tatort

# <markdowncell>

# The following code takes all tweets with the keyword 'Tatort' and sends the time it was created, text of the tweet, location (if available), and the source of the tweet to a local [MongoDB](http://docs.mongodb.org/manual/installation/) database. 

# <codecell>

import tweepy
import sys
import pymongo
import datetime

# <codecell>

hashtag = 'tatort'

# <codecell>

date = datetime.datetime.now()

# <markdowncell>

# Developer Authentification

# <codecell>

# Get yours at https://dev.twitter.com//
consumer_key=""
consumer_secret=""

access_token=""
access_token_secret=""

# <codecell>

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# <codecell>

class CustomStreamListener(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api
        
        super(tweepy.StreamListener, self).__init__()
        
        self.db = pymongo.MongoClient().TatortTweets

    def on_status(self, status):
        print status.text , "\n"

        data ={}
        # Tweet
        data['text'] = status.text
        
        # Metadata
        data['created_at'] = status.created_at
        data['replyto'] = status.in_reply_to_screen_name
        data['geo'] = status.geo
        data['source'] = status.source
        
        # Userdata
        data['user'] = status.user.screen_name
        data['userfriends'] = status.user.friends_count
        data['follower'] = status.user.followers_count
        

        self.db.TatortTweets.insert(data)

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

# <codecell>

print('Listening to Twitter for #%s...' % hashtag)

# <codecell>

sapi = tweepy.streaming.Stream(auth, CustomStreamListener(api))
sapi.filter(track=[hashtag])

# <codecell>


# <codecell>


