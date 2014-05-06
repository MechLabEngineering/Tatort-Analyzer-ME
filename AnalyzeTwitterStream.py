# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Analyse Twitter Stream for German #Tatort ("Am Ende des Flurs") on 04.05.2014

# <markdowncell>

# ![](http://www.daserste.de/unterhaltung/krimi/tatort/specials/die-kommissare-leitmayr-und-batic-102~_v-banner316_c9e2d0.jpg)
# **»I hob scho immer Frauen mögn, wo ma übern Zaun steigen muass.«**

# <headingcell level=3>

# OK, lets go...

# <markdowncell>

# Hopefully, you have a local MongoDB up and running with all Tweets in it.
# 
# You can start a Mongo Deamon with the Tweets in it with
# 
# ```
# mongod --dbpath db
# ```

# <markdowncell>

# First, import stuff we need.

# <codecell>

import pandas as pd
from pandas.tseries.resample import TimeGrouper
from pandas.tseries.offsets import DateOffset
from pymongo import MongoClient
import matplotlib.pyplot as plt
from datetime import datetime
%pylab inline --no-import-all

# <markdowncell>

# Define functions to read the tweets out of the database. Thanks to [this](http://stackoverflow.com/a/16255680), I made this. 

# <codecell>

def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)


    return conn[db]

# <codecell>

def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df

# <headingcell level=1>

# Connect and get the Data out of the DB

# <codecell>

hashtag = 'Tatort'

# <codecell>

dbname = 'TatortTweets'
collection = 'TatortTweets'
tweets = read_mongo(dbname, collection)

# <codecell>

tweets.set_index('created_at', inplace=True)
tweets.index = tweets.index.tz_localize('GMT').tz_convert('CET')
tweets.index.name = 'Zeit'
tweets.index

# <headingcell level=3>

# Show something

# <codecell>

tweets.tail(10)

# <headingcell level=2>

# Tweets per Minute

# <codecell>

tweetsperminute = tweets['text'].resample('1t', how='count')

# <codecell>

plt.figure(figsize=(14,4))
tweetsperminute.plot()
plt.ylabel('#%s Tweets per Minute' % hashtag)

# <markdowncell>

# What do you think, when it started? :)

# <headingcell level=1>

# Define a Time Range to Analyze

# <codecell>

fr = '201405042000'
to = '201405042200'

# <headingcell level=3>

# cut the data

# <codecell>

tweets = tweets[fr:to]

# <headingcell level=1>

# Localisation

# <codecell>

from IPython.display import HTML
import folium

# <markdowncell>

# Thanks to [this](http://nbviewer.ipython.org/gist/bburky/7763555/folium-ipython.ipynb)

# <codecell>

def inline_map(map):
    """
    Embeds the HTML source of the map directly into the IPython notebook.
    
    This method will not work if the map depends on any files (json data). Also this uses
    the HTML5 srcdoc attribute, which may not be supported in all browsers.
    """
    map._build_map()
    return HTML('<iframe srcdoc="{srcdoc}" style="width: 100%; height: 510px; border: none"></iframe>'.format(srcdoc=map.HTML.replace('"', '&quot;')))

def embed_map(map, path="map.html"):
    """
    Embeds a linked iframe to the map into the IPython notebook.
    
    Note: this method will not capture the source of the map into the notebook.
    This method should work for all maps (as long as they use relative urls).
    """
    map.create_map(path=path)
    return HTML('<iframe src="files/{path}" style="width: 100%; height: 510px; border: none"></iframe>'.format(path=path))

# <codecell>

locations = [l for l in tweets.geo.values if l!=None]

# <codecell>

map = folium.Map(location=[50, 10], zoom_start=5, tiles='Stamen Toner')
for marker in locations:
    map.simple_marker(marker.values()[1])
embed_map(map)

# <headingcell level=1>

# Text Processing with the Natural Language Toolkit

# <markdowncell>

# ![](http://covers.oreilly.com/images/9780596516499/cat.gif)
# That great Book covers almost everything shown here:
# 
# [Natural Language Processing with Python](http://www.nltk.org/book/)
# by Steven Bird, Ewan Klein, and Edward Loper
# O'Reilly Media, 2009

# <codecell>

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist

text = tweets['text']

# <markdowncell>

# Common Words of a Language to filter out

# <codecell>

stop_eng = stopwords.words('english')
stop_ger = stopwords.words('german')
customstopwords = ['tatort', 'mal', 'heute', 'gerade', 'erst', 'macht', 'eigentlich', 'warum', 'gibt', 'gar', 'immer', 'schon', 'beim', 'ganz', 'dass', 'wer', 'mehr', 'gleich', 'wohl']

# <markdowncell>

# Clean the Tweets from a bunch of stuff we are not interested in

# <codecell>

tokens = []
sentences = []
for txt in text.values:
    sentences.append(txt.lower())
    tokens.extend([t.lower().encode('utf-8').strip(":,.!?") for t in txt.split()])

hashtags = [w for w in tokens if w.startswith('#')]
mentions = [w for w in tokens if w.startswith('@')]
links = [w for w in tokens if w.startswith('http') or w.startswith('www')]
filtered_tokens = [w for w in tokens \
                   if not w in stop_eng \
                   and not w in stop_ger \
                   and not w in customstopwords \
                   and w.isalpha() \
                   and not len(w)<3 \
                   and not w in hashtags \
                   and not w in links \
                   and not w in mentions]

# <headingcell level=2>

# Top 30 Words

# <codecell>

freq_dist = nltk.FreqDist(filtered_tokens)
freq_dist

# <codecell>

plt.figure(figsize=(16,5))
plt.xticks(size=16)
freq_dist.plot(31)

# <headingcell level=2>

# When does the community got, who the murderer was?

# <markdowncell>

# The murderer was the neighbour Ms Höllerer, an pharmacist ([ger] 'Apothekerin')

# <codecell>

tweets[tweets.text.str.contains('Apothekerin')==True][['user','text','follower']].head(5)

# <markdowncell>

# Congrats [@ClaudeeyaS](https://twitter.com/ClaudeeyaS/status/463028923178418176), you are the first one on Twitter, who got it!

# <codecell>

Let's take a look at the 

# <codecell>

plt.figure(figsize=(16,5))
tweets.text.str.contains(u"apothekerin").resample('1t').plot(label='Apothekerin')
tweets.text.str.contains(u"Höllerer").resample('1t').plot(label='Höllerer')
plt.legend(loc='best')
plt.axvline(datetime(2014, 5, 4, 18, 15, 0, 0), label='Begin', color='k', alpha=0.6)
plt.axvline(datetime(2014, 5, 4, 19, 45, 0, 0), label='End', color='k', alpha=0.6)

# <markdowncell>

# The Tatort ended at 21:45, the peaks with `Apothekerin` after that are reviews and mostly, because it was Trending Topic and so the bots came to use the hashtag while real people ended writing about #Tatort.

# <codecell>

tweets[tweets.text.str.contains('Apothekerin')==True]['201405042145':][['user','text','follower']].sort('follower', ascending=False).head(10)

# <headingcell level=2>

# Concordance

# <markdowncell>

# Use of the same word in context

# <headingcell level=4>

# Praktikant

# <codecell>

tweettokens = nltk.wordpunct_tokenize(unicode(sentences))
rawtweettext = nltk.Text(tweettokens)
rawtweettext.concordance("praktikant")

# <headingcell level=4>

# What else the community said to the young man?

# <codecell>

rawtweettext.similar('praktikant')

# <headingcell level=4>

# (Justin) Bieber

# <codecell>

rawtweettext.concordance("Bieber")

# <headingcell level=2>

# Collocations

# <markdowncell>

# In corpus linguistics, a collocation is a sequence of words or terms that co-occur more often than would be expected by chance.

# <codecell>

tweettext = nltk.Text(filtered_tokens)
tweettext.collocations()

# <headingcell level=2>

# Search for Words

# <codecell>

fdist = nltk.FreqDist([w.lower() for w in tweettext])
modals = ['apothekerin', 'angst', 'leitmayr', 'nutte', 'messer', 'irre', 'professionelle', 'praktikant']
for m in modals:
    print m + ':', fdist[m],

# <headingcell level=2>

# Names in this Tatort

# <codecell>

names = nltk.corpus.names

# <codecell>

namen = [n.lower().encode('utf-8') for n in names.words('male.txt') or names.words('female.txt')]

# <codecell>

name_freq = nltk.FreqDist([w for w in filtered_tokens if w in namen])

# <codecell>

name_freq.plot(6)

# <codecell>

s = []
x = []
y = []
for val in name_freq.values():
    sn = float(val)/np.max(name_freq.values()) # Normalize
    if sn<0.11:  # ignore unimportant names below this
        continue
    s.append((100.0*sn)**2) # size of bubble
da = 2.0*np.pi/(len(s)-1)
for p in range(len(s)):
    a=p*da   # angle
    r=0.1/np.sqrt(s[p])
    if p==0: # most important name in the middle
        plt.text(0,0,name_freq.keys()[p], ha='center', va='center')
        x.append(0)
        y.append(0)
    else:
        plt.text(np.cos(a)*r,np.sin(a)*r,name_freq.keys()[p], ha='center', va='center')
        x.append(np.cos(a)*r)
        y.append(np.sin(a)*r)
        
# Plot it
plt.scatter(x=x, y=y, s=s, alpha=0.5)
plt.axis('equal')
plt.axis('off');
plt.title('important people in #%s' % hashtag);
plt.savefig('important-people-%s.png' % hashtag, bbox_inches='tight', dpi=300)

# <markdowncell>

# People named the assistant `Justin Bieber`, music was not from `Johnny Cash` but from `Waylon Jenning`. And obviously, `Mike 'Magnum' Hansen` was famous.

# <headingcell level=2>

# Dispersion Plot

# <markdowncell>

# Determine the location of a word in the text: how many words from the beginning it appears. This positional information can be displayed using a dispersion plot.

# <codecell>

plt.figure(figsize=(16,2))
rawtweettext.dispersion_plot(["franz", u"mike", "justin", "johnny"])

# <headingcell level=2>

# Sentiment Analysis

# <markdowncell>

# Use [SentiWS](http://asv.informatik.uni-leipzig.de/download/sentiws.html) as training set.
# 
# ```
# R. Remus, U. Quasthoff & G. Heyer: SentiWS - a Publicly Available German-language Resource for Sentiment Analysis.
# In: Proceedings of the 7th International Language Ressources and Evaluation (LREC'10), pp. 1168--1171, 2010
# 
# SentiWS is licensed under a Creative Commons Attribution-Noncommercial-Share Alike 3.0 Unported License.
# ```

# <codecell>

training_set=[]

# <codecell>

import csv
poswords = csv.reader(open('SentiWS_v1.8c/SentiWS_v1.8c_Positive.txt', 'rb'), delimiter='|')
training_set.extend([(pos[0].lower(), 'positive') for pos in poswords])

# <codecell>

negwords = csv.reader(open('SentiWS_v1.8c/SentiWS_v1.8c_Negative.txt', 'rb'), delimiter='|')
training_set.extend([(neg[0].lower(), 'negative') for neg in negwords])

# <markdowncell>

# Additionally, you may want to specify positive or negative tweet examples:

# <codecell>

pos_tweet = ['Das war ein guter', \
             'Spitze', \
             'Guter', \
             'spannend bis zum schluss', \
             'bester tatort seit langem', \
             'top', \
             'Ein skurriler tatort heut. #ilike', \
             'erstaunlich guter', \
             u'Wirklich überraschend dieser', \
             'Grandios']
#training_set.extend([(pos_tweets.lower().encode('utf-8'), 'positive') for pos_tweets in pos_tweet])

# <codecell>

neg_tweet = ['Langweilig', \
             u'Blöder', \
             'umschalten', \
             'er soll sterben', \
             'abschalten bei dem mist', \
             u'absehbar, wer der mörder ist', \
             'was will dieser justin bieber', \
             u'Ich hab so ein ungutes Gefühl', \
             'justin bieber', \
             'so ein praktikant mit der waffe', \
             'was will dieser bubi', \
             u'Warum tragen die Frauen im heutigen Tatort so miese Perücken?', \
             u'Ja, diese Musik stört', \
             'Ich schalte um']
#training_set.extend([(neg_tweets.lower().encode('utf-8'), 'negativ') for neg_tweets in neg_tweet])

# <markdowncell>

# The more realistic, the better, but language is complicated and especially on twitter, where actually nobody using sentences but abbraviations etc.
# Never the less, let's try it:

# <headingcell level=3>

# Train a Naive Bayes Classifier

# <markdowncell>

# Basically, this is supervised machine learning and Jake Vanderplas made a great talk about that: [Machine Learning with Scikit-Learn - Jake Vanderplas on Vimeo](https://vimeo.com/80093925)

# <markdowncell>

# First, we need samples. Our samples are the 2000 most used words out of all tweets (`freq_dist` to just use every word once):

# <codecell>

samples = freq_dist.keys()[:2000]
samples[:10]

# <codecell>

print('We have %s samples.' % len(samples))

# <markdowncell>

# Second, we need a feature.
# 
# A feature here is following:
# 
# * Every word from the collected Tweets get it's feature with `True` or `False` value, depending on, if it is in the Tweet or not
# * so by iterating over every Tweet, every word in the sample set should at least one time get the feature `True`
# * because we use a training set and have known sentiment values (Supervised Learning), these `true` or `false` will get `positive` or `negative` values as features later
# 
# That is the easiest way of sentiment analysis.
# It will not cover negations, like
# 
# ```
# this was not a good movie
# ```
# 
# because it is just checking for `good` and `movie`.

# <markdowncell>

# The dictionary that is returned by this function is called a feature set and maps from
# features’ names to their values. Feature names are case-sensitive strings that typically
# provide a short human-readable description of the feature. Feature values are values
# with simple types, such as Booleans, numbers, and strings.

# <codecell>

def tweet_features(tweet):
    features={}
    for word in samples:
        features['contains(%s)' % word] = (word in tweet)
    return features

# <headingcell level=4>

# Create a Training Featureset

# <markdowncell>

# Now we take our SentiWS training set and threat it like it were a Tweet. So, if a word from the SentiWS training set is in the samples list of the words we have in all the Tweets, we also have a sentiment (positive or negative) to classify it.
# 
# All that is saved in the `trainingfeatureset`.

# <codecell>

trainingfeatureset = [(tweet_features(word), sentiment) for (word, sentiment) in training_set]

# <headingcell level=4>

# Build the Classifier

# <markdowncell>

# The classfier now checks, if some words are more likely tagged with `positive` or `negative` values.

# <codecell>

classifier = nltk.NaiveBayesClassifier.train(trainingfeatureset)

# <markdowncell>

# And there are some words:

# <codecell>

classifier.show_most_informative_features(14)

# <markdowncell>

# These ratios are known as likelihood ratios, and can be useful for comparing different feature-outcome relationships.
# 
# Notice the last shown: If a tweets contains `RTL` (a german TV channel), the tweet is 4.5x more likely to be negative. :)

# <headingcell level=3>

# Example Automatic Sentiment Classification based on the SentiWS Training Set

# <markdowncell>

# just Tweets from 10 seconds after the end of the Tatort.

# <codecell>

fr = '201405042145'
to = '20140504214510'
positivtweets = []
negativtweets = []
for t in range(len(tweets[fr:to].text)):
    tt = tweets[fr:to].text[t]
    ts = classifier.classify(tweet_features(tt))
    if ts=='positive':
        positivtweets.append(tt)
    else:
        negativtweets.append(tt)

# <headingcell level=4>

# Positive

# <codecell>

for tweet in positivtweets:
    print tweet

# <headingcell level=4>

# Negative

# <codecell>

for tweet in negativtweets:
    print tweet

# <markdowncell>

# Not bad for such a simple classifier!

# <headingcell level=3>

# Now let's do it for all collected Tweets

# <markdowncell>

# Define a function which returns the sentiment from our classifier

# <codecell>

def classifytweet(dataframe):
    return classifier.classify(tweet_features(dataframe.text))

# <markdowncell>

# Apply to all Tweets (takes a while!)

# <codecell>

tweets['sentiment'] = tweets.apply(classifytweet, axis=1)

# <markdowncell>

# Now we can look, how the mood of the crowd was

# <codecell>

plt.figure(figsize=(16,5))
pd.ewma(tweets.sentiment.str.match('positive').resample('1t'), 2).plot(label='mood of the Twitter crowd', color='g', alpha=0.8)
plt.ylim(0.3,0.7)
plt.axhline(0.5, alpha=0.2, label='indifferent')
plt.axvline(datetime(2014, 5, 4, 18, 15, 0, 0), label='Begin of Tatort', color='k', alpha=0.1)
plt.axvline(datetime(2014, 5, 4, 19, 45, 0, 0), label='End of Tatort', color='k', alpha=0.1)
plt.annotate('positive', xy=(0.5, 0.8),  xycoords='axes fraction',
            xytext=(0.5, 0.8), textcoords='axes fraction', size=16,
            horizontalalignment='center', verticalalignment='center')
plt.annotate('negative', xy=(0.5, 0.8),  xycoords='axes fraction',
            xytext=(0.5, 0.2), textcoords='axes fraction', size=16,
            horizontalalignment='center', verticalalignment='center')
plt.legend(loc=4)
plt.title('Mood of the Twitter Crowd for #Tatort, estimated by a Naive Bayes Sentiment Classificator')
plt.savefig('mood-crowd-%s.png' % hashtag, bbox_inches='tight', dpi=300)

# <markdowncell>

# Thanks for watching. :)

