# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Analyse Twitter Stream

# <markdowncell>

# Hopefully, you have a local MongoDB up and running with all Tweets in it

# <markdowncell>

# Thanks to [this](http://stackoverflow.com/a/16255680), I can made this.

# <codecell>

import pandas as pd
from pandas.tseries.resample import TimeGrouper
from pandas.tseries.offsets import DateOffset
from pymongo import MongoClient
import matplotlib.pyplot as plt
%pylab inline --no-import-all

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

# Connect and get the Data

# <codecell>

dbname = 'Tweets'
collection = 'Tweets'
tweets = read_mongo(dbname, collection)

# <codecell>

tweets.set_index('created_at', inplace=True)
tweets.index = tweets.index.tz_localize('GMT').tz_convert('CET')
tweets.index.name = 'Zeit'
tweets.index

# <headingcell level=3>

# Show it

# <codecell>

tweets.tail(10)

# <headingcell level=2>

# Tweets per Minute

# <codecell>

tweetsperminute = tweets['text'].resample('1t', how='count')

# <codecell>

plt.figure(figsize=(14,4))
tweetsperminute.plot()
plt.ylabel('Tweets per Minute')

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


# <codecell>

locations = [l for l in tweets.geo.values if l!=None]

# <codecell>

map = folium.Map(location=[50, 10], zoom_start=5, tiles='Stamen Toner')
for marker in locations:
    map.simple_marker(marker.values()[1])
inline_map(map)

# <headingcell level=1>

# Text Processing with the Natural Language Toolkit

# <codecell>

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
stop_eng = stopwords.words('english')
stop_ger = stopwords.words('german')
text = tweets['text']

# <codecell>


# <codecell>

tokens = []
sentences = []
for txt in text.values:
    sentences.append(txt.lower())
    tokens.extend([t.lower().strip(":,.!?") for t in txt.split()])

hashtags = [w for w in tokens if w.startswith('#')]
mentions = [w for w in tokens if w.startswith('@')]
links = [w for w in tokens if w.startswith('http') or w.startswith('www')]
filtered_tokens = [w for w in tokens \
                   if not w.encode('utf-8') in stop_eng \
                   and not w.encode('utf-8') in stop_ger \
                   and w.isalpha() \
                   and not len(w)<3 \
                   and not w in hashtags \
                   and not w in links \
                   and not w in mentions]

# <headingcell level=2>

# Top 20

# <codecell>

freq_dist = nltk.FreqDist(filtered_tokens)
freq_dist

# <codecell>

freq_dist.plot(20)

# <codecell>

tweets.text.str.contains("raab").resample('1t', how='mean').plot()
tweets.text.str.contains("sandburg").resample('1t', how='mean').plot()

# <codecell>


# <headingcell level=2>

# Concorance

# <codecell>

tweettokens = nltk.wordpunct_tokenize(unicode(sentences))
rawtweettext = nltk.Text(tweettokens)
rawtweettext.concordance("ente")

# <codecell>

rawtweettext.similar('raab')

# <headingcell level=2>

# Collocations

# <codecell>

tweettext = nltk.Text(filtered_tokens)
tweettext.collocations()

# <codecell>


# <codecell>


# <headingcell level=2>

# Search for Words

# <codecell>

fdist = nltk.FreqDist([w.lower() for w in tweettext])
modals = ['gewinnen', 'verlieren', 'peinlich', 'gewonnen', 'raab', 'will']
for m in modals:
    print m + ':', fdist[m],

# <headingcell level=2>

# People Involved

# <codecell>

names = nltk.corpus.names

# <codecell>

namen = [n.lower() for n in names.words('male.txt') or names.words('female.txt')]

# <codecell>

name_freq = nltk.FreqDist([w for w in filtered_tokens if w in namen])

# <codecell>

s = []
x = []
y = []
for val in name_freq.values():
    sn = float(val)/np.max(name_freq.values()) # Normalize
    if sn<0.2:  # ignore unimportant names below this
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
        
    

plt.scatter(x=x, y=y, s=s, alpha=0.5)
plt.axis('equal')
plt.axis('off');
plt.title('important people');

# <headingcell level=2>

# Dispersion Plot

# <markdowncell>

# Determine the location of a word in the text: how many words from the beginning it appears. This positional information can be displayed using a dispersion plot.

# <codecell>

rawtweettext.dispersion_plot(["stefan", "maximilian"])

# <codecell>


# <codecell>


