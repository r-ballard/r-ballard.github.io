---
layout: post
title:  "K-Means Clustering Using Public Data"
---


# Twitter K-Means Clustering Sample Analysis
R. Ballard - May 2020

This markdown Jupyter notebook contains a write-up and python script for a relatively simple k-means analysis on recent Tweets with given search parameters.


# Overview
1. A Twitter Dev account was created and an API key was generated.
2. An Environment is deployed which has the necessary analysis packages installed #TODO: CREATE ENVIRONMENT REPO.
3. Search parameters are set. In this case they relate to COVID-19 and are either geotagged or belong to accounts with locations listed within 10km of Reagan National Airport.
4. 1000 Recent Tweets with the above parameters are returned from the Twitter search API.
5. This is saved to an archive directory. By repeatedly extracting the data over time trend analysis becomes possible #TODO: DEVELOP TREND ANALYSIS
6. 


```python
from pathlib import Path
import json
import configparser
import requests
import pandas
import numpy
import tweepy
import datetime
import nltk
import string
import collections
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

This will download nltk stopwords and punkt if not already downloaded.


```python
%%capture
nltk_pkg = ['stopwords','punkt']
for i in nltk_pkg:
    try:
        nltk.data.find(i)
    except LookupError:
        nltk.download(i)
```

## Define Constants and Global Variables
Global constants within the script are defined. Timestamp enumerated, secrets path named.


```python
config = configparser.ConfigParser()

#Set current datetime, will be used for processing timestamps
ymdhms = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

#Set location of keyfile .ini - contains Twitter app API secrets
keyfile_path=[secrets.ini]

#Read secrets, could be assigned as environment variables, etc
config.read(keyfile_path);
```

Stop word list is selected, nltk English stopwords are used.
Then, additional strings are added. These were selected by the analyst as they occured frequently in the dataset and have limited value in the context of analysis. These additional strings were identified by manual inspection of the tokenized dataset.


```python
#Set variable to be used for stop words set to nltk English stopwords
stop_words = list(nltk.corpus.stopwords.words('english'))
stop_words.extend(['rt','\n','\t','`','``','”','“','’','•','\'s','\'\'','\'ve','\'ll','‼️',"'d", "'re", 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would'])
stop_words.extend(string.punctuation)

```


```python
#This regex will perform a capturing look-forward from http* or ftp* to the first following space. This is used to parse out urls in tweets
url_target = r'(http[s]?|ftp)([^\s]+)'

#This regex searches for words ending in ellipses, which are truncated and may inflate counts
#e.g. con... short for contagious? congress? continent? ..etc
trunc_str_target = r'(\S+\.\.\.)'
trunc_str_target2 = r'(\S+…)'
```

## Interact with recent Tweets using Tweepy
In this step we pass Twitter API keys read in from the config file defined above to the Twitter API using tweepy. With an authenticated token passed an API instance is generated and we are able to select from recent Tweets for our analysis.


```python
#Authenticate key parsed from secrets
auth = tweepy.AppAuthHandler(config['consumer_keys']['API_KEY'], config['consumer_keys']['API_SECRET_KEY'])

#Instatiate tweepy API object, set large timeout value to extract many tweets
api = tweepy.API(auth,timeout=100000)
```

## Define Search Terms
We define search criteria to be passed to the Twitter API. Documentation here : #TODO LIST DOCUMENTATION


```python
#TODO List Twitter API documentation
#Sets search terms, joins term list using OR as concatenator to generate search string

target_terms=['coronavirus','covid-19','covid19','air travel']
separator = " OR " 
target_term_str = separator.join(target_terms)

#TODO LIST API SEARCH LOCATION DEFINITION ETC

"""Sets search criteria for 10km of Reagan airport
could be extended to include additional airports or other locations and use this in an iterator"""

airport_locs={'reagan':{'lat':38.8512,'long':-77.0402}}
radius = 10
r_unit ='km'

loc_target = f"{airport_locs['reagan']['lat']},"+\
                f"{airport_locs['reagan']['long']},"+\
                f"{radius}{r_unit}"
```

The below chunk will set a value for max_tweets, 1000.
Instantiate an empty list of searched_tweets
Set a last_id value of -1 (most recent) and iterate while the length of the list searched_tweets is less than 1000.
For each iteration, query the Twitter API via the Tweepy authentication object to return a Tweet with max_id of last_id
Returned tweets will be apended to a list of searched_tweets
last_id will be backwards iterated from most recent found Tweet.


```python
max_tweets = 1000
searched_tweets = []
last_id = -1
while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        new_tweets = api.search(q=target_term_str, geocode = loc_target, count=count, max_id=str(last_id - 1))
        if not new_tweets:
            break
        searched_tweets.extend(new_tweets)
        last_id = new_tweets[-1].id
    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break
```

The below chunk will instantiate an empty dataframe, df.
It will then iterate over the items in searched_tweets and for each item in searched_tweets append a row to df (Append a Tweet item to df) using json_normalize.
Subsequently, the DataFrame index will be reset and our previously defined timestmap will be appended to the dataframe.


```python
#Convert list of tweets to dataframe
df = pandas.DataFrame()
for j in range(len(searched_tweets)):
    df = df.append(pandas.io.json.json_normalize(searched_tweets[j]._json),sort=True)
    
#Reset Index, format created_at field to timestamp from Twitter datetime
df.reset_index(drop=True,inplace=True)
df['created_at']=pandas.to_datetime(df['created_at'], format= '%a %b %d %H:%M:%S %z %Y')
```

The below chunk will write the Tweet dataframe to a user defined write directory as a pipe-delimited csv.


```python
#Write data to csv

#TODO abstract write path
write_dir = Path(config['paths']['write_path'])
file_name = f'twitter_dataset_{len(df)}_{ymdhms}.csv'

write_path = Path(write_dir,file_name)
df.to_csv(write_path,sep='|',index=False)
```

## Parse Text
This Section contains scripting for parsing and cleaning the returned Tweet dataframe containing Tweets with the above defined search criteria.


```python
#Create a new field in the Tweet dataframe containing Tweets cast to lowercase values
df['text_lower'] = df['text'].str.lower()

#This section drops url_target, trunc_str_target, trunc_str_target2
df['text_lower'] = df['text_lower'].str.replace(url_target, '',regex=True)
df['text_lower'] = df['text_lower'].str.replace(trunc_str_target, '',regex=True)
df['text_lower'] = df['text_lower'].str.replace(trunc_str_target2, '',regex=True)
```


```python
#This creates a new field in dataframe that applies nlt.word_tokenize to lowercase string Tweet in df
df['tokenized_text'] = df['text_lower'].apply(nltk.word_tokenize)

#Further subsets Tweets by dropping stop words identified above.
df['tokenized_text_drop_sw'] =df['tokenized_text'].apply(lambda x: [w for w in x if w not in stop_words])
```

### Bag Of Words
The below cell uses a list generator to create a single list containing all tokenized Tweets.


```python
#This creates a bag of words from the tweets. Concatenates all tokenized tweet text into a list.
#https://stackoverflow.com/questions/716477/join-list-of-lists-in-python

token_list = [j for i in df['tokenized_text_drop_sw'] for j in i]
```

## Exploratory Charts
This section contains exploratory graphs visualizing the Tweet dataset.

### 1. Common word frequency distribution of most common words in token 
Below is a histogram containing counts of the 25 most common words


```python
#Common word frequency distribution
plt.figure(figsize=(12,5))
plt.title('Top 25 most common words')
plt.xticks(fontsize=13, rotation=90)
fd = nltk.FreqDist(token_list)
fd.plot(25,cumulative=False)
```


![png](/images/kmeans_charts/output_26_0.png)


### 2. Log-Log Plot
The below plot is a Log-Log plot of the Tweet bag of words. This chart can indicate focus of conversation. If we see the most common words used much more frequently with plateaus present that indicates that the more common words are used exponentially more than the less-common ones. Which could be an indicator that many people are Tweeting about the same thing.

A wider array of topics would lead to a more diffuse lexicon which would lead to a more tapered decrease in the log-log chart. 


```python
#Log-Log Plot
word_counts = sorted(collections.Counter(token_list).values(), reverse=True)
plt.figure(figsize=(12,5))
plt.loglog(word_counts, linestyle='-', linewidth=1.5)
plt.ylabel("Freq")
plt.xlabel("Word Rank")
plt.title('log-log plot of words frequency')
```




    Text(0.5,1,'log-log plot of words frequency')




![png](/images/kmeans_charts/output_28_1.png)


## K-Means Clustering
In this section we will create a Term Frequency Inverse Document Frequency Vectorizer with the modified nltk stopwords list and use this to transform the Tweets.


```python
#Vectorizer reverts to untokenized field, could be adapted to use tokenized field with stop words already removed.
vectorizer = TfidfVectorizer(stop_words = stop_words)
#vectorizer = TfidfVectorizer(stop_words = stop_words,tokenizer=nltk.word_tokenize)

desc = df['text_lower'].values

#This reverts to fitting the lowercase text of the tweets rather than the tokenized version.
X = vectorizer.fit_transform(desc)
```

### Elbow Chart
With the vectorized set of Tweets we generate an Elbow chart to attempt to find the inflection point where the slope of the Within-Cluster Sum of Squares line begins to become flattened. This will aid us in determining the optimal number of clusters for the dataset.

In the below example 9 looks to be a candidate for the number of clusters.


```python
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```


![png](/images/kmeans_charts/output_32_0.png)


### Silhouette Chart

Similar to the Elbow Chart The Silhoette Chart will help us in determining the optimal number of clusters K to in which to group the Tweets.

The silhouette value measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation) The silhouette value is between -1 and 1, and higher values indicate better cohesion and separation (more optimal clustering).

Based on this example 9 clusters appears to be optimal.


```python
sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(X)
  labels = kmeans.labels_
  sil.append(silhouette_score(X, labels, metric = 'euclidean'))

plt.plot(range(2,kmax+1),sil)
plt.title('The Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Value')
plt.show()
```


![png](/images/kmeans_charts/output_34_0.png)


## Clustering
With the number of clusters identified using the elbow method above we then fit our model and for each cluster list the top 25 words most common words in that cluster. This gives us some indication of different topical groupings for recent Tweets mentioning COVID-19 in their text bodies for recent Tweets and Twitter accounts based near Reagan National Airport.


```python
word_features = vectorizer.get_feature_names()
kmeans = KMeans(n_clusters = 9, n_init = 20, n_jobs = 1) # n_init(number of iterations for clustering) n_jobs(number of cpu cores to use)
kmeans.fit(X)
# We look at n the clusters generated by k-means.
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(word_features[word] for word in centroid))
```

    0 : new, squad, name, afthunderbirds, formation, flyovers, blueangels, series, deptofdefense, better, york, travel, feet, feel, facing, flyingwithsara, flying, flu, federal, fallen, flowers, flower, floating, fever, flights
    1 : covid, 19, cdc, overruled, troubling, screenings, fever, restart, headline, push, scientists, airport, brasilmagic, like, white, house, states, united, countries, cities, wildfire, transit, border, mexican, spreading
    2 : symbolic, ignored, wasted, kicked, feet, dncwarroom, issued, ban, experts, trump, said, time, travel, flower, floating, flowers, flu, flights, flight, zealand, first, flying, finds, fill, fever
    3 : wane, message, reopen, country, scottdetrow, lots, week, white, house, coronavirus, time, first, zealand, flight, fiqaajamal, floating, flower, flowers, flu, flights, fever, finds, fill, feet, feel
    4 : put, ranttmedia, truth, twice, shoes, swift, jonathan, around, lie, world, said, travel, facing, flower, floating, flights, flight, experts, first, fiqaajamal, extremely, finds, face, fallen, fill
    5 : en, estados, ser, para, recibir, de, ciudadanos, unidos, inscríbase, viajeros, alertas, usembassyve, localizado, federal, facing, flower, face, floating, flights, flight, first, fiqaajamal, finds, feel, fallen
    6 : statedept, alerts, 00, please, located, citizen, enroll, receive, travelers, ensure, related, covid19, visit, step, enrollment, answers, questions, et, 21, published, health, international, belarus, croatia, canada
    7 : president, coronavirus, air, travel, quarantine, trump, urge, paris, two, viewership, government, strongly, british, borisjohnson, devastate, rethink, natgeotravel, several, airlines, due, transatlantic, nilegardiner, week, pence, officials
    8 : confidence, crews, measures, increased, restore, needed, public, safety, keep, safe, amp, air, passengers, flyingwithsara, pax, agree, afa_cwa, travel, floating, flights, flight, flower, zealand, fever, first
    
