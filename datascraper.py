from pmaw import PushshiftAPI
import csv 
from keybert import KeyBERT
from time import sleep
import matplotlib.pyplot as plt
import urllib.request
import calendar
import datetime
import time
import plotly.graph_objs as go
import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
import torch
import time
import json
import pickle
import numpy as np
from time import sleep
from nltk import tokenize
from urllib.parse import urlparse
from newsplease import NewsPlease
from bs4 import BeautifulSoup
from plotly import tools
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
import random
from numba import cuda 
from pymilvus import (
    connections, list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection
)
import logging
import pytz


device_cuda = cuda.get_current_device()
device_cuda.reset()

logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)

device = torch.device("cuda")
selected_model = 'paraphrase-mpnet-base-v2' # 'microsoft/deberta-xlarge', 'microsoft/deberta-v2-xlarge', 'paraphrase-MiniLM-L6-v2'

    
def get_duplicates(field = '$id'):        
    cursor = data_col.aggregate([{'$group': {'_id': field, 'count': {'$sum': 1}}}, {'$match': {'count': {'$gt': 1}}}],allowDiskUse=True)
    query_result = list(cursor)
    duplicates_ids = [d['_id'] for d in query_result]
    return duplicates_ids



# search_limit_nb_comments = 500
# search_limit_nb_subs = 50
# nb_tweets_per_keyword = 100
# search_limit_nb_comments = 8000
# search_limit_nb_subs = 2000
# nb_tweets_per_keyword = 300

# search_limit_nb_comments = 12000
# search_limit_nb_subs = 3000
# nb_tweets_per_keyword = 200

search_limit_nb_comments = 5000
search_limit_nb_subs = 200
nb_tweets_per_keyword = 400
select_top_n_kw = 20
select_random_kw = 10


select_top_n_kw = 15
select_random_kw = 5
th_accepted_keywords = 0.8
# create Milvus DB connection
connections.connect()

print(f"\nList Milvus collections...")
print(list_collections())

# milvus_collection = Collection('web_data')
# milvus_collection = Collection('data')
milvus_collection = Collection('web')

print("Connected to collection = ",milvus_collection)

mongo_db_hostname = "127.0.0.1"
print(f"\nList MongoDB collections...")
mongo_db_client = MongoClient(mongo_db_hostname, 27017)
print("mongo_db_client = ",mongo_db_client)

################################################################
print("Loading MongoDB db & collection...")
# mongo_db = mongo_db_client.data
# data_col = mongo_db["scraped_data"]

DATABASE_NAME = "begmatwinbv4mfwiqpcm"
DATABASE_HOST = "begmatwinbv4mfwiqpcm-mongodb.services.clever-cloud.com"
DATABASE_PORT = 2504

DATABASE_USERNAME = "uffjfliayvnvoczan023"
DATABASE_PASSWORD = "Nqmlm3oQgoBWOgQak8B"

mongo_db_client = MongoClient(DATABASE_HOST, DATABASE_PORT)
r = mongo_db_client.admin.authenticate(DATABASE_USERNAME, DATABASE_PASSWORD, mechanism = 'SCRAM-SHA-1', source=DATABASE_NAME)

mongo_db = mongo_db_client.begmatwinbv4mfwiqpcm
data_col = mongo_db["scraped_data"]
print("Connection MongoDB :", r)
################################################################


api = PushshiftAPI(file_checkpoint = 10, rate_limit = 10, num_workers = 16)

financial_subreddits =  ['cryptocurrency', 'cryptomoonshots', 'dogecoin', 'ethtrader', 'Bitcoin', 'ethereum',
                        'wallstreetbets', 'stocks', 'superstonk', 'investing', 'pennystocks', 'AlgoTrading', 'daytrading', 
                        'technews', 'FuturesTrading', 'MasterTheMarket', 'RobinHoodPennyStocks', 'ValueInvesting',
                        'personalfinance', 'financialindependence', 'tax', 'securityanalysis', 'options', 'economy',
                        'creditcards', 'povertyfinance', 'stockmarket','forex']
print("\n Subreddits to be scraped = ",financial_subreddits,"...")
financial_subreddits = list(set(financial_subreddits))

filter_texts = ['removed','deleted']

kw_model = KeyBERT()

debug = False

nb_iterations = 10000

print("\n")
################################################################################################################################
# BERT BASED MODEL SELECTION
#  best & most efficient sentence embedder for clustering tasks and semantic search, nice
model = SentenceTransformer(selected_model) 
model.to(device) ## GPU ENABLED
################################################################################################################################


for iteration in range(nb_iterations):

    print("\n_______ Iteration ",iteration," _______")

    #time boundaries
    yesterday = datetime.date.today() - datetime.timedelta(1)
    ts_now = int(time.time())
    ts_last_day = ts_now - 86400
    after = ts_last_day
    before = ts_now

    try:
        print("\nScraping Reddit comments on ",len(financial_subreddits)," financial-related subreddits... of the previous 24h\n")
        scraped_comments = api.search_comments(subreddit=financial_subreddits, limit=search_limit_nb_comments, mem_safe=True, safe_exit=True,  after=after, before = before)    
        print("Scraping Reddit submissions...")
        scraped_submissions = api.search_submissions(subreddit=financial_subreddits, limit=search_limit_nb_subs, mem_safe=True, safe_exit=True, after=after, before = before)
    except:
        print("failed reddit scrape, retry")
        continue
    print("Pushshift API: ",len(scraped_comments)," comments & ",len(scraped_submissions)," submissions scraped.\n")

    nb_scraped_comments = 0
    nb_scraped_subs = 0

    comment_keywords = []
    comment_texts = []
    reddit_comments_scrapes = []

    for comment in scraped_comments:        
        if 'body' in comment:
            text = comment['body']
        else:
            text = None
            continue

        if debug: 
            print(comment['author']," - ",text)
        if text is not None and len(text)> 150:       
            try:
                keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', 
                                    use_maxsum=True, nr_candidates=5, top_n=5)
            except:        
                keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
        else:        
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
        if debug: 
            print("Keywords = ",keywords)

        nb_scraped_comments += 1

        comment_keywords.append(keywords)
        comment_texts.append(text)
        reddit_comments_scrapes.append(comment)


    scraped_reddit_comments_embeddings = model.encode(   sentences=comment_texts,
                                                batch_size= 32, 
                                                show_progress_bar=True)
    scraped_reddit_comments_embeddings_list =list(scraped_reddit_comments_embeddings)

    print(scraped_reddit_comments_embeddings.shape)
    print(len(reddit_comments_scrapes))

    print("total ",(nb_scraped_comments)," comments scraped.")
    ############################################################################
    print("Vector insertion in MilvusDB...")
    mr = milvus_collection.insert([scraped_reddit_comments_embeddings_list])
    vector_ids = mr.primary_keys

    # ADD VECTOR IDs with new id_ key
    for sc, vid, kws in zip(reddit_comments_scrapes, vector_ids, comment_keywords):
        sc['id_'] = vid
        sc['keywords_'] = [x[0] for x in kws]
        sc['source_'] = "reddit"
        sc['type_'] = "comment"
        uniso_date = datetime.datetime.utcfromtimestamp( sc['created_utc'] ) # /1e3 if in ms
        iso_date = uniso_date.replace(tzinfo=pytz.UTC)
        sc['date_'] =  iso_date 
        sc['timestamp_'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("Sample = ",reddit_comments_scrapes[0],"\n")


    print("Data Insertion in MongoDB...")
    data_col.insert_many(reddit_comments_scrapes)
    ############################################################################


    post_keywords = []
    post_texts = []
    reddit_comments_scrapes = []
    reddit_posts_scrapes = []

    for post in scraped_submissions:
        if 'selftext' in post:
            text = post['selftext']
        else:
            text = None
            continue
        
        if debug: 
            print("POST:", post['author'] ," - ", post['title'], " - ", text)
        if text is not None and len(text)> 100:       
            try:
                keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', 
                                use_maxsum=True, nr_candidates=20, top_n=5)
            except:        
                keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
        elif text is not None:        
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')

        nb_scraped_subs += 1

        post_keywords.append(keywords)
        post_texts.append(text)
        reddit_posts_scrapes.append(post)



    scraped_reddit_post_embeddings = model.encode(      sentences=post_texts,
                                                        batch_size= 32, 
                                                        show_progress_bar=True)
    scraped_reddit_post_embeddings_list =list(scraped_reddit_post_embeddings)

    print(scraped_reddit_post_embeddings.shape)
    print(len(reddit_posts_scrapes))

    print("total ",(nb_scraped_subs)," reddit submissions scraped.")
    ############################################################################
    if len(scraped_reddit_post_embeddings_list)>0:
        print("Vector insertion in MilvusDB...")
        mr = milvus_collection.insert([scraped_reddit_post_embeddings_list])
        vector_ids = mr.primary_keys
        

        # ADD VECTOR IDs with new id_ key
        for sc, vid, kws in zip(reddit_posts_scrapes, vector_ids, post_keywords):
            sc['id_'] = vid
            sc['keywords_'] = [x[0] for x in kws]
            sc['source_'] = "reddit"
            sc['type_'] = "submission"
            uniso_date = datetime.datetime.utcfromtimestamp( sc['created_utc'] ) # /1e3 if in ms
            iso_date = uniso_date.replace(tzinfo=pytz.UTC)
            sc['date_'] =  iso_date 
            sc['timestamp_'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("Sample = ",reddit_posts_scrapes[0],"\n")


        print("Data Insertion in MongoDB...")
        data_col.insert_many(reddit_posts_scrapes)
        ############################################################################
    

    # print(comment_keywords)

    comment_keywords = [item for sublist in comment_keywords for item in sublist]
    filtered_keywords = dict()
    for kw_item in comment_keywords:
        kw = kw_item[0]
        weight = kw_item[1]
        if kw not in filter_texts:
            if weight> 0.05:
                if kw in filtered_keywords:
                    filtered_keywords[kw] += weight #cumulate all weights
                else: #if not in dict, initialize
                    filtered_keywords[kw] = weight 

    # print(filtered_keywords)
    weighted_keywords = filtered_keywords
    # weighted_keywords = dict(sorted(filtered_keywords.items(), key=lambda item: item[1]), reverse=True)


    weighted_keywords = dict(sorted(weighted_keywords.items(), key=lambda x: x[1], reverse=True))
    weighted_keywords = dict((k, v) for k, v in weighted_keywords.items() if v>=th_accepted_keywords)

    # print(weighted_keywords)

    selected_keywords = list(weighted_keywords.keys())[:select_top_n_kw]
    keywords_remaining = list(weighted_keywords.keys())[(select_top_n_kw+1):]

    additional_keywords = random.sample(keywords_remaining, min(select_random_kw,int(len(keywords_remaining)/2)))
    print("\nRandomly selected keywords = ",additional_keywords)
    selected_keywords += additional_keywords
    selected_keywords = list(set(selected_keywords)) #make sure no doublons
    print("Twitter Keywords List = ",selected_keywords,"\n")


    ####################################################################################################
    #################### TWITTER #######################################################################
    ####################################################################################################


    # device = torch.device("cuda")
    selected_model = 'paraphrase-mpnet-base-v2' # 'microsoft/deberta-xlarge', 'microsoft/deberta-v2-xlarge', 'paraphrase-MiniLM-L6-v2'
    # Creating list to append tweet data 
    figure_display_enabled = True

    # filtered_lang = ['en','fr']
    tweets_text = []
    tweets_scrapes = []
    tweets_keywords = []
    
    try:
        # Using TwitterSearchScraper to scrape data and append tweets to list
        for keyword in selected_keywords:
            print("Scraping tweets with keyword = ",keyword)
            c = 0
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper('{} since:2021-10-01 lang:en'.format(keyword)).get_items()): #declare a username 
                if i >= nb_tweets_per_keyword: 
                    break
                if len(str(tweet.content))>25:
                    dict_obj = {'url': tweet.url , 'date': tweet.date, 'id': tweet.id, 'text': tweet.content, 'username': tweet.user.username, 'lang': tweet.lang, 'hashtags': tweet.hashtags, 'keyword_': keyword}
                    tweets_scrapes.append(dict_obj) #declare the attributes to be returned
                    tweets_text.append(tweet.content)

                    keywords = kw_model.extract_keywords(tweet.content, keyphrase_ngram_range=(1, 1), stop_words='english')
                    tweets_keywords.append(keywords)
                    c += 1
            print(c," tweets extracted.")
            

        print(len(tweets_scrapes)," tweets scraped.")

    except:
        print("Snscrape error.. continuing")
        continue


    if len(tweets_text) == 0:
        continue

    scraped_tweets_embeddings = model.encode(   sentences=tweets_text,
                                                batch_size= 32, 
                                                show_progress_bar=True)
    scraped_tweets_embeddings_list =list(scraped_tweets_embeddings)

    print(scraped_tweets_embeddings.shape)
    print(len(tweets_scrapes))

    print("Vector insertion ...")
    mr = milvus_collection.insert([scraped_tweets_embeddings_list])
    vector_ids = mr.primary_keys

    # print("IDs = ",vector_ids)
    # ADD VECTOR IDs with new id_ key
    for ts, vid, kws in zip(tweets_scrapes, vector_ids, tweets_keywords):
        ts['id_'] = vid
        ts['source_'] = "twitter"
        ts['type_'] = "tweet"
        ts['keywords_'] = [x[0] for x in kws]
        ts['date_'] =  ts['date'] 
        ts['timestamp_'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    
    print("Sample = ",tweets_scrapes[0],"\n")


    print("Data Insertion in MongoDB...")
    data_col.insert_many(tweets_scrapes)

    field = '$id'
    duplicates_ids =  get_duplicates()
    print("\nNumber of duplicates based on data field [", field,"] = ",len(duplicates_ids))
    print("Deletion of duplicates...")
    mongo_db.scraped_data.delete_many({'id':{"$in":duplicates_ids}})  
    print("MongoDB Database stats after deletion: ", mongo_db.scraped_data.estimated_document_count(), " documents" )
        
print("End.")