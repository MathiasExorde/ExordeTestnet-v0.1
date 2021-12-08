from pmaw import PushshiftAPI
import csv 
from keybert import KeyBERT
from time import sleep
import matplotlib.pyplot as plt
import urllib.request
import calendar
import datetime
from datetime import timedelta
import time
import plotly.graph_objs as go
import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
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
import urllib.request
from pymilvus import (
    connections, list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection
)
import asyncio
import logging
import torch

torch.cuda.empty_cache()

logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)

device = torch.device("cuda")
selected_model = 'paraphrase-mpnet-base-v2' # 'microsoft/deberta-xlarge', 'microsoft/deberta-v2-xlarge', 'paraphrase-MiniLM-L6-v2'

    

def get_duplicates(field = '$id'):        
    cursor = data_col.aggregate([{'$group': {'_id': field, 'count': {'$sum': 1}}}, {'$match': {'count': {'$gt': 1}}}],allowDiskUse=True)
    query_result = list(cursor)
    duplicates_ids = [d['_id'] for d in query_result]
    return duplicates_ids


async def get_last_n_tweets(kw_model, nb_tweets, keyword, date_threshold, lang = 'en'):   
    tweets_scrapes = []
    tweets_keywords = []
    # Using TwitterSearchScraper to scrape data
    print("Scraping tweets with keyword = ",keyword)
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('{} since:{}  lang:{}'.format(keyword, date_threshold, lang)).get_items()): #declare a username 
        if i >= nb_tweets: 
            break
        if len(str(tweet.content))>10:
            dict_obj = {'url': tweet.url , 'date': tweet.date, 'id': tweet.id, 'text': tweet.content, 'username': tweet.user.username, 'lang': tweet.lang, 'hashtags': tweet.hashtags, 'keyword_': keyword}
            keywords = kw_model.extract_keywords(tweet.content, keyphrase_ngram_range=(1, 1), stop_words='english')
            tweets_scrapes.append(dict_obj)
            tweets_keywords.append(keywords)            
    print(len(tweets_scrapes)," tweets scraped.")
    # queue.task_done()
    return (tweets_scrapes, tweets_keywords)




nb_tweets_per_keyword = 50

# create Milvus DB connection
connections.connect()

print(f"\nList Milvus collections...")
print(list_collections())

milvus_collection = Collection('web')

print("Connected to collection = ",milvus_collection)

mongo_db_hostname = "127.0.0.1"
print(f"\nList MongoDB collections...")
mongo_db_client = MongoClient(mongo_db_hostname, 27017)
print("mongo_db_client = ",mongo_db_client)

################################################################
print("Loading MongoDB db & collection...")
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

filter_texts = ['removed','deleted']

kw_model = KeyBERT()

debug = False

nb_iterations = 10000


################################################################################################################################
# BERT BASED MODEL SELECTION
#  best & most efficient sentence embedder for clustering tasks and semantic search, nice
model = SentenceTransformer(selected_model) 
model.to(device) ## GPU ENABLED
################################################################################################################################
print("\n")

async def main():
    for iteration in range(nb_iterations):
        print("\n_______ Iteration ",iteration," _______")

        #time boundaries
        yesterday = datetime.date.today() - datetime.timedelta(1)
        ts_now = int(time.time())
        ts_last_day = ts_now - 86400
        after = ts_last_day

        d = datetime.datetime.today() - timedelta(hours=3)

        datetime_now_minus_xh_formatted = d.strftime('%Y-%m-%d-%H:%M:%S')
        print("Scraping since:", datetime_now_minus_xh_formatted)
        URL_keywords = "https://exploreserverstaging0nnzxf9h-exploreserverstaging.functions.fnc.fr-par.scw.cloud/api/v1/keywords"
        url_response_content = urllib.request.urlopen(URL_keywords).read()
        # print("url_response_content = ",url_response_content)
        keywords_json = json.loads(url_response_content.decode())

        ####################################################################################################
        followed_keywords = []
        for keyword_struct in keywords_json:
            label_ = keyword_struct['label']
            followed_keywords.append(label_)
        print("followed_keywords = ",followed_keywords)
        selected_keywords = followed_keywords

        #################### TWITTER SCRAPING ##############################################################
        
        followed_keywords = followed_keywords + followed_keywords +followed_keywords


        scrape_tasks = []

        for fk in followed_keywords:
            kw_model_ = kw_model
            nb_tweets_ = nb_tweets_per_keyword
            keyword_ = fk
            date_threshold_ = datetime_now_minus_xh_formatted
            
            print("adding task for kw = ",keyword_)
            scrape_tasks.append(get_last_n_tweets(kw_model_, 
                                    nb_tweets_, 
                                    keyword_, 
                                    date_threshold_,
                                    lang = 'en'))

        print("wait for workers to end their work")

        N = 3
        res =  await asyncio.gather(*scrape_tasks)
        print(res.shape)


        # scrape_tasks = asyncio.Queue()

        # for fk in followed_keywords:
        #     kw_model_ = kw_model
        #     nb_tweets_ = nb_tweets_per_keyword
        #     keyword_ = fk
        #     date_threshold_ = datetime_now_minus_xh_formatted
            
        #     print("adding task for kw = ",keyword_)
        #     scrape_tasks.put_nowait(get_last_n_tweets(kw_model_, 
        #                             nb_tweets_, 
        #                             keyword_, 
        #                             date_threshold_, scrape_tasks,
        #                             lang = 'en'))
        # async def worker():
        #     while not scrape_tasks.empty():
        #         await scrape_tasks.get_nowait()       

        # print("wait for workers to end their work")
        # res =  await asyncio.gather(*[worker() for _ in range(N)])
        # print(scrapes_lists)


        tweets_text = []
        tweets_scrapes = []
        tweets_keywords = []


        exit(1)

    #     # Using TwitterSearchScraper to scrape data and append tweets to list
    #     for keyword in selected_keywords:
    #         print("Scraping tweets with keyword = ",keyword)
    #         c = 0
    #         for i,tweet in enumerate(sntwitter.TwitterSearchScraper('{} since:{}  lang:en'.format(keyword, datetime_now_minus_xh_formatted)).get_items()): #declare a username 
    #             if i >= nb_tweets_per_keyword: 
    #                 break
    #             if len(str(tweet.content))>10:
    #                 dict_obj = {'url': tweet.url , 'date': tweet.date, 'id': tweet.id, 'text': tweet.content, 'username': tweet.user.username, 'lang': tweet.lang, 'hashtags': tweet.hashtags, 'keyword_': keyword}
    #                 tweets_scrapes.append(dict_obj) #declare the attributes to be returned
    #                 tweets_text.append(tweet.content)

    #                 keywords = kw_model.extract_keywords(tweet.content, keyphrase_ngram_range=(1, 1), stop_words='english')
    #                 tweets_keywords.append(keywords)
    #                 c += 1
    #         print(c," tweets extracted.")
            

    #     print(len(tweets_scrapes)," tweets scraped.")


    #     scraped_tweets_embeddings = model.encode(   sentences=tweets_text,
    #                                                 batch_size= 32, 
    #                                                 show_progress_bar=True)
    #     scraped_tweets_embeddings_list =list(scraped_tweets_embeddings)

    #     print(scraped_tweets_embeddings.shape)
    #     print(len(tweets_scrapes))

    #     print("Vector insertion ...")
    #     mr = milvus_collection.insert([scraped_tweets_embeddings_list])
    #     vector_ids = mr.primary_keys

    #     # print("IDs = ",vector_ids)
    #     # ADD VECTOR IDs with new id_ key
    #     for ts, vid, kws in zip(tweets_scrapes, vector_ids, tweets_keywords):
    #         ts['id_'] = vid
    #         ts['source_'] = "twitter"
    #         ts['type_'] = "tweet"
    #         ts['keywords_'] = [x[0] for x in kws]
    #         ts['timestamp_'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        
    #     print("Sample = ",tweets_scrapes[0],"\n")


    #     print("Data Insertion in MongoDB...")
    #     data_col.insert_many(tweets_scrapes)

    #     if iteration % 10 == 0 and iteration>0:
    #         field = '$id'
    #         duplicates_ids =  get_duplicates()
    #         print("\nNumber of duplicates based on data field [", field,"] = ",len(duplicates_ids))
    #         print("Deletion of duplicates...")
    #         mongo_db.scraped_data.delete_many({'id':{"$in":duplicates_ids}})  
    #         print("MongoDB Database stats after deletion: ", mongo_db.scraped_data.estimated_document_count(), " documents" )
            
    # print("End.")


asyncio.run(main())