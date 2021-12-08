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
import logging
import pytz


logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)

device = torch.device("cuda")
selected_model = 'paraphrase-mpnet-base-v2' # 'microsoft/deberta-xlarge', 'microsoft/deberta-v2-xlarge', 'paraphrase-MiniLM-L6-v2'

    
def get_duplicates(field = '$id'):        
    cursor = data_col.aggregate([{'$group': {'_id': field, 'count': {'$sum': 1}}}, {'$match': {'count': {'$gt': 1}}}],allowDiskUse=True)
    query_result = list(cursor)
    duplicates_ids = [d['_id'] for d in query_result]
    return duplicates_ids


search_limit_nb_comments = 1000

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

financial_subreddits =  ['cryptocurrency', 'cryptomoonshots', 'dogecoin', 'ethtrader', 'Bitcoin', 'ethereum']
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
    except:
        print("failed reddit scrape, retry")
        continue
    print("Pushshift API: ",len(scraped_comments)," comments scraped.\n")

    nb_scraped_comments = 0

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

    # ADD VECTOR IDs with new id_ key
    for sc, vid, kws in zip(reddit_comments_scrapes, vector_ids, comment_keywords):
        sc['id_'] = -1
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
            
print("End.")