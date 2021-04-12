import sys, os
import json, re
import pandas as pd
import numpy as np
import string

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

from helper_functions import *
from preprocess_covidbert import *

import argparse

parser = argparse.ArgumentParser(description='Training SVM')
parser.add_argument('--dset', type=str, default='clef_en',
                    help='clef_en | clef_ar | mediaeval | lesa')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


text_processor = get_text_processor(word_stats='twitter')
        
def process_tweet(tweet):

    proc_tweet = text_processor.pre_process_doc(tweet)

    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]

    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)


text_processor_ht = get_text_processor_hashtag(word_stats='twitter')
        
def process_tweet_ht(tweet):

    proc_tweet = text_processor_ht.pre_process_doc(tweet)

    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]

    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)


dset = args.dset
if dset == 'clef_en':
    data_loc = 'data/clef/english/'
elif dset == 'mediaeval':
    data_loc = 'data/mediaeval'
else:
    data_loc = 'data/lesa/'

for dt_name in [(dset, data_loc)]:

    data_dict = json.load(open('data/'+dt_name[1]+'new_lists/data.json', 'r'))

    ## English BERT base models
    for bert_type in [(BertModel,    BertTokenizer,    'bert-base-uncased')]:
        tokenizer = bert_type[1].from_pretrained(bert_type[2], use_fast=False)
        model = bert_type[0].from_pretrained(bert_type[2], output_hidden_states=True)
        model.to(device).eval()
        
        for text_type in [(process_tweet, 'clean'), (process_tweet_ht, 'ht'), (None, 'raw')]:
            embed_dict = {'sent_word_catavg':{}, 'sent_word_sumavg': {}, 'sent_emb_2_last': {},
                                'sent_emb_last': {}}


            if not os.path.exists('features/text/%s_%s_%s'%(dt_name[0], bert_type[2], text_type[1])):
                os.makedirs('features/text/%s_%s_%s'%(dt_name[0], bert_type[2], text_type[1]))

            for i, idx in enumerate(data_dict):
                print(i)
                
                text = data_dict[idx]['full_text']

                if text_type[0]:
                    text = text_type[0](text)

                sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last \
                    = get_word_sent_embedding(text, model, tokenizer, device)
                
                embed_dict['sent_word_catavg'][str(idx)] = sent_word_catavg.tolist()
                embed_dict['sent_word_sumavg'][str(idx)] = sent_word_sumavg.tolist()
                embed_dict['sent_emb_2_last'][str(idx)] = sent_emb_2_last.tolist()
                embed_dict['sent_emb_last'][str(idx)] = sent_emb_last.tolist() 

            
            json.dump(embed_dict['sent_word_catavg'], open(os.path.join('features/text/%s_%s_%s'%
                    (dt_name[0], bert_type[2], text_type[1]), 'catavg.json'), 'w'))
            json.dump(embed_dict['sent_word_sumavg'], open(os.path.join('features/text/%s_%s_%s'%
                    (dt_name[0], bert_type[2], text_type[1]), 'sumavg.json'), 'w'))
            json.dump(embed_dict['sent_emb_2_last'], open(os.path.join('features/text/%s_%s_%s'%
                    (dt_name[0], bert_type[2], text_type[1]), 'last2.json'), 'w'))
            json.dump(embed_dict['sent_emb_last'], open(os.path.join('features/text/%s_%s_%s'%
                    (dt_name[0], bert_type[2], text_type[1]), 'last.json'), 'w'))



    ## Twitter English BERT model
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True, use_fast=False)
    model = AutoModel.from_pretrained('vinai/bertweet-base', output_hidden_states=True)
    model.to(device).eval()
    
    embed_dict = {'sent_word_catavg':{}, 'sent_word_sumavg': {}, 'sent_emb_2_last': {},
                        'sent_emb_last': {}}

    if not os.path.exists('features/text/%s_bertweet'%(dt_name[0])):
        os.makedirs('features/text/%s_bertweet'%(dt_name[0]))

    for i, idx in enumerate(data_dict):
        print(i)
        
        text = data_dict[idx]['full_text']

        sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last \
                    = get_word_sent_embedding(text, model, tokenizer, device)
        
        embed_dict['sent_word_catavg'][str(idx)] = sent_word_catavg.tolist()
        embed_dict['sent_word_sumavg'][str(idx)] = sent_word_sumavg.tolist()
        embed_dict['sent_emb_2_last'][str(idx)] = sent_emb_2_last.tolist()
        embed_dict['sent_emb_last'][str(idx)] = sent_emb_last.tolist() 

    
    json.dump(embed_dict['sent_word_catavg'], open(os.path.join('features/text/%s_bertweet'
        %(dt_name[0]),'catavg.json'), 'w'))
    json.dump(embed_dict['sent_word_sumavg'], open(os.path.join('features/text/%s_bertweet'
        %(dt_name[0]),'sumavg.json'), 'w'))
    json.dump(embed_dict['sent_emb_2_last'], open(os.path.join('features/text/%s_bertweet'
        %(dt_name[0]),'last2.json'), 'w'))
    json.dump(embed_dict['sent_emb_last'], open(os.path.join('features/text/%s_bertweet'
        %(dt_name[0]),'last.json'), 'w'))


    ## Covid-Twitter English BERT model
    tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', use_fast=False)
    model = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', output_hidden_states=True)
    model.to(device).eval()
    
    embed_dict = {'sent_word_catavg':{}, 'sent_word_sumavg': {}, 'sent_emb_2_last': {},
                        'sent_emb_last': {}}

    if not os.path.exists('features/text/%s_covid_twitter'%(dt_name[0])):
        os.makedirs('features/text/%s_covid_twitter'%(dt_name[0]))

    for i, idx in enumerate(data_dict):
        print(i)
        
        text = data_dict[idx]['full_text']

        text = preprocess_bert(text)

        sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last \
                    = get_word_sent_embedding(text, model, tokenizer, device)
        
        embed_dict['sent_word_catavg'][str(idx)] = sent_word_catavg.tolist()
        embed_dict['sent_word_sumavg'][str(idx)] = sent_word_sumavg.tolist()
        embed_dict['sent_emb_2_last'][str(idx)] = sent_emb_2_last.tolist()
        embed_dict['sent_emb_last'][str(idx)] = sent_emb_last.tolist() 

    
    json.dump(embed_dict['sent_word_catavg'], open(os.path.join('features/text/%s_covid_twitter'
        %(dt_name[0]),'catavg.json'), 'w'))
    json.dump(embed_dict['sent_word_sumavg'], open(os.path.join('features/text/%s_covid_twitter'
        %(dt_name[0]),'sumavg.json'), 'w'))
    json.dump(embed_dict['sent_emb_2_last'], open(os.path.join('features/text/%s_covid_twitter'
        %(dt_name[0]),'last2.json'), 'w'))
    json.dump(embed_dict['sent_emb_last'], open(os.path.join('features/text/%s_covid_twitter'
        %(dt_name[0]),'last.json'), 'w'))