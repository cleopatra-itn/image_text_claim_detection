import sys, os
import json, re
import pandas as pd
import numpy as np
import string

from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import torch

from helper_functions import *
from preprocess_covidbert import *

import argparse

parser = argparse.ArgumentParser(description='Extract English Tweet Sentence Embeddings')
parser.add_argument('--dtype', '-d', type=str, default='clef_en',
                    help='clef_en | mediaeval | lesa')
parser.add_argument('--mtype','-m', type=str, default='bertbase',
                    help='bertbase | bertweet | covidbert')
parser.add_argument('--proc','-p', type=str, default='clean',
                    help='clean | raw')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


text_processor = get_text_processor(word_stats='twitter', keep_hashtags=True)
        
def process_tweet(tweet):

    proc_tweet = text_processor.pre_process_doc(tweet)

    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]

    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)


dtype = args.dtype
mtype = args.mtype
proc = args.proc

if dtype == 'clef_en':
    data_loc = 'data/clef_en/'
elif dtype == 'mediaeval':
    data_loc = 'data/mediaeval/'
elif dtype == 'lesa':
    data_loc = 'data/lesa/'


data_dict = json.load(open(data_loc+'data.json', 'r'))

if mtype == 'bertbase':
    preproc_func = process_tweet if proc == 'clean' else lambda x: x
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
elif mtype == 'bertweet':
    ## Bertweet - https://huggingface.co/vinai/bertweet-base
    preproc_func = process_tweet if proc == 'clean' else lambda x: x
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True, use_fast=False)
    model = AutoModel.from_pretrained('vinai/bertweet-base', output_hidden_states=True)
elif mtype == 'covidbert':
    ## Covid-Twitter BERT - https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2
    preproc_func = preprocess_bert if proc == 'clean' else lambda x: x
    tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', use_fast=False)
    model = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', output_hidden_states=True)


model.to(device).eval()

embed_dict = {'sent_word_catavg':{}, 'sent_word_sumavg': {}, 'sent_emb_2_last': {},
                    'sent_emb_last': {}}

output_loc = 'features/text/%s_%s_%s'%(dtype, mtype, proc)
if not os.path.exists(output_loc):
        os.makedirs(output_loc)


for i, idx in enumerate(data_dict):
    print(i)
    
    text = data_dict[idx]['full_text']

    proc_text = preproc_func(text)

    sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last \
        = get_word_sent_embedding(proc_text, model, tokenizer, device)
    
    embed_dict['sent_word_catavg'][str(idx)] = sent_word_catavg.tolist()
    embed_dict['sent_word_sumavg'][str(idx)] = sent_word_sumavg.tolist()
    embed_dict['sent_emb_2_last'][str(idx)] = sent_emb_2_last.tolist()
    embed_dict['sent_emb_last'][str(idx)] = sent_emb_last.tolist() 


json.dump(embed_dict['sent_word_catavg'], open(os.path.join(output_loc, 'catavg.json'), 'w'))
json.dump(embed_dict['sent_word_sumavg'], open(os.path.join(output_loc, 'sumavg.json'), 'w'))
json.dump(embed_dict['sent_emb_2_last'], open(os.path.join(output_loc, 'last2.json'), 'w'))
json.dump(embed_dict['sent_emb_last'], open(os.path.join(output_loc, 'last.json'), 'w'))