import sys, os
import json, re
import pandas as pd
import numpy as np
import string

from transformers import AutoModel, AutoTokenizer
import torch

from arabert.preprocess import ArabertPreprocessor

from helper_functions import *

from urlextract import URLExtract

import argparse

parser = argparse.ArgumentParser(description='Extract Arabic Tweet Sentence Embeddings')
parser.add_argument('--mtype','-m', type=str, default='arabert',
                    help='arabert | arabicbert')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

url_extractor = URLExtract()


## Remove urls and mentions
def process_tweet(tweet):

    urls = url_extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, ' ')

    tweet = re.sub(r'@[a-zA-Z0-9]+', ' ', tweet)

    return tweet

    
mtype = args.mtype
if mtype == 'arabert':
    ## AraBERT - https://huggingface.co/aubmindlab/bert-base-arabertv02
    preproc_ar = ArabertPreprocessor(model_name='bert-base-arabertv2')
    tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2', use_fast=False)
    model = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2', output_hidden_states=True)
else:
    ## Arabic BERT - https://huggingface.co/asafaya/bert-base-arabic
    preproc_ar = lambda x: x
    tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-base-arabic', use_fast=False)
    model = AutoModel.from_pretrained('asafaya/bert-base-arabic', output_hidden_states=True)
    


model.to(device).eval()

data_dict = json.load(open('data/clef_ar/data.json', 'r'))

embed_dict = {'sent_word_catavg':{}, 'sent_word_sumavg': {}, 'sent_emb_2_last': {},
                    'sent_emb_last': {}}

output_loc = 'features/text/clef_ar_%s'%(mtype)
if not os.path.exists(output_loc):
        os.makedirs(output_loc)

for i, idx in enumerate(data_dict):
    print(i)
    
    text = data_dict[idx]['full_text']
    text = process_tweet(text)

    proc_text = preproc_ar.preprocess(text)

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
