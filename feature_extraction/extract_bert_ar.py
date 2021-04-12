import sys, os
import json, re
import pandas as pd
import numpy as np
import string

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

from arabert.preprocess import ArabertPreprocessor

from helper_functions import *

from urlextract import URLExtract

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

url_extractor = URLExtract()


## Remove urls and mentions
def process_tweet(tweet):

    urls = url_extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, ' ')

    tweet = re.sub(r'@[a-zA-Z0-9]+', ' ', tweet)

    return tweet

    

arabert_prep = ArabertPreprocessor(model_name='bert-base-arabertv2')

data_dict = json.load(open('data/clef/arabic/new_lists/data.json', 'r'))

## AraBERT - https://huggingface.co/aubmindlab/bert-base-arabertv02
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2', use_fast=False)
model = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2', output_hidden_states=True)
model.to(device).eval()

embed_dict = {'sent_word_catavg':{}, 'sent_word_sumavg': {}, 'sent_emb_2_last': {},
                    'sent_emb_last': {}}

for text_type in [(arabert_prep.preprocess, 'pp'), (None, 'raw')]:
    if not os.path.exists('features/text/clef_ar_arabert_%s'%(text_type[1])):
        os.makedirs('features/text/clef_ar_arabert_%s'%(text_type[1]))

    for i, idx in enumerate(data_dict):
        print(i)
        
        text = data_dict[idx]['full_text']
        text = process_tweet(text)

        if text_type[0]:
            text = arabert_prep.preprocess(text)

        sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last \
            = get_word_sent_embedding(text, model, tokenizer, device)
        
        embed_dict['sent_word_catavg'][str(idx)] = sent_word_catavg.tolist()
        embed_dict['sent_word_sumavg'][str(idx)] = sent_word_sumavg.tolist()
        embed_dict['sent_emb_2_last'][str(idx)] = sent_emb_2_last.tolist()
        embed_dict['sent_emb_last'][str(idx)] = sent_emb_last.tolist()


    json.dump(embed_dict['sent_word_catavg'], open(os.path.join('features/text/clef_ar_arabert_%s'%(text_type[1]), 'catavg.json'), 'w'))
    json.dump(embed_dict['sent_word_sumavg'], open(os.path.join('features/text/clef_ar_arabert_%s'%(text_type[1]), 'sumavg.json'), 'w'))
    json.dump(embed_dict['sent_emb_2_last'], open(os.path.join('features/text/clef_ar_arabert_%s'%(text_type[1]), 'last2.json'), 'w'))
    json.dump(embed_dict['sent_emb_last'], open(os.path.join('features/text/clef_ar_arabert_%s'%(text_type[1]), 'last.json'), 'w'))



## Arabic BERT - https://huggingface.co/asafaya/bert-base-arabic
tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-base-arabic')
model = AutoModel.from_pretrained('asafaya/bert-base-arabic', output_hidden_states=True)
model.to(device).eval()

embed_dict = {'sent_word_catavg':{}, 'sent_word_sumavg': {}, 'sent_emb_2_last': {},
                    'sent_emb_last': {}}

if not os.path.exists('features/text/clef_ar_bertarabic'):
    os.makedirs('features/text/clef_ar_bertarabic')

for i, idx in enumerate(data_dict):
    print(i)
    
    text = data_dict[idx]['full_text']

    sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last \
                = get_word_sent_embedding(text, model, tokenizer, device)
    
    embed_dict['sent_word_catavg'][str(idx)] = sent_word_catavg.tolist()
    embed_dict['sent_word_sumavg'][str(idx)] = sent_word_sumavg.tolist()
    embed_dict['sent_emb_2_last'][str(idx)] = sent_emb_2_last.tolist()
    embed_dict['sent_emb_last'][str(idx)] = sent_emb_last.tolist() 


json.dump(embed_dict['sent_word_catavg'], open(os.path.join('features/text/clef_ar_bertarabic','catavg.json'), 'w'))
json.dump(embed_dict['sent_word_sumavg'], open(os.path.join('features/text/clef_ar_bertarabic','sumavg.json'), 'w'))
json.dump(embed_dict['sent_emb_2_last'], open(os.path.join('features/text/clef_ar_bertarabic','last2.json'), 'w'))
json.dump(embed_dict['sent_emb_last'], open(os.path.join('features/text/clef_ar_bertarabic','last.json'), 'w'))
