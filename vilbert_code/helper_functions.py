import torch
from torch import nn
from pytorch_transformers import BertModel
from torchvision import transforms, models

import numpy as np
import os
import pandas as pd
import json
import re
from PIL import Image

from torch.utils.data import DataLoader, Dataset

from urlextract import URLExtract

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url_extractor = URLExtract()

transform_config = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])


## Remove urls and mentions
def process_tweet(tweet):

    urls = url_extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, ' ')

    tweet = re.sub(r'@[a-zA-Z0-9]+', ' ', tweet)

    return tweet

    
def extract_imgnet_feats(img_model, img_id):

    feats, boxes = [], []

    img = Image.open(data_loc+'images/'+img_id+'.png').convert('RGB')
    resz = transforms.Resize(256)
    img = resz(img)
    width, height = float(img.size[0]), float(img.size[1])
    for i in range(101):
        crop = transforms.RandomCrop((224,224))
        params = crop.get_params(img, (224,224))
        img_crop = transforms.functional.crop(img, *params)
        y, x, h, w = params
        img_tensor = transform_config(img_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out = img_model(img_tensor).cpu().numpy().flatten()

        feats.append(out)
        box = [x/width, y/height, (x+w-1)/width, (y+h-1)/(height), w*h/(width*height)]
        boxes.append(box)

    return np.array(feats), np.array(boxes)



class CustomDataset(Dataset):
    def __init__(self, dset, tokenizer, img_feat_header, missing_feats, indx_df, data_dict, max_seq_length=96, max_region_num=101):
        super().__init__()

        self.tokenizer = tokenizer
        self.image_feature_reader = img_feat_header
        self.missing_feats = missing_feats
        self.max_seq_length = max_seq_length
        self.padding_index = 0
        self.max_region_num = max_region_num
        self.indx_df = indx_df
        self.text_dict = data_dict
        self.dset = dset

    def __len__(self):
        return len(self.indx_df)


    def tokenize(self, text):

        tokens = self.tokenizer.encode(text)
        tokens = tokens[: self.max_seq_length-2]
        tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        if len(tokens) < self.max_seq_length:
            # Note here we pad in front of the sentence
            padding = [self.padding_index] * (self.max_seq_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        tokens = torch.from_numpy(np.array(tokens))
        segment_ids = torch.from_numpy(np.array(segment_ids))
        input_mask = torch.from_numpy(np.array(input_mask))

        return tokens, segment_ids, input_mask

    def __getitem__(self, index):
        art_id, img_id = str(self.indx_df.iloc[index, 0]), str(self.indx_df.iloc[index, 0])
        label = int(self.indx_df.iloc[index, 1])

        ## First get text tokens, ids and mask for bert
        text = self.text_dict[art_id]['text_en'] if 'ar' in self.dset  else self.text_dict[art_id]['full_text']
        text = self.process_tweet(text)

        tokens, segment_ids, input_mask = self.tokenize(text)

        ## Get image features and boxes
        try:
            features, num_boxes, boxes, _ = self.image_feature_reader[img_id]
        except:
            features = np.array(self.missing_feats['feats'][img_id])
            num_boxes = 101
            boxes = np.array(self.missing_feats['boxes'][img_id])

        mix_num_boxes = min(int(num_boxes), self.max_region_num)
        mix_boxes_pad = np.zeros((self.max_region_num, 5))
        mix_features_pad = np.zeros((self.max_region_num, 2048))

        ## Compute image relevant masks and co-attention mask
        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        co_attention_mask = torch.zeros((self.max_region_num, self.max_seq_length))

        return tokens, features, spatials, segment_ids, input_mask, image_mask, \
            co_attention_mask, label 




