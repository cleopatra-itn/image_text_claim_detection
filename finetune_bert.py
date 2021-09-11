from transformers import BertTokenizer, BertModel, RobertaConfig, BertConfig, AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup, AdamW
import transformers

from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch
from torch import nn

import re, os, random, copy, json
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics

from preprocess_covidbert import *
from arabert.preprocess import ArabertPreprocessor

from helper_functions import *

from urlextract import URLExtract

import argparse

parser = argparse.ArgumentParser(description='Training Transformer')
parser.add_argument('--btype', type=str, default='bert-base-uncased',
                    help='bert-base-uncased | bertweet | covid_twitter | arabert | bertarabic')           
parser.add_argument('--ttype', type=str, default='clean',
                    help='clean | ht | raw | pp (arabic)')
parser.add_argument('--dset', type=str, default='lesa',
                    help='clef_en | clef_ar | mediaeval | lesa')
parser.add_argument('--freeze', type=int, default=1,
                    help='True (1) | False (0)')
parser.add_argument('--fr_no', type=int, default=8,
                    help='0-12 | Number of encoder layers to freeze')
parser.add_argument('--wt_ce', type=int, default=0,
                    help='True (1) | False (0)')
parser.add_argument('--lr', type=str, default='5e-5',
                    help='5e-5 | 3e-5 (Default) | 2e-5')
parser.add_argument('--epochs', type=int, default=6,
                    help='Number of epochs to train the model')
parser.add_argument('--bs', type=int, default=16,
                    help='4 (for Clef_en), 8, 16')
parser.add_argument('--split', type=int, default=0,
                    help='0-4')
parser.add_argument('--gpu', type=int, default=0,
                    help='0,1,2,3')


args = parser.parse_args()  

seed = 42
transformers.set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:%d"%(args.gpu) if torch.cuda.is_available() else "cpu")

if args.dset == 'clef_ar':
    arabert_prep = ArabertPreprocessor(model_name='bert-base-arabertv2')

url_extractor = URLExtract()

class CustomDataset(Dataset):
    def __init__(self, data_dict, indx_df, transform=None):
        self.indx_df = indx_df
        self.text_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.indx_df)

    def __getitem__(self, idx):
        txt_id = str(self.indx_df.iloc[idx,0])
        label = int(self.indx_df.iloc[idx,1])

        text = self.text_dict[txt_id]['full_text']
    
        if self.transform:
            text = self.transform(text)  

        return text, label


def process_tweet_en(tweet):
    proc_tweet = text_processor.pre_process_doc(tweet)

    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]

    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)


## Remove urls and mentions
def process_tweet_ar(tweet):

    urls = url_extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, ' ')

    tweet = re.sub(r'@[a-zA-Z0-9]+', ' ', tweet)

    return tweet


## Remove urls and mentions
def process_tweet_arabert(tweet):

    urls = url_extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, ' ')

    tweet = re.sub(r'@[a-zA-Z0-9]+', ' ', tweet)

    tweet = arabert_prep.preprocess(tweet)

    return tweet


def get_accuracy(preds, gt_labs):
    preds = np.argmax(preds, axis=1).flatten()
    gt_labs = gt_labs.flatten()
    return np.sum(preds == gt_labs) / len(gt_labs)


## Custom Architecture
class Bert_Clf(nn.Module):
    def __init__(self, bert, dim):
        super(Bert_Clf, self).__init__()

        self.bert = bert

        self.dp = nn.Dropout(0.2)
        self.bn = nn.LayerNorm(dim)

        self.cf = nn.Linear(dim, 2)

    def forward(self, inp_ids, masks):
        _, cls_out = self.bert(inp_ids, attention_mask=masks, return_dict=False)

        cls_out = self.dp(cls_out)

        return self.cf(cls_out)


## Argument Inputs 
dset = args.dset
btype = args.btype
num_epochs = args.epochs
init_lr = float(args.lr)
batch_size = args.bs
freeze = args.freeze
fr_no = args.fr_no
wt_ce = args.wt_ce
ttype = args.ttype
split = args.split

if btype == 'arabert' and ttype == 'pp':
    print('hi')
    transform_config = process_tweet_arabert
    tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2', use_fast=False)
    bert = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')
    dim = 768
elif btype == 'arabert' and ttype == 'raw':
    transform_config = process_tweet_ar
    tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2', use_fast=False)
    bert = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')
    dim = 768
elif btype == 'bertarabic':
    transform_config = process_tweet_ar
    tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-base-arabic')
    bert = AutoModel.from_pretrained('asafaya/bert-base-arabic')
    dim = 768
elif btype == 'covid_twitter':
    transform_config = preprocess_bert
    tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', use_fast=False)
    bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
    dim = 1024
elif btype == 'bertweet':
    transform_config = None
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True, use_fast=False)
    bert = AutoModel.from_pretrained('vinai/bertweet-base')
    dim = 768
elif btype == 'bert-base-uncased' and ttype == 'raw':
    transform_config = None
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
    bert = BertModel.from_pretrained('bert-base-uncased')
    dim = 768
elif btype == 'bert-base-uncased' and ttype == 'ht':
    text_processor = get_text_processor_hashtag(word_stats='twitter')
    transform_config = process_tweet_en
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
    bert = BertModel.from_pretrained('bert-base-uncased')
    dim = 768
elif btype == 'bert-base-uncased' and ttype == 'clean':
    text_processor = get_text_processor(word_stats='twitter')
    transform_config = process_tweet_en
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
    bert = BertModel.from_pretrained('bert-base-uncased')
    dim = 768


## Freeze Bert parameters
if freeze:
    vec = list(range(0,args.fr_no))
    for param in bert.embeddings.parameters():
        param.requires_grad = False
    for name, param in bert.encoder.named_parameters():
        if int(re.findall(r'[0-9]+', name)[0]) in vec:
            param.requires_grad = False


## Initialize model
model = Bert_Clf(bert, dim)
model = nn.DataParallel(model)
model.to(device)


if dset == 'clef_en':
    data_loc = 'data/clef/english/'
elif dset == 'clef_ar':
    data_loc = 'data/clef/arabic/'
elif dset == 'mediaeval':
    data_loc = 'data/mediaeval/'
else:
    data_loc = 'data/lesa/'


## Training, Validation and Test Text
data_dict = json.load(open(data_loc+'new_lists/data.json', 'r', encoding='utf-8'))

train_df = pd.read_csv(data_loc+'splits/train_%d.txt'%(split), header=None)
val_df = pd.read_csv(data_loc+'splits/val.txt', header=None)
test_df = pd.read_csv(data_loc+'splits/test_%d.txt'%(split), header=None)

## Get maximum sentence length
tr_data = CustomDataset(data_dict, train_df, transform_config)
tr_loader = DataLoader(tr_data, batch_size=1)

max_len = 0
all_tr_labs = []
for batch in tr_loader:
    text_inps, lab = batch

    input_ids = tokenizer.encode(text_inps[0], add_special_tokens=True)

    max_len = max(max_len, len(input_ids))

    all_tr_labs.append(lab[0].item())


## Weights for weighted cross entropy loss
class_weights = compute_class_weight('balanced', classes=[0,1], y=all_tr_labs)
class_weights = torch.tensor(class_weights,dtype=torch.float)
class_weights = class_weights.to(device)

## Encode whole dataset
proc_dsets = []
for phase in [train_df, val_df, test_df]:
    ph_data = CustomDataset(data_dict, phase, transform_config)
    ph_loader = DataLoader(ph_data, batch_size=1)

    inp_ids = []
    attn_masks = []
    labels = []

    for batch in ph_loader:
        text, lab = batch
        enc_dict = tokenizer.encode_plus(
                        text[0],                      
                        add_special_tokens=True, 
                        padding = 'max_length',
                        max_length = max_len, 
                        truncation=True,     
                        return_attention_mask = True, 
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
        inp_ids.append(enc_dict['input_ids'])
        attn_masks.append(enc_dict['attention_mask'])
        labels.append(lab[0])

    inp_ids = torch.cat(inp_ids, dim=0)
    attn_masks = torch.cat(attn_masks, dim=0)
    labels = torch.tensor(labels)

    proc_dsets.append(TensorDataset(inp_ids, attn_masks, labels))


def evaluate(model, loader, phase):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        inp_ids, inp_mask, labs = batch[0].to(device), \
                    batch[1].to(device), batch[2].to(device)

        with torch.no_grad():
            logits = model(inp_ids, inp_mask)

        loss = criterion(logits, labs)

        total_loss += loss.item()

        preds = torch.argmax(logits.data, 1)

        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labs.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(loader)
    avg_acc = metrics.accuracy_score(all_labels, all_preds)
    avg_f1 = metrics.f1_score(all_labels, all_preds, average='weighted')

    print('%s Loss: %.4f : %s Acc: %.4f : F1: %.4f'%(phase, avg_loss, phase, avg_acc, avg_f1))

    return avg_acc, avg_f1, avg_loss



## Training

## DataLoaders
tr_loader = DataLoader(proc_dsets[0], sampler=RandomSampler(proc_dsets[0]), batch_size=batch_size)
vl_loader = DataLoader(proc_dsets[1], sampler=SequentialSampler(proc_dsets[1]), batch_size=batch_size)
te_loader = DataLoader(proc_dsets[2], sampler=SequentialSampler(proc_dsets[2]), batch_size=batch_size)

no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=init_lr, eps=1e-8)

print('\n---------------Training the below parameters:------------')
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
        params_to_update.append(param)
print('----------------------------------------------------------\n')


total_steps = len(tr_loader) * num_epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                    num_warmup_steps = int(total_steps*0.1),
                    num_training_steps = total_steps)


if args.wt_ce:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()


best_epoch = 0
best_vl_acc = 0
best_vl_loss = 100
best_model = 0

for epoch in range(0, num_epochs):

    print("----- Epoch %d/%d ------"%(epoch+1, num_epochs))
    
    total_tr_loss = 0
    
    model.train()

    for idx, batch in enumerate(tr_loader):

        inp_ids, inp_mask, labs = batch[0].to(device), \
                    batch[1].to(device), batch[2].to(device) 
        
        optimizer.zero_grad()

        logits = model(inp_ids, inp_mask)

        loss = criterion(logits, labs)

        total_tr_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        if idx%40 == 0:
            logits = logits.detach().cpu().numpy()
            print('Batch %d of %d, Train Acc: %.4f'%(idx, len(tr_loader), 
                        get_accuracy(logits, labs.cpu().numpy())))

    
    avg_tr_loss = total_tr_loss/len(tr_loader)

    print('Avg. Train loss: %.4f'%(avg_tr_loss))

    val_acc, _, val_loss = evaluate(model, vl_loader, 'Val')

    if val_acc >= best_vl_acc:
        best_vl_acc = val_acc
        best_vl_loss = val_loss
        best_epoch = epoch
        best_model = copy.deepcopy(model)

    print()


print("----------- All Set Results------------\n")
evaluate(best_model, tr_loader, 'Train')
evaluate(best_model, vl_loader, 'Val')
## Accuracy on Test Set
test_acc, test_f1, test_loss = evaluate(best_model, te_loader, 'Test')

print('Best Epoch: %d : Acc: %.4f : F1: %.4f'%(best_epoch, test_acc, test_f1))
print("---------------------------------------\n")

# ## Save Model
# output_loc = 'models/text/%s_%s_%s'%(mvsa, args.ttype, bert_list[btype])

# torch.save(model.state_dict(), output_loc+'.pt')
# torch.save(args, output_loc+'_args.bin')
