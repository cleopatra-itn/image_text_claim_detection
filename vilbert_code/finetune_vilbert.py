from vilbert.vilbert import VILBertForVLTasks
from vilbert.vilbert import BertConfig
from vilbert.vilbert import GeLU
from vilbert.optimization import RAdam
from torch.nn import functional as F

from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm

import torch
from torch.optim  import lr_scheduler
from torch import nn
import torch.optim as optim

import random, json
from sklearn import metrics
from itertools import combinations as combs
import copy
from sklearn.utils.class_weight import compute_class_weight

from helper_functions import *

import argparse

parser = argparse.ArgumentParser(description='Train VilBERT with pooled features')
parser.add_argument('--model', type=str, default='image_ret')
parser.add_argument('--dset', type=str, default='mediaeval',
                    help='clef_en | clef_ar | mediaeval | lesa')
parser.add_argument('--split', type=int, default=0,
                    help='0-4')
parser.add_argument('--unfr', type=int, default=2,
                    help='2 | 4 | 6 (Number of co-attention layers to un-freeze/train')
parser.add_argument('--lr', type=str, default='5e-5',
                    help='2e-5 | 3e-5 | 5e-5')
parser.add_argument('--wt_ce', type=int, default=0,
                    help='0 | 1')
parser.add_argument('--bs', type=int, default=16,
                    help='4 | 8 | 16')
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--pool', type=str, default='add')
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


## A simple classifier on top of ViLBert pooled features.
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.vilbert = vilbert
        self.dropout = nn.Dropout(dropout)
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            # GeLU(),
            nn.ReLU(inplace=True),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )
        self.pool = self.add if args.pool == 'add' else self.mul

    def add(self, a, b):
        return a+b
    
    def mul(self, a, b):
        return a*b

    def forward(self, tokens, features, spatials, segment_ids, input_masks, image_masks, co_attention_masks, task_tokens):
        _, _, pooled_output_t, pooled_output_v, _ = self.vilbert(tokens, features, spatials, segment_ids, input_masks, image_masks, co_attention_masks, task_tokens)
        pooled_out = self.pool(pooled_output_t, pooled_output_v)
        return self.logit_fc(self.dropout(pooled_out))


def evaluate(model, loader, phase):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    for i, batch in enumerate(loader):
    
        tokens, features, spatials, segment_ids, input_masks, image_masks, \
            co_attention_masks, labels = batch

        tokens, features, spatials, segment_ids, input_masks, image_masks, \
            co_attention_masks, labels = tokens.to(device), features.to(device), spatials.to(device), \
            segment_ids.to(device), input_masks.to(device), image_masks.to(device), co_attention_masks.to(device), \
                labels.to(device)

        task_tokens = tokens.new().resize_(tokens.size(0), 1).fill_(1).to(device)

        with torch.no_grad():
            logits = model(
            tokens, features, spatials, segment_ids, input_masks, image_masks, co_attention_masks, task_tokens).squeeze(1)

            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = logits.max(1)[1]

        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(loader)
    avg_acc = metrics.accuracy_score(all_labels, all_preds)
    avg_f1 = metrics.f1_score(all_labels, all_preds, average='weighted')

    return avg_acc, avg_f1, avg_loss, all_preds


## Arguments
vilbert_type = args.model
dset = args.dset
split = args.split

## Initialize VilBERT
if vilbert_type == 'concap8':
    config = BertConfig.from_json_file("vilbert-multi-task/config/bert_base_8layer_8conect.json")
else:
    config = BertConfig.from_json_file("vilbert-multi-task/config/bert_base_6layer_6conect.json")
vilbert = VILBertForVLTasks.from_pretrained(
            'vilbert-multi-task/data/pretrained/%s_model.bin'%(vilbert_type),
            config=config,
            num_labels=2,
            default_gpu=False,
        )

vilbert = vilbert.bert

## Dataset Stuff
data_loc = 'data/%s/'%(args.dset)

## Training, Validation and Test Text
data_dict = json.load(open(data_loc+'data.json', 'r', encoding='utf-8'))

train_df = pd.read_csv(data_loc+'splits/train_%d.txt'%(split), header=None)
val_df = pd.read_csv(data_loc+'splits/val.txt', header=None)
test_df = pd.read_csv(data_loc+'splits/test_%d.txt'%(split), header=None)

test_idxs = np.array([idx for idx in test_df[0]])


## Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
img_feat_header = ImageFeaturesH5Reader(data_loc+'rcnn_lmdbs/')
missing_feats = json.load(open(data_loc+'rcnn_missing/%s.json'%(dset), 'r'))

tr_data = CustomDataset(dset, tokenizer, img_feat_header, missing_feats, train_df, data_dict)
tr_loader = DataLoader(tr_data, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)

vl_data = CustomDataset(dset, tokenizer, img_feat_header, missing_feats, val_df, data_dict)
vl_loader = DataLoader(vl_data, batch_size=int(args.bs/2), num_workers=2, pin_memory=True)

te_data = CustomDataset(dset, tokenizer, img_feat_header, missing_feats, test_df, data_dict)
te_loader = DataLoader(te_data, batch_size=int(args.bs/2), num_workers=2, pin_memory=True)

num_epochs = args.epochs

model = SimpleClassifier(1024, 128, 2, 0.2)

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

if args.unfr == 2:
    not_freeze = ['c_layer.4', 'c_layer.5', 't_pooler', 'v_pooler']
elif args.unfr == 4:
    not_freeze = ['c_layer.2', 'c_layer.3', 'c_layer.4', 'c_layer.5', 't_pooler', 'v_pooler']
elif args.unfr == 6:
    not_freeze = ['c_layer.0','c_layer.1','c_layer.2', 'c_layer.3', 'c_layer.4', 'c_layer.5', 't_pooler', 'v_pooler']
else:
    not_freeze = ['t_pooler', 'v_pooler']

print(not_freeze)

for name, param in model.named_parameters():
    if not any(nf in name for nf in not_freeze):
        param.requires_grad = False

optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

print('\n---------------Training the below parameters:------------')
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print('----------------------------------------------------------\n')

model.to(device)

optimizer = AdamW(optimizer_grouped_parameters, lr=float(args.lr))
total_steps = num_epochs*len(tr_loader)
warmup_steps = int(total_steps*0.1)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps, total_steps)

if args.wt_ce:
    class_weights = compute_class_weight('balanced', classes=[0,1], y=train_df[1].to_numpy())
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
else:
    criterion = nn.CrossEntropyLoss().to(device)

best_model = model
best_acc = 0.0
best_val_loss = 100
best_epoch = 0
best_test_acc = 0.0
best_test_f1 = 0.0
best_test_loss = 0.0
best_preds = []


for epoch in range(0, num_epochs):

    print("----- Epoch %d/%d ------"%(epoch+1, num_epochs))
    
    running_loss = 0.0
    running_corrects = 0

    tot = 0.0
    
    model.train()
    for i, batch in enumerate(tr_loader):
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        tokens, features, spatials, segment_ids, input_masks, image_masks, \
            co_attention_masks, labels = batch

        tokens, features, spatials, segment_ids, input_masks, image_masks, \
            co_attention_masks, labels = tokens.to(device), features.to(device), spatials.to(device), \
            segment_ids.to(device), input_masks.to(device), image_masks.to(device), co_attention_masks.to(device), \
                labels.to(device)

        task_tokens = tokens.new().resize_(tokens.size(0), 1).fill_(1).to(device)

        logits = model(
            tokens, features, spatials, segment_ids, input_masks, image_masks, co_attention_masks, task_tokens).squeeze(1)

        preds = logits.max(1)[1]

        loss = criterion(logits, labels)

        # backward + optimize
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

        # statistics
        running_loss += loss.item()
        running_corrects += preds.eq(labels.view_as(preds)).sum().item()
        tot += len(labels)

        if i % 20 == 0:
            print('[%d, %5d] loss: %.5f, Acc: %.2f' %
                    (epoch+1, i + 1, loss.item(), (100.0 * running_corrects) / tot))


    train_loss = running_loss / len(tr_loader)
    train_acc = running_corrects * 1.0 / (len(tr_loader.dataset))

    print('Training Loss: {:.6f} Acc: {:.2f}'.format(train_loss, 100.0 * train_acc))

    val_acc, val_f1, val_loss, _ = evaluate(model, vl_loader, 'val')

    print('Epoch: {:d}, Val Loss: {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}'.format(epoch+1, 
                                        val_loss,val_acc, val_f1))

    test_acc, test_f1, test_loss, test_preds = evaluate(model, te_loader, 'test')
    print('Epoch: {:d}, Test Loss: {:.4f}, Test Acc: {:.4f}, Test F1: {:.4f}'.format(epoch+1, 
                                        test_loss, test_acc, test_f1))
    

    # deep copy the model
    if val_acc >= best_acc:
        best_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch
        best_test_acc = test_acc
        best_test_f1 = test_f1
        best_test_loss = test_loss
        best_preds = test_preds


print('Best Epoch: {} : Test Loss: {:.4f} : Test Acc: {:.4f} : Test F1: {:.4f}'.format(best_epoch+1, best_test_loss, best_test_acc, best_test_f1))