from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
from helper_functions import *

import argparse

parser = argparse.ArgumentParser(description='Extract VilBERT feats')
parser.add_argument('--model', type=str, default='multi_task',
                    help='use model type name saved in pretrained directory')
parser.add_argument('--dset', type=str, default='clef_en',
                    help='clef_en | clef_ar | mediaeval | lesa')
args = parser.parse_args()



def tokenize(self, max_length=96, padding_index=0):

    tokens = tokenizer.encode(text)
    tokens = tokens[: max_length-2]
    tokens = tokenizer.add_special_tokens_single_sentence(tokens)

    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)

    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [padding_index] * (max_length - len(tokens))
        tokens = tokens + padding
        input_mask += padding
        segment_ids += padding

    tokens = torch.from_numpy(np.array(tokens))
    segment_ids = torch.from_numpy(np.array(segment_ids))
    input_mask = torch.from_numpy(np.array(input_mask))

    return tokens, segment_ids, input_mask


vilbert_type = args.model
dset = args.dset

## Initialize VilBERT
from vilbert.vilbert import VILBertForVLTasks
from vilbert.vilbert import BertConfig
if vilbert_type == 'concap8':
    config = BertConfig.from_json_file("vilbert-multi-task/config/bert_base_8layer_8conect.json")
else:
    config = BertConfig.from_json_file("vilbert-multi-task/config/bert_base_6layer_6conect.json")
model = VILBertForVLTasks.from_pretrained(
            'vilbert-multi-task/data/pretrained/%s_model.bin'%(vilbert_type),
            config=config,
            num_labels=2,
            default_gpu=False,
        )

model.eval()
model.to(device)

## Bert Base tokenizer for Input IDs
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
max_seq_length = 96

data_loc = 'data/%s/'%(args.dset)
save_loc = 'features/mult/'

##ImgNet Model for images where no objects detected
img_model = models.resnet152(pretrained=True)
img_model = nn.Sequential(*list(img_model.children())[:-1])
img_model.eval()
img_model.to(device)
##--------------------------------------------------

## Training, Validation and Test Text
data_dict = json.load(open(data_loc+'data.json', 'r', encoding='utf-8'))

visual_feat_dict = {'pooled': {}, 'average': {}}
text_feat_dict = {'pooled': {}, 'average': {}}

## Extracted Faster-RCNN Features
image_feature_reader = ImageFeaturesH5Reader(data_loc+'rcnn_lmdbs/')
max_region_num = 101

for i, idx in enumerate(data_dict):
    print(i)
    txt_id, img_id = idx, idx

    if 'ar' in dset:
        text = process_tweet(data_dict[txt_id]['text_en'])
    else:
        text = process_tweet(data_dict[txt_id]['full_text'])

    ## First get text tokens, ids and mask for bert
    tokens, segment_ids, input_mask = tokenize(text)
    
    ## Get image features and boxes
    try:
        features, num_boxes, boxes, _ = image_feature_reader[img_id]
    except:
        num_boxes = 101
        features, boxes = extract_imgnet_feats(img_model, img_id, dset)

    mix_num_boxes = min(int(num_boxes), max_region_num)
    mix_boxes_pad = np.zeros((max_region_num, 5))
    mix_features_pad = np.zeros((max_region_num, 2048))

    ## Compute image relevant masks and co-attention mask
    image_mask = [1] * (int(mix_num_boxes))
    while len(image_mask) < max_region_num:
        image_mask.append(0)

    mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
    mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

    features = torch.tensor(mix_features_pad).float()
    image_mask = torch.tensor(image_mask).long()
    spatials = torch.tensor(mix_boxes_pad).float()

    co_attention_mask = torch.zeros((max_region_num, max_seq_length))

    ## Expand sizes and move to gpu
    features = features.view(-1, features.size(0), features.size(1)).to(device)
    spatials = spatials.view(-1, spatials.size(0), spatials.size(1)).to(device)
    image_mask = image_mask.view(-1, image_mask.size(0)).to(device)
    tokens = tokens.view(-1, tokens.size(0)).to(device)
    input_mask = input_mask.view(-1, input_mask.size(-0)).to(device)
    segment_ids = segment_ids.view(-1, segment_ids.size(0)).to(device)
    co_attention_mask = co_attention_mask.view(
        -1, co_attention_mask.size(0), co_attention_mask.size(1)).to(device)

    task_tokens = tokens.new().resize_(tokens.size(0), 1).fill_(14).to(device)
    
    ## Extract Features
    with torch.no_grad():
        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = model.bert(
            tokens, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens)

    average_output_t = np.mean(sequence_output_t.squeeze(0).cpu().numpy(), axis=0).flatten().tolist()
    average_output_v = np.mean(sequence_output_v.squeeze(0).cpu().numpy(), axis=0).flatten().tolist()
    pooled_output_t = pooled_output_t.cpu().numpy().flatten().tolist()
    pooled_output_v = pooled_output_v.cpu().numpy().flatten().tolist()

    visual_feat_dict['pooled'][img_id] = pooled_output_v
    visual_feat_dict['average'][img_id] = average_output_v
    text_feat_dict['pooled'][txt_id] = pooled_output_t
    text_feat_dict['average'][txt_id] = average_output_t

json.dump(visual_feat_dict, open(save_loc+'%s_vilbert_%s_viz.json'%(dset, vilbert_type), 'w'))
json.dump(text_feat_dict, open(save_loc+'%s_vilbert_%s_text.json'%(dset, vilbert_type), 'w'))
