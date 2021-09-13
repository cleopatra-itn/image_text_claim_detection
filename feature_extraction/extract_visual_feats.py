from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import torch
import torch.nn.functional as F
from torch import nn

import os, re
import numpy as np
from PIL import Image
import pandas as pd
import json

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse

parser = argparse.ArgumentParser(description='Extract Visual Features for Tweet Images')
parser.add_argument('--vmodel', '-v', type=str, default='resnet152',
                    help='resnet50 | resnet101 | resnet152')
parser.add_argument('--vtype', '-t', type=str, default='imgnet',
                    help='imgnet | plc | hybrid')
parser.add_argument('--dset', '-d', type=str, default='clef_en',
                    help='clef_en | clef_ar | mediaeval | lesa')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_config = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.file_names = np.array(os.listdir(root))
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        im_name = self.file_names[idx]

        img = Image.open(os.path.join(self.root, im_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)    

        return im_name.strip('.png'), img



def get_visual_feats():
    print(vtype, vmodel, dset)
    feats, logits, im_names = [], [], []

    def feature_hook(module, input, output):
        return feats.extend(output.view(-1,output.shape[1]).data.cpu().numpy().tolist())

    if vtype == 'imgnet':
        model = models.__dict__[vmodel](pretrained=True)
    elif vtype == ['plc', 'hybrid']:
        model_file = 'pretrained_models/%s_places_best.pth.tar'%(vmodel) if vtype == 'plc' else \
            'pretrained_models/%s_hybrid_best.pth.tar'%(vmodel)
        model = models.__dict__[vmodel](num_classes=365)  if vtype == 'plc' else \
            models.__dict__[vmodel](num_classes=1365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

    model.eval().to(device)
    model._modules.get('avgpool').register_forward_hook(feature_hook)

    dataset = CustomDataset(dloc, transform_config)
    dt_loader = DataLoader(dataset, batch_size=32, num_workers=2)

    for i, batch in enumerate(dt_loader):
        print(i)
        im_nms, images = batch

        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
        
        logits.extend(outputs.view(-1,outputs.shape[1]).data.cpu().numpy().tolist())

        im_names.extend(im_nms)

    return feats, logits, im_names


dset = args.dset
vmodel = args.vmodel
vtype = args.vtype

if dset == 'clef_en':
    dloc = 'data/clef_en/images/'
elif dset == 'clef_ar':
    dloc = 'data/clef_ar/images/'
elif dset == 'mediaeval':
    dloc = 'data/mediaeval/images/'
else:
    dloc = 'data/lesa/images/'


## Extract and Save Features
feat_dict = {}
feats, logits, im_names = get_visual_feats()

feat_dict['feats'] = {name:feat for name,feat in zip(im_names, feats)}
feat_dict['logits'] = {name:feat for name,feat in zip(im_names, logits)}

output_loc = 'features/image/'
if not os.path.exists(output_loc):
        os.makedirs(output_loc)

json.dump(feat_dict, open(output_loc+'/%s_%s_%s.json'%(dset, vtype, vmodel), 'w'))
