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

parser = argparse.ArgumentParser(description='Training SVM')
parser.add_argument('--vmodel', type=str, default='resnet152',
                    help='resnet50 | resnet101')
parser.add_argument('--vtype', type=str, default='imgnet',
                    help='imgnet | plc | hybrid')
parser.add_argument('--dset', type=str, default='clef_en',
                    help='clef_en | clef_ar | mediaeval | lesa')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='0,1,2,3')

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



def get_imagenet_feats(root, model_nm):

    feats, logits, im_names = [], [], []

    def feature_hook(module, input, output):
        return feats.extend(output.view(-1,output.shape[1]).data.cpu().numpy().tolist())

    def feature_hook_dense(module, input, output):
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = torch.flatten(output, 1)
        return feats.extend(output.view(-1,output.shape[1]).data.cpu().numpy().tolist())

    model = models.__dict__[model_nm](pretrained=True)

    model.eval().to(device)

    if 'dense' in model_nm:
        model._modules.get('features').register_forward_hook(feature_hook_dense)
    else:
        model._modules.get('avgpool').register_forward_hook(feature_hook)

    dataset = CustomDataset(root, transform_config)
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



def get_places365_feats(root, model_nm):

    feats, logits, im_names = [], [], []

    def feature_hook(module, input, output):
        return feats.extend(output.view(-1,output.shape[1]).data.cpu().numpy().tolist())

    model_file = '/media/gullal/Extra_Disk_1/places_models/%s_best.pth.tar'%(model_nm)
    model = models.__dict__[model_nm](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    
    model.load_state_dict(state_dict)
    model.eval().to(device)

    model._modules.get('avgpool').register_forward_hook(feature_hook)

    dataset = CustomDataset(root, transform_config)
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



def get_hybrid_feats(root, model_nm):

    feats, logits, im_names = [], [], []

    def feature_hook(module, input, output):
        return feats.extend(np.squeeze(output.data.cpu().numpy()).tolist())

    model_file = '/media/gullal/Extra_Disk_1/places_models/%s_hybrid_best.pth.tar'%(model_nm)
    model = models.__dict__[model_nm](num_classes=1365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    
    model.load_state_dict(state_dict)
    model.eval().to(device)

    model._modules.get('avgpool').register_forward_hook(feature_hook)

    dataset = CustomDataset(root, transform_config)
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
    dloc = 'data/clef/english/images/'
elif dset == 'clef_ar':
    dloc = 'data/clef/arabic/images/'
elif dset == 'mediaeval':
    dloc = 'data/mediaeval/images/'
else:
    dloc = 'data/lesa/images/'


## Extract and Save Features
feat_dict = {}
feats, logits, im_names = get_imagenet_feats(dloc, vmodel)

feat_dict['feats'] = {name:feat for name,feat in zip(im_names, feats)}
feat_dict['logits'] = {name:feat for name,feat in zip(im_names, logits)}

json.dump(feat_dict, open('features/image/%s_%s_%s.json'%(dset, vtype, vmodel), 'w'))
