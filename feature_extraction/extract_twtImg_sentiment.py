## Extraction code based on: https://github.com/fabiocarrara/visual-sentiment-analysis/blob/main/predict.py

import os, sys, json
import argparse
import numpy as np
import torch
import torchvision.transforms as t

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from t4sa.alexnet import KitModel as AlexNet
from t4sa.vgg19 import KitModel as VGG19


class ImageListDataset (Dataset):

    def __init__(self, img_names, root=None, transform=None):
        super(ImageListDataset).__init__()
    
        self.list = img_names
        self.root = root
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.list[index]
        if self.root:
            path = os.path.join(self.root, path)
            
        x = default_loader(path)
        if self.transform:
            x = self.transform(x)
        
        return x, self.list[index].strip('.png')
    
    def __len__(self):
        return len(self.list)


def main(args):

    feats1, logits, im_names = [], [], []
    data_dict = {}
    
    def feature_hook(module, input, output):
        return feats2.extend(output.view(-1,output.shape[1]).data.cpu().numpy().tolist())

    transform = t.Compose([
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Lambda(lambda x: x[[2,1,0], ...] * 255),  # RGB -> BGR and [0,1] -> [0,255]
        t.Normalize(mean=[116.8007, 121.2751, 130.4602], std=[1,1,1]),  # mean subtraction
    ])

    image_list = os.listdir(args.root)
    data = ImageListDataset(image_list, root=args.root, transform=transform)
    dataloader = DataLoader(data, batch_size=args.batch_size, num_workers=2)
    
    model = AlexNet if 'hybrid' in args.model else VGG19
    model = model('t4sa/{}.pth'.format(args.model)).to('cuda')
    model.eval()
    
    model._modules.get('fc7_1').register_forward_hook(feature_hook)

    with torch.no_grad():
        for x, im_nms in tqdm(dataloader):
            p, logs = model(x.to('cuda'))  # order is (NEG, NEU, POS)

            logits.extend(logs.cpu().numpy().tolist())
            im_names.extend(im_nms)

    data_dict['feats_fc7'] = {name:feat for name,feat in zip(im_names, feats1)}
    data_dict['logits'] = {name:feat for name,feat in zip(im_names, logits)}

    if not os.path.exists('features/image/'):
        os.makedirs('features/image/')
    json.dump(data_dict, open('features/image/%s_t4sa_%s.json'%(args.dset, args.model), 'w'))
    
if __name__ == '__main__':
    models = ('hybrid_finetuned_fc6+',
          'hybrid_finetuned_all',
          'vgg19_finetuned_fc6+',
          'vgg19_finetuned_all')

    parser = argparse.ArgumentParser(description='Extract Visual Sentiment Features')
    parser.add_argument('-d', '--dset', default=None, help='Which dataset (clef_en | clef_ar | mediaeval | lesa)')
    parser.add_argument('-r', '--root', default=None, help='Root path to prepend to image list')
    parser.add_argument('-m', '--model', type=str, choices=models, default='vgg19_finetuned_all', help='Pretrained model')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    main(args)