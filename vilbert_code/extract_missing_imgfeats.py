from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
from helper_functions import *

import argparse

parser = argparse.ArgumentParser(description='Extract Missing Image Feats')
parser.add_argument('--dset', type=str, default='clef_en',
                    help='clef_en | clef_ar | mediaeval | lesa')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


transform_config = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

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


data_loc = 'data/%s/'%(args.dset)

##ImgNet Model for images where no objects detected
img_model = models.resnet152(pretrained=True)
img_model = nn.Sequential(*list(img_model.children())[:-1])
img_model.eval()
img_model.to(device)
##--------------------------------------------------

## Data Dict
data_dict = json.load(open(data_loc+'data.json', 'r', encoding='utf-8'))

visual_feat_dict = {'pooled': {}, 'average': {}}
text_feat_dict = {'pooled': {}, 'average': {}}

## Extracted Faster-RCNN Features
image_feature_reader = ImageFeaturesH5Reader(data_loc+'rcnn_lmdbs/')
max_region_num = 101

imgfeat_dict = {'feats': {}, 'boxes': {}}

for i, idx in enumerate(data_dict):
    print(i)
    img_id = idx
    
    ## Get image features and boxes
    try:
        features, num_boxes, boxes, _ = image_feature_reader[img_id]
    except:
        num_boxes = 101
        features, boxes = extract_imgnet_feats(img_model, img_id)

        imgfeat_dict['feats'][idx] = features.tolist()
        imgfeat_dict['boxes'][idx] = boxes.tolist()
        

json.dump(imgfeat_dict, open(data_loc+'rcnn_missing/%s.json'%(dset), 'w'))

