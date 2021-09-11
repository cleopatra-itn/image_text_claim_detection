# On the Role of Images for Analyzing Claims in Social Media

The repository for the source code and the dataset used in the paper

> Gullal S. Cheema, Sherzod Hakimov, Eric Müller-Budack and Ralph Ewerth “On the Role of Images for Analyzing Claims in Social Media“, *Proceedings of the 2nd International Workshop on Cross-lingual Event-centric Open Analytics* co-located with the 30th The Web Conference (WWW 2021).

The paper is available here: http://ceur-ws.org/Vol-2829/paper3.pdf

Extended dataset can be downloaded from here: https://zenodo.org/record/4592249

## Setup
### For SVM Training and BERT Fine-tuning
- Cuda 10.2
- `conda env create -f environment.yml python=3.6.12`
- To install ThunderSVM on linux system,  
`pip install wheel https://github.com/Xtra-Computing/thundersvm/releases/download/v0.3.4/thundersvm_cuda10.1-0.3.4-cp36-cp36m-linux_x86_64.whl`
- Alternatively, ThunderSVM release for windows can be found [here](https://github.com/Xtra-Computing/thundersvm/releases).

### For VilBert Extraction and Fine-tuning
- Install dependencies and vilbert in a different environment by following instructions [here](https://github.com/facebookresearch/vilbert-multi-task/tree/9d645085b54fac6a781257133edabefdfb23e547).
- Download vilbert pretrained models either from [here](https://github.com/facebookresearch/vilbert-multi-task/tree/9d645085b54fac6a781257133edabefdfb23e547) or [here](https://github.com/jiasenlu/vilbert_beta) in folder named `vilbert-multi-task/data/pretrained/`.
- Download detectron model in `vilbert-multi-task/data/`.
- `wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth`
- `wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml`

## Data and Feature Extraction
- Download data from the zenodo repository
- Extract each zip file in `data/`
- Download pretrained [places](https://drive.google.com/file/d/1ARP8GS5LMGYc8T8lFTuYkBl9I9kJoIiL/view?usp=sharing) and [hybrid](https://drive.google.com/file/d/1i2NjrxRwQ3IQDrzmrmQEbCEOb8H6HXgM/view?usp=sharing) models.
- Extract Textual Features,
    - English : `python feature_extraction/extract_bert_en.py -d clef_en -m bertbase -p clean`
    - Arabic : `python feature_extraction/extract_bert_ar.py -m arabert`
- Extract Visual Features
    - Visual Sentiment : `python feature_extraction/extract_sent_feats.py -d clef_en -m vgg19_finetuned_all -b 32`
    - Visual Scene : `python feature_extraction/extract_visual_feats.py -v resnet152 -t imgnet -d clef_en`
- Extract VilBert Features
    - Extract bottom-up faster-RCNN features from images:
        - `python vilbert-multi-task/script/extract_features.py --model_file vilbert-multi-task/data/detectron_model.pth --config_file vilbert-multi-task/data/detectron_config.yaml --image_dir data/lesa/images/ --output_folder data/lesa/rcnn_feats/`
        - `python vilbert-multi-task/script/convert_to_lmdb.py --features_dir data/lesa/rcnn_feats/ --lmdb_file data/lesa/rcnn_lmdbs/`

    - Some images have no detectable objects. For those images, we take random crops and extract ResNet-152 last layer features. `python vilbert_code/extract_missing_feats.py --dset lesa`
    - Extract VilBert features: `python vilbert_code/extract_features.py --model multi_task --dset lesa`
- Use `-h` to see options

## SVM Training and Evaluation
- Text-based,  
`python svm_training/svm_textfeats.py --tfeat sumavg --tmodel bertbase --ttype clean --dset lesa --split 0`
- Image-based,  
`python svm_training/svm_imgfeats.py --vfeat feats --vmodel resnet152 --vtype imgnet --dset lesa --split 0`
- Image and Text based,  
`python svm_training/svm_imgText.py  --tfeat sumavg --tmodel bertbase --ttype clean --vfeat feats --vmodel resnet152 --vtype imgnet --split 0`
- VilBert based, `python svm_training/svm_vilbertfeats.py --normalize 1 --feat pooled --model multi_task --dset lesa`


## BERT Fine-tuning
- For ClEF_En: `python finetune_bert.py --btype bertweet --dset clef_en --bs 4`
- For LESA: `python finetune_bert.py --btype bertweet --dset lesa`
- For MediaEval: `python finetune_bert.py --btype covid_twitter --dset mediaeval`
- For CLEF_Ar: `python finetune_bert.py --btype arabert --dset clef_ar`

## VilBert Fine-tuning
- Using pooled token embeddings, `python finetune_vilbert.py --model image_ret --dset lesa --split 0 --un_fr 2 --pool add`
- Using averaged token embeddings, `python finetune_vilbert2.py --model image_ret --dset lesa --split 0 --un_fr 2`



If you find the shared resources useful, please cite:
```
@inproceedings{DBLP:conf/www/CheemaHME21,
  author    = {Gullal S. Cheema and
               Sherzod Hakimov and
               Eric M{\"{u}}ller{-}Budack and
               Ralph Ewerth},
  title     = {On the Role of Images for Analyzing Claims in Social Media},
  booktitle = {Proceedings of the 2nd International Workshop on Cross-lingual Event-centric
               Open Analytics co-located with the 30th The Web Conference {(WWW}
               2021), Ljubljana, Slovenia, April 12, 2021 (online event due to {COVID-19}
               outbreak)},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2829},
  pages     = {32--46},
  publisher = {CEUR-WS.org},
  year      = {2021},
  url       = {http://ceur-ws.org/Vol-2829/paper3.pdf}
}
```
