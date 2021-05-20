# On the Role of Images for Analyzing Claims in Social Media

The repository for the source code and the dataset used in the paper

> Gullal S. Cheema, Sherzod Hakimov, Eric Müller-Budack and Ralph Ewerth “On the Role of Images for Analyzing Claims in Social Media“, *Proceedings of the 2nd International Workshop on Cross-lingual Event-centric Open Analytics* co-located with the 30th The Web Conference (WWW 2021).

The paper is available here: http://ceur-ws.org/Vol-2829/paper3.pdf

Extended dataset can be downloaded from here: https://zenodo.org/record/4592249

## Setup
### For SVM Training and BERT Fine-tuning
- Cuda 10.2
- `conda env create -f environment.yml python=3.6.12`
- `python -m spacy download en-core-web-lg==2.3.1`
- To install ThunderSVM on linux system `pip install wheel https://github.com/Xtra-Computing/thundersvm/releases/download/v0.3.4/thundersvm_cuda10.1-0.3.4-cp36-cp36m-linux_x86_64.whl`
- Alternatively, ThunderSVM release for windows can be found [here](https://github.com/Xtra-Computing/thundersvm/releases).

## Data and Feature Extraction
- Download data from the zenodo repository
- Extract each zip file in `data/`


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
