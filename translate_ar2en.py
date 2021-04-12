from googletrans import Translator
import json
from urlextract import URLExtract
import time


translator = Translator()
extractor = URLExtract()

data_dict = json.load(open('data/clef/arabic/data.json', 'r'))

for i, idx in enumerate(data_dict):
    urls = extractor.find_urls(data_dict[idx]['full_text'])
    text = data_dict[idx]['full_text']
    for url in urls:
        text = text.replace(url, ' ')

    try:
        trans_text = translator.translate(text, src='ar', dest='en')
        data_dict[idx]['text_en'] = trans_text.text
    except:
        json.dump(data_dict, open('data/clef/arabic/data.json', 'w', encoding='utf-8'))

    print(i)
    time.sleep(1)


json.dump(data_dict, open('data/clef/arabic/data_translated.json', 'w', encoding='utf-8'))