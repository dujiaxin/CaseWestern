import nltk
import pandas as pd
import json
import random
from nltk.stem.porter import PorterStemmer
from FeatureClass import FeatureClass
import sentiment
from tqdm import tqdm


def main():
    random.seed(0)
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    train_filepath = './train_sak.json'
    # fc = FeatureClass()
    with open(train_filepath, 'r', encoding='utf-8') as ft:
        train_data = json.load(ft)
    sntmnt = sentiment.SentimentAnalysis()
    df_labels = pd.read_csv('./Police_reports_geocodings.csv')
    df_indexed = df_labels.set_index('USER_BCI_N')
    categories = list(df_indexed.columns.values)
    for t in tqdm(train_data):
        t.update({'sentiment': str(sntmnt.score(t['document']))})
        for category in categories:
            try:
                t.update({category : str(df_indexed.loc[t['BCI_Number_Compl']][category])})
            except KeyError:
                t.update({category : '999'})

    with open('./train_sak_geo.json', 'w', encoding='utf-8') as fw:
        json.dump(train_data, fw)
    return 0



if __name__ == '__main__':
    print('start program')
    main()
    print('end program')