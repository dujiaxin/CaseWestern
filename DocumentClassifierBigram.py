import nltk
from nltk.tokenize import word_tokenize,RegexpTokenizer
import os
import docx
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import json
import random
from nltk.stem.porter import PorterStemmer
from FeatureClass import FeatureClass
import sentiment

def main():
    random.seed(0)
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    train_filepath = './labelled_report.json'
    # fc = FeatureClass()
    with open(train_filepath, 'r', encoding='utf-8') as ft:
        train_data = json.load(ft)
    sntmnt = sentiment.SentimentAnalysis()
    for category in train_data:
        print(sntmnt.score(category['document']))
    return
    documents = [(category['document'], category['Coded As'])
                     for category in train_data]
    random.shuffle(documents)
    featuresets = [(fc.document_features(d), c) for (d, c) in documents]
    train_set, test_set = featuresets[:250], featuresets[-25:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(50)
    return 0



if __name__ == '__main__':
    print('start program')
    main()
    print('end program')