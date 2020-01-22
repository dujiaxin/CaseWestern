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


def splite_file(text_filepath, to_filepath):
    print('splite file')
    content_pos = ''
    content_neg = ''
    with open(text_filepath, mode='r', encoding='utf-8') as fin:
        train_data = json.load(fin)
        pos_count = 0
        neg_count = 0
        for i in train_data:
            if i['credible_issue']:
                content_pos = content_pos + i['document']
                pos_count = pos_count + 1
            else:
                content_neg = content_neg + i['document']
                neg_count = neg_count + 1
        print('pos_count : ' + str(pos_count))
        print('neg_count : ' + str(neg_count))
        return train_data, content_pos, content_neg


def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)


def replace_marks(string, maxsplit=0):
    # replace special marks
    # string.replace('\\n','').replace('\\t','')
    markers = "*", "/", "+"
    regexPattern = '|'.join(map(re.escape, markers))
    return re.sub(regexPattern, ' ', string)


def document_features(document,word_features, ps=None):
    document_words = set(document)
    features = {}
    for word in word_features:
        if ps == None:
            features['contains({})'.format(word)] = (word in document_words)
        else:
            features['contains({})'.format(word)] = (ps.stem(word) in document_words)
    return features


def main():
    random.seed(0)
    nltk.download('punkt')
    nltk.download('words')
    text_filepath = './docs.json'
    train_filepath = './closure_words.json'
    ps = PorterStemmer()

    stopwords_file = './stopwords.txt'
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        custom_stopwords = set(f.read().splitlines())
    all_stopwords = default_stopwords | custom_stopwords
    with open(train_filepath, 'r', encoding='utf-8') as ft:
        train_data = json.load(ft)
    all_words = []
    for category in train_data:
        all_words += word_tokenize(replace_marks(category['Actual language from police report'].lower()))
    feature_set = set(all_words)
    # Remove single-character tokens (mostly punctuation)
    words = [word for word in feature_set if len(word) > 1]
    # Remove numbers
    # words = [word for word in tokens if not word.isnumeric()]
    feature_set_stem_nostop = set(w for w in words if w not in all_stopwords)
    documents = [(word_tokenize(replace_marks(category['Actual language from police report'].lower())), category['Coded As'])
                     for category in train_data]
    random.shuffle(documents)
    featuresets = [(document_features(d, feature_set_stem_nostop), c) for (d, c) in documents]
    train_set, test_set = featuresets[:750], featuresets[-75:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(50)
    return 0



if __name__ == '__main__':
    print('start program')
    main()
    print('end program')