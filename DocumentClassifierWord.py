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

def lexical_diversity(text):
    return len(set(text)) / len(text)

def percentage(count, total):
    return 100 * count / total


def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)


def replace_marks(string, maxsplit=0):
    # replace special marks
    # string.replace('\\n','').replace('\\t','')
    markers = "*", "/"
    regexPattern = '|'.join(map(re.escape, markers))
    return re.sub(regexPattern, ' ', string)


def document_features(document,word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


def main():
    random.seed(1)
    nltk.download('punkt')
    nltk.download('words')
    text_filepath = './docs.json'
    to_filepath = './content/all.json'
    content = ''
    #content = prepare_file(text_filepath, to_filepath)
    train_data, content_pos, content_neg = splite_file(text_filepath, to_filepath)
    tokens_pos = word_tokenize(replace_marks(content_pos.lower()))
    tokens_neg = word_tokenize(replace_marks(content_neg.lower()))
    documents = [(word_tokenize(replace_marks(category['document'].lower())), category['credible_issue'])
                     for category in train_data]
    random.shuffle(documents)
    featuresets = [(document_features(d, set(tokens_neg+tokens_pos)), c) for (d, c) in documents]
    train_set, test_set = featuresets, featuresets
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(50)
    return 0



if __name__ == '__main__':
    print('start program')
    main()
    print('end program')