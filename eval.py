from tqdm import tqdm
import numpy as np
from joblib import dump, load
import json
import nltk
import mxnet as mx
import gluonnlp as nlp
import random
from sklearn import svm
from pprint import pprint


glove = nlp.embedding.create('glove', source='glove.840B.300d', embedding_root='./data/')
word_embeddings = mx.ndarray.empty((300,))
labels = []
docs = []
docs_eval = []
clf = load('./model/svm_glove_word_mean.joblib')
with open('./data/docs.json', 'r') as f:
    trainloader = json.load(f)
    for i, items in enumerate(tqdm(trainloader)):
        if items['credible_issue']:
            tokens = nltk.word_tokenize(items['document'].lower())
            for ii, word in enumerate(tokens, 10):
                word_embeddings = mx.ndarray.concat(word_embeddings, glove[word], dim=0)
            doc_mean = word_embeddings.reshape(len(tokens), 300).mean(axis=0)
            doc_max = word_embeddings.reshape(len(tokens), 300).max(axis=0)
            docs.append(np.nan_to_num(doc_mean.asnumpy()))

print(clf.predict(docs))