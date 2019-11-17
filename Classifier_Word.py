from tqdm import tqdm
import numpy as np
from joblib import dump, load
import json
import nltk
import mxnet as mx
import gluonnlp as nlp
import random
from sklearn import svm

if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    mx.random.seed(1)

    glove = nlp.embedding.create('glove', source='glove.840B.300d', embedding_root='./data/')

    word_embeddings = mx.ndarray.empty((300,))
    labels = []
    docs = []
    with open('./data/input.json', 'r') as f:
        trainloader = json.load(f)
        for i, items in enumerate(tqdm(trainloader[:10])):
            doc = []
            tokens = nltk.word_tokenize(items['document'].lower())
            for ii, word in enumerate(tokens, 10):
                word_embeddings = mx.ndarray.concat(word_embeddings, glove[word], dim=0)
            doc_mean = word_embeddings.reshape(len(tokens), 300).mean(axis=0)
            docs.append(doc_mean.asnumpy())
            labels.append(items['is_credible'])
    clf = svm.SVC()
    clf.fit(docs, labels)
    dump(clf, './model/svm_glove_word_mean.joblib')
    print(clf.predict(docs[0]))
