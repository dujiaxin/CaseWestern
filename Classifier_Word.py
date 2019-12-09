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


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    mx.random.seed(1)

    glove = nlp.embedding.create('glove', source='glove.840B.300d', embedding_root='./data/')

    word_embeddings = mx.ndarray.empty((300,))
    labels = []
    docs = []
    docs_eval = []
    with open('./data/docs.json', 'r') as f:
        trainloader = json.load(f)
        for i, items in enumerate(tqdm(trainloader)):
            tokens = nltk.word_tokenize(items['document'].lower())
            for ii, word in enumerate(tokens, 10):
                word_embeddings = mx.ndarray.concat(word_embeddings, glove[word], dim=0)
            doc_max = word_embeddings.reshape(len(tokens), 300).max(axis=0)
            docs.append(np.nan_to_num(doc_max.asnumpy()))
            if items['credible_issue']:
                labels.append(1.0)
            else:
                labels.append(0.0)
    clf = svm.SVC()
    pprint(docs)
    print('labels')
    pprint(labels)
    clf.fit(docs, labels)
    dump(clf, './model/svm_glove_word_max.joblib')
    print('predict')
    results = clf.predict(docs)
    pprint(results)


