import bcolz
import numpy as np
import pickle
import json
import nltk

if __name__ == '__main__':

    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='./data/glove.840B.300d.dat', mode='w')

    with open('./data/glove.840B.300d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            if line[0] == '.':
                vect = np.array(line[3:]).astype(np.float)
            else:
                vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((2196017, 300)), rootdir='./data/glove.840B.300d.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open('./data/glove.840B.300d_words.pkl', 'wb'))
    pickle.dump(word2idx, open('./data/glove.840B.300d_idx.pkl', 'wb'))

    vectors = bcolz.open('./data/glove.840B.300d.dat')[:]
    words = pickle.load(open('./data/glove.840B.300d_words.pkl', 'rb'))
    word2idx = pickle.load(open('./data/glove.840B.300d_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    word_embeddings = []
    labels = []
    with open('./data/input.json', 'r') as f:
        trainloader = json.load(f)
        for i in trainloader:
            doc = []
            tokens = nltk.word_tokenize(i['document'].lower())
            for ii in tokens:
                doc.append(glove(ii))
            doc_mean = np.ndarray(doc).mean(0)
            word_embeddings.append((doc_mean, i['is_credible']))

    all = np.ndarray(word_embeddings)
