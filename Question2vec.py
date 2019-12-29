# -*- coding: utf-8 -*-
"""
author: Wenbo
Dec 25, 2019
"""

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
# from nltk.tokenize import word_tokenize
import torch
import torchtext.vocab as vocab
import os
# import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
# import mpld3

# print(set(stopwords.words('english')))

class Question2vec():
    def __init__(self):
        self.glove = vocab.GloVe(name='840B', dim=300)
        self.stop_words = set(stopwords.words('english'))
        print('Loaded {} words'.format(len(self.glove.itos)))
        pass

    def get_word(self, word):
        if word in self.glove.stoi.keys():
            return self.glove.vectors[self.glove.stoi[word]]
        else:
            print("*** no this key:", word)
            # TODO : fix OOV problem
            return self.glove.vectors[self.glove.stoi['unknown']]

    def closest(self, vec, n=10):
        """
        Find the closest words for a given vector
        """
        all_dists = [(w, torch.dist(vec, self.get_word(w))) for w in self.glove.itos]
        return sorted(all_dists, key=lambda t: t[1])[:n]

    def print_tuples(self, tuples):
        for tuple in tuples:
            print('(%.4f) %s' % (tuple[1], tuple[0]))

    # In the form w1 : w2 :: w3 : ?
    def analogy(self, w1, w2, w3, n=5, filter_given=True):
        print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))

        # w2 - w1 + w3 = w4
        closest_words = self.closest(self.get_word(w2) - self.get_word(w1) + self.get_word(w3))

        # Optionally filter out given words
        if filter_given:
            closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]

        self.print_tuples(closest_words[:n])

    def filter_sentence(self, sentence, stop_words):
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(sentence)
        # word_tokens = word_tokenize(sentence)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return filtered_sentence

    def get_sentence_vec(self, sentence, is_stop_words=True):
        if is_stop_words:
            stop_words = self.stop_words
        else:
            stop_words = []

        filtered_sentence = self.filter_sentence(sentence, stop_words)
        vec_arr = []
        for word in filtered_sentence:
            vec = self.get_word(word)
            vec_arr.append(vec)
        sentence_vec = torch.mean(torch.stack(vec_arr), dim=0)
        # print(sentence_vec[0:20])
        return sentence_vec



class My_show():
    def __init__(self, points, labels, colors):
        x = points[:, 0]
        y = points[:, 1]
        self.names = labels

        self.c = colors

        self.norm = plt.Normalize(1, 4)
        self.cmap = plt.cm.RdYlGn

        self.fig, self.ax = plt.subplots()
        self.sc = plt.scatter(x, y, c=self.c, s=100, cmap=self.cmap, norm=self.norm)

        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

    def update_annot(self, ind):
        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
        #                        " ".join([self.names[n] for n in ind["ind"]]))
        text = " ".join([self.names[n] for n in ind["ind"]])

        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor(self.cmap(self.norm(self.c[ind["ind"][0]])))
        self.annot.get_bbox_patch().set_alpha(0.8)

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()

    def show(self):
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        plt.show()

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    short_questions = []
    with open(BASE_DIR + '/data/short_question.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            l = line.strip()  # delete "\n"
            short_questions.append(l)

    long_questions = []
    with open(BASE_DIR + '/data/long_question.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            l = line.strip()  # delete "\n"
            long_questions.append(l)

    similarities = []

    q2v = Question2vec()

    # test1 = "This is a sample sentence, showing off the stop words filtration."
    # test2 = "This is a easy sentence, showing off the stop words filtration."
    # vec1 = q2v.get_sentence_vec(test1)
    # vec2 = q2v.get_sentence_vec(test2)
    # similarity = torch.cosine_similarity(vec1, vec2, 0)
    # print("test similarity:", similarity)

    short_vec_arr = []
    for i in range(len(short_questions)):
        short_q = short_questions[i]
        short_vec = q2v.get_sentence_vec(short_q)
        short_vec_arr.append(list(short_vec))

        # long_q = long_questions[i]
        # long_vec = q2v.get_sentence_vec(long_q)

        # similarity = float(torch.cosine_similarity(short_vec, long_vec, 0))
        # print(i, similarity)
        # print(short_q)
        # print(long_q)
        # print()
        # similarities.append(similarity)

    # data = {'short_question':short_questions, 'long_question':long_questions, 'similarity': similarities}
    # df = pd.DataFrame(data)
    # df.to_csv(BASE_DIR + '/data/question_similarity.csv', index=False, quoting=1)

    pca = PCA(n_components=2)
    pca.fit(short_vec_arr)
    X = pca.transform(short_vec_arr)

    """ show the results of dimension reduction """
    # plt.scatter(X[:, 0], X[:, 1], marker='o')
    # plt.show()

    """ Kmeans """
    y_pred = KMeans(n_clusters=10, random_state=9).fit_predict(X)

    """ show all the points """
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    # plt.show()

    """ show points with colors and labels """
    my = My_show(X, short_questions, y_pred)
    my.show()






