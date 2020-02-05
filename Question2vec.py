"""
Use tf-idf to vectorize questions
Wenbo
Jan 26, 2020
"""
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft

# from nltk.tokenize import word_tokenize
import torch
import torchtext.vocab as vocab
import os
import matplotlib.colors as mcolors

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# import time
# from datetime import datetime
import json
import re

import matplotlib.pyplot as plt
import pandas as pd



class Question2vec():
    def __init__(self):
        stopwords_file = './stopwords.txt'
        default_stopwords = set(nltk.corpus.stopwords.words('english'))
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            custom_stopwords = set(f.read().splitlines())
        self.stop_words = default_stopwords | custom_stopwords
        # self.stop_words = set(stopwords.words('english'))
        pass

    def filter_sentence(self, sentence, is_stop_words=False):
        if is_stop_words:
            stop_words = self.stop_words
        else:
            stop_words = []

        sentence = sentence.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(sentence)
        # word_tokens = word_tokenize(sentence)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return " ".join(filtered_sentence)

    def get_sentence_vec(self, sentence, is_stop_words=True):
        if is_stop_words:
            stop_words = self.stop_words
        else:
            stop_words = []

        filtered_sentence = self.filter_sentence(sentence.lower(), stop_words)
        # print(filtered_sentence)
        vec_arr = []
        for word in filtered_sentence:
            vec = self.get_word(word)
            vec_arr.append(vec)
        sentence_vec = torch.mean(torch.stack(vec_arr), dim=0)
        # print(sentence_vec[0:20])
        return sentence_vec



class My_show():
    def __init__(self, points, labels, colors, title = "Figure"):
        x = points[:, 0]
        y = points[:, 1]
        num_colors = 10

        self.names = labels

        self.c = colors
        self.title = title

        self.norm = plt.Normalize(1, num_colors)

        # TABLEAU_COLORS has 100 different colors, see:
        # https://matplotlib.org/3.1.0/gallery/color/named_colors.html

        lcmap = mcolors.ListedColormap(mcolors.TABLEAU_COLORS)
        self.cmap = lcmap

        self.fig, self.ax = plt.subplots()
        self.sc = plt.scatter(x, y, c=self.c, vmin=0, vmax=num_colors, s=100, cmap=self.cmap, norm=self.norm)

        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

    def update_annot(self, ind):
        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
        #                        " ".join([self.names[n] for n in ind["ind"]]))
        text = "\n".join([self.names[n] for n in ind["ind"]])

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
        plt.title(self.title)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        plt.colorbar()
        plt.show()


def get_questions(filepath):
    questions = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            l = line.strip()  # delete "\n"
            questions.append(l)
    return questions


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    questions = []

    trainfile = json.loads(pd.read_csv('./train_sak_geo.csv').to_json(orient='records'))

    q2v = Question2vec()
    good_results = []
    bad_results = []
    ycolor = []
    for q in trainfile:
        # if q['ReasonGiven_Closing_recoded'] in ['5','9','12']:
        if q['success_outcome'] == 1:
            # paras = q['document'].split('\n')
            bad_results.append(q2v.filter_sentence(q['document'], is_stop_words=True))
            ycolor.append(1)
            # for p in paras:
            #     bad_results.append(q2v.filter_sentence(q['document'], is_stop_words=True))
            #     questions.append(p)

        elif q['success_outcome'] == 0:
            # paras = q['document'].split('\n')
            bad_results.append(q2v.filter_sentence(q['document'], is_stop_words=True))
            ycolor.append(0)
            # for p in paras:
            #     bad_results.append(q2v.filter_sentence(p, is_stop_words=True))
            #     questions.append(p)


    # print(filtered_questions[:20])
    cv = ft.TfidfVectorizer()
    # tf_good_mat = cv.fit_transform(good_results).toarray()
    tf_bad_mat = cv.fit_transform(bad_results).toarray()
    words = cv.get_feature_names()
    print("len(words):", len(words))

    # for t in tfmat[:30]:
    #     print(list(t))

    """ dimension reduction """
    pca = PCA(n_components=2)
    pca.fit(tf_bad_mat)
    X = pca.transform(tf_bad_mat)

    """ Kmeans """
    y_pred = KMeans(n_clusters=10, random_state=9).fit_predict(X)

    """ show points with colors and labels """
    my = My_show(X, ycolor, ycolor, "Paragraph tf-idf features")
    my.show()
    short_questions = []

