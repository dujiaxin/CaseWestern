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
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import classification_report

def main():
    random.seed(0)
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    train_data = pd.read_json('./train_sak.json')
    df_labels = pd.read_csv('policereport.csv')
    # fc = FeatureClass()
    categories = list(df_labels.columns.values)
    # draw categories
    # rowsums = df_labels.iloc[:, 20:].sum(axis=1)
    # x = rowsums.value_counts()
    # plt.figure(figsize=(8, 5))
    # ax = sns.barplot(x.index, x.values)
    # plt.title("Multiple categories per comment")
    # plt.ylabel('# of Occurrences', fontsize=12)
    # plt.xlabel('# of categories', fontsize=12)
    # plt.show()

    stopwords_file = './stopwords.txt'
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        custom_stopwords = set(f.read().splitlines())
    all_stopwords = default_stopwords | custom_stopwords

    with open('catagories.csv', 'r', encoding='utf-8') as f:
        to_be_classified = list(f.read().splitlines())
    # with open(train_filepath, 'r', encoding='utf-8') as ft:
    #     train_data = json.load(ft)
    # df_indexed = df_labels.set_index('Standardized_RMS')
    # categories = list(df_indexed.columns.values)
    # for t in train_data:
    #     for category in categories:
    #         t.update({category : str(df_indexed.loc[t['rms']][category])})

    NB_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=all_stopwords)),
        ('clf', OneVsRestClassifier(MultinomialNB(
            fit_prior=True, class_prior=None))),
    ])
    SVC_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=all_stopwords)),
        ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
    ])
    LogReg_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=all_stopwords)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
    ])
    train, test = train_test_split(train_data, random_state=0, test_size=0.33, shuffle=True)
    X_train = train.document
    X_test = test.document
    for category in to_be_classified:
        # train the model using X_dtm & y
        NB_pipeline.fit(X_train, train[category])
        # compute the testing accuracy
        prediction = NB_pipeline.predict(X_test)
        print(category + ',' + str(accuracy_score(test[category], prediction)))
    return 0



if __name__ == '__main__':
    print('start program')
    main()
    print('end program')