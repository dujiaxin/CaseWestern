import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from hmmlearn import hmm
import nltk
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


if __name__ == '__main__':
    n_samples = 2000
    n_features = 5000
    n_components = 10
    n_top_words = 20

    questions = []
    trainfile = json.loads(pd.read_csv('./train_sak_geo.csv').to_json(orient='records'))

    X = []
    lengths = []
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                    ngram_range=(1, 1),
                                    #token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b'),
                                stop_words=default_stopwords)
    tfidf_vectorizer = TfidfVectorizer(
        # max_df=0.95, min_df=2,
        #                         max_features=n_features,
                                    ngram_range=(1, 1),
                                       # token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b'),
                                stop_words=default_stopwords)
    docs = [q['document'] for q in trainfile]
    # cv.fit(docs)
    sentences = []
    for q in trainfile:
        # bad_results.append(q2v.filter_sentence(q['document'], is_stop_words=True))
        # q['document'] = re.sub('\b[0-9][0-9.,-]*\b', 'NUMBER-SPECIAL-TOKEN', q['document'])
        paras = q['document'].split('\n')

        sentences = sentences + paras
        # X.append(cv.transform(paras))
        # lengths.append(len(paras))
    tfidf = tfidf_vectorizer.fit_transform(sentences)
    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, 10)

    # words = cv.get_feature_names()

    # Fit the NMF model
    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
          "tf-idf features, n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    tf = tf_vectorizer.fit_transform(sentences)
    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    # Xall = np.concatenate([X])
    # remodel = hmm.GaussianHMM(n_components=4).fit(Xall, lengths)
    #
    # import pickle
    # with open("hmm.pkl", "wb") as file: pickle.dump(remodel, file)
    # with open("hmm.pkl", "rb") as file: pickle.load(file)