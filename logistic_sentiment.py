from sklearn.linear_model import LogisticRegression
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import classification_report
import warnings
from glob import glob
import nltk
from collections import Counter


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


def print_top20(vectorizer, clf, category, X_train):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    class_labels = set(category)
    # add whatever settings you want
    countVec = CountVectorizer(ngram_range=(2, 3),
        token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b')
    # fit transform
    cv = countVec.fit_transform(X_train)
    # feature names
    cv_feature_names = countVec.get_feature_names()
    # feature counts
    feature_count = cv.toarray().sum(axis=0)
    # feature name to count
    all_count_dict = dict(zip(cv_feature_names, feature_count))
    for i, class_label in enumerate(class_labels):
        print("category:")
        print(class_label)
        idx_list = [i for i,c in enumerate(category) if c == class_label]
        countVec2 = CountVectorizer(ngram_range=(2, 3),
                                   token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b')
        # fit transform
        X_train_category = []
        for idx in idx_list:
            X_train_category.append(X_train[idx])
        cv2 = countVec2.fit_transform(X_train_category)
        # feature names
        cv_feature_names2 = countVec2.get_feature_names()
        # feature counts
        feature_count2 = cv2.toarray().sum(axis=0)
        # feature name to count
        category_count_dict = dict(zip(cv_feature_names2, feature_count2))
        if i == len(class_labels)-1:#for liner svc
            top10 = np.argsort(clf.coef_[i-1])[:20]
            for j in top10:
                try:
                    print(feature_names[j] + " (appears: " + str(
                        category_count_dict[feature_names[j]]) + "/" +
                          str(all_count_dict[feature_names[j]])
                          + " times in the category)")
                except KeyError:
                    print(feature_names[j] + " (appears: 0/" +
                          str(all_count_dict[feature_names[j]])
                          + " times in the category)")
            break
        top10 = np.argsort(clf.coef_[i])[-20:]
        for j in top10:
            print('coef_: ' + str(clf.coef_[j]))
            try:
                print(feature_names[j] + " (appears: " + str(
                    category_count_dict[feature_names[j]]) + "/" +
                      str(all_count_dict[feature_names[j]])
                      + " times in the category)")
            except KeyError:
                print(feature_names[j] + " (appears: 0/" +
                      str(all_count_dict[feature_names[j]])
                      + " times in the category)")
        print("---least_coef_order---")
        neg_top10 = np.argsort(clf.coef_[i])[:20]
        for j in neg_top10:
            print('coef_: ' + str(clf.coef_[j]))
            try:
                print(feature_names[j] + " (appears: " + str(
                    category_count_dict[feature_names[j]]) + "/" +
                      str(all_count_dict[feature_names[j]])
                      + " times in the category)")
            except KeyError:
                print(feature_names[j] + " (appears: 0/" +
                      str(all_count_dict[feature_names[j]])
                      + " times in the category)")



def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        if i == class_labels.shape[0]-1:#for liner svc
            break
        top10 = np.argsort(clf.coef_[i])[-20:]
        print("%s: %s" % (class_label,
              ", ".join(feature_names[j] for j in top10)))
        print()


def convert_prosecutor_involvement(reason):
    outcome = 99
    if reason == 'no prosecutor involvement' or reason == 'suspect charged, arrested, warrant for S' or reason == 'investigative involved dcfs substantiation':
        outcome = 0
    elif reason == 'conferred with prosecutor' or reason =='case will be referred or sent to juv court' or reason =='case will be forwarded to prosector':
        outcome = 1
    return outcome

def convert_closing_reason(reason):
    outcome = 'missing'
    if reason == 'lack of V follow up' or reason == 'victim no prosecute':
        outcome = 'Victim no engagement (italics indicate latent variable naming/theme name)'
    elif reason == 'NFIL NOS' or reason == 'held in abeyance plus some closing language' or reason == 'held in abeyance pending' or reason == 'insuff evidence':
        outcome = 'Stalled'
    elif reason == 'Issued No Papers':
        outcome = 'Could be do not believe victim or could be nothing else that could be done'
    elif reason == 'none given':
        outcome = 'Didn’t do anything'
    elif reason == 'Victim recanted' or reason == 'V lied or doubted V':
        outcome = 'Don’t believe victim'
    elif reason == 'Suspect was charged' or reason == 'grand jury' or reason == 'warrant for S arrest' or reason == 'Assistant Pros issued papers' or reason == 'forwarded to juv court' or reason == 'sent to prosecutor or juv court':
        outcome = 'Arrest, charged, indicted'
    return outcome



def sentiment_classify():
       df = pd.read_excel('MasterAsOf.xlsx')
       df = pd.read_spss('MasterAsOf7.sav')
       X =df[['max_sentence_subjectivity','max_sentence_polarity',	'max_sentence_sntmnt_score','max_paragraph_subjectivity']]

       X=df[['Victim_vulnerability_any',
              'vulnerability_in_situ', 'vulnerability_status_longterm',
               'Vulnerability_reason_longterm',
              'Vulnerability_reason_in_situ',
              'runaway_unruly_vulnerability',
              'mental_cognitive_illness_vulnerability', 'prostitution_vulnerability',
              'physical_impairment_vulnerability',
              'intoxicated_drugged_during_assault_vulnerability',
              'drug_alcohol_related_vulnerability', 'V_homeless_vulnerability']]
       y = df['reason_closing_flow_chart_numeric']

       y = df['reason_closing_flow_chart_numeric']
       clf = LogisticRegression(random_state=0).fit(X, y)
       clf.predict(X[:2, :])

       clf.predict_proba(X[:2, :])


       clf.score(X, y)


def get_original_narrative(string_all):
    string_all = clean_text(string_all)
    if "original narrative" in string_all:
        string_all = string_all.split("original narrative")[1]
    else:
        print("no original_narrative")
    list_of_title = ["Additional Narrative",
                     "Additional Information",
                     "Detective Narrative",
                     "Detective Follow Up",
                     "Detective followup",
                     "SCU Narrative",
                     "Supplement Narrative",
                     "Supplemental Narrative",
                     "SUPPLEMENT #01 NARRATIVE"
                     ]
    for title in list_of_title:
        title = clean_text(title)
        if title in string_all:
            string_all = string_all.split(title)[0]
            #print("delete "+title)
    return string_all




def main():
    df = pd.read_spss('MasterAsOf7.sav')
    with open('big_txt.txt', 'r', encoding='utf-8') as f:
        bigtxt = f.read()
        seperated_files = bigtxt.split('.txt\n')
        f_c = {}
        first_file_name = seperated_files[0]+'.txt'
        f_c[first_file_name]=seperated_files[1]
        for i in range(1, len(seperated_files)-1):
            filename = seperated_files[i].split('\n')[-1]+'.txt'
            f_c[filename]=seperated_files[i+1]

    NB_pipeline = Pipeline([
          ('tfidf', TfidfVectorizer(#stop_words=all_stopwords,
                 ngram_range=(1, 2))),
          ('clf', OneVsRestClassifier(MultinomialNB(
                 fit_prior=True, class_prior=None))),
    ])
    SVC_pipeline = Pipeline([
          ('tfidf', TfidfVectorizer(stop_words='english',
                                    ngram_range=(3, 3),
                                    #token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b'
                                    )
           ),
        ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
          #('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
    ])
    LogReg_pipeline = Pipeline([
          ('tfidf', TfidfVectorizer(#stop_words=all_stopwords
                  )),
          ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
    ])
    X_train = []
    category = []
    for i in range(0, len(df)):
        try:
            X_train.append(f_c[df['file_name'].iloc[i].replace('rtf', 'txt')])
            category.append(convert_prosecutor_involvement(df['Prosecutor_involved'].iloc[i]))
        except KeyError:
            print(df['file_name'].iloc[i].replace('rtf', 'txt'))
            continue
    SVC_pipeline.fit(X_train, category)
    #NB_pipeline = NB_pipeline.fit(X_train, category)
    #print_top10(NB_pipeline.steps[0][1], NB_pipeline.steps[1][1], NB_pipeline.classes_)
    print_top10(SVC_pipeline.steps[0][1], SVC_pipeline.steps[1][1], SVC_pipeline.classes_)

    df2 = df[df['vulnerability_status_longterm'].isna() == False]
    X_train = []
    category = []
    for i in range(0, len(df2)):
        try:
            X_train.append(f_c[df2['file_name'].iloc[i].replace('rtf', 'txt')])
            category.append(df2['vulnerability_status_longterm'].iloc[i])
        except KeyError:
            print(df2['file_name'].iloc[i].replace('rtf', 'txt'))
            continue
    SVC_pipeline.fit(X_train, category)
    print_top10(SVC_pipeline.steps[0][1], SVC_pipeline.steps[1][1], SVC_pipeline.classes_)


def prosecutor_involved_main():
    df = pd.read_spss('MasterAsOf7.sav')
    with open('big_txt.txt', 'r', encoding='utf-8') as f:
        bigtxt = f.read()
        seperated_files = bigtxt.split('.txt\n')
        f_c = {}
        first_file_name = seperated_files[0] + '.txt'
        # print(first_file_name)
        # f_c[first_file_name]=seperated_files[1]
        f_c[first_file_name] = get_original_narrative(seperated_files[1])
        for i in range(1, len(seperated_files) - 1):
            filename = seperated_files[i].split('\n')[-1] + '.txt'
            # print(filename)
            # f_c[filename]=seperated_files[i+1]
            f_c[filename] = get_original_narrative(seperated_files[i + 1])

    X_train = []
    category = []
    for i in range(0, len(df)):
        try:
            X_train.append(f_c[df['file_name'].iloc[i].replace('rtf', 'txt')])
            involve_prosecuter = convert_prosecutor_involvement(df['Prosecutor_involved'].iloc[i])
            if involve_prosecuter == 1:
                category.append("prosecutor involvement=1")
            else:
                category.append("prosecutor involvement=0")
        except KeyError:
            print(df['file_name'].iloc[i].replace('rtf', 'txt'))
            continue
    tfidf3 = TfidfVectorizer(  # stop_words=stopwords_bi_tri+nltk.corpus.stopwords.words('english'),
        ngram_range=(3, 3),
        token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b',
        max_df=0.5
    )
    tfidf2 = TfidfVectorizer(  # stop_words=stopwords_bi_tri+nltk.corpus.stopwords.words('english'),
        ngram_range=(2, 2),
        token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b',
        max_df=0.5
    )

    X_train2 = tfidf2.fit_transform(X_train)
    X_train3 = tfidf3.fit_transform(X_train)
    print("LinearSVC()")
    clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("MultinomialNB(fit_prior=True, class_prior=None)")
    clf = OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("LogisticRegression(solver='sag')")
    clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))


def closing_reason_main():
    df = pd.read_spss('MasterAsOf7.sav')
    with open('big_txt.txt', 'r', encoding='utf-8') as f:
        bigtxt = f.read()
        seperated_files = bigtxt.split('.txt\n')
        f_c = {}
        first_file_name = seperated_files[0]+'.txt'
        f_c[first_file_name] = get_original_narrative(seperated_files[1])
        for i in range(1, len(seperated_files)-1):
            filename = seperated_files[i].split('\n')[-1]+'.txt'
            #print(filename)
            #f_c[filename]=seperated_files[i+1]
            f_c[filename] = get_original_narrative(seperated_files[i + 1])

    tfidf3=TfidfVectorizer(  # stop_words=stopwords_bi_tri+nltk.corpus.stopwords.words('english'),
        ngram_range=(3, 3),
        token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b',
        max_df=0.5
    )
    tfidf2=TfidfVectorizer(  # stop_words=stopwords_bi_tri+nltk.corpus.stopwords.words('english'),
        ngram_range=(2, 2),
        token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b',
        max_df=0.5
    )
    X_train = []
    category = []
    for i in range(0, len(df)):
        try:
            X_train.append(f_c[df['file_name'].iloc[i].replace('rtf', 'txt')])
            category.append(convert_closing_reason(df['reason_closing_flow_chart_numeric'].iloc[i]))
        except KeyError:
            print(df['file_name'].iloc[i].replace('rtf', 'txt'))
            continue

    X_train2 = tfidf2.fit_transform(X_train)
    X_train3 = tfidf3.fit_transform(X_train)
    print("LinearSVC()")
    clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("MultinomialNB(fit_prior=True, class_prior=None)")
    clf = OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("LogisticRegression(solver='sag')")
    clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))


def combined_main():
    df_original = pd.read_spss('MasterAsOf7.sav')
    with open('big_txt.txt', 'r', encoding='utf-8') as f:
        bigtxt = f.read()
        seperated_files = bigtxt.split('.txt\n')
        f_c = {}
        first_file_name = seperated_files[0]+'.txt'
        #print(first_file_name)
        #f_c[first_file_name]=seperated_files[1]
        f_c[first_file_name] = get_original_narrative(seperated_files[1])
        for i in range(1, len(seperated_files)-1):
            filename = seperated_files[i].split('\n')[-1]+'.txt'
            #print(filename)
            #f_c[filename]=seperated_files[i+1]
            f_c[filename] = get_original_narrative(seperated_files[i + 1])

    tfidf3=TfidfVectorizer(  # stop_words=stopwords_bi_tri+nltk.corpus.stopwords.words('english'),
        ngram_range=(3, 3),
        token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b',
        max_df=0.5
    )
    tfidf2=TfidfVectorizer(  # stop_words=stopwords_bi_tri+nltk.corpus.stopwords.words('english'),
        ngram_range=(2, 2),
        token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b',
        max_df=0.5
    )

    X_train = []
    category = []
    print('-'*10)
    print('insuff evidence')
    df = df_original[df_original['reason_closing_flow_chart_numeric']=='insuff evidence']
    for i in range(0, len(df)):
        try:
            X_train.append(f_c[df['file_name'].iloc[i].replace('rtf', 'txt')])
            involve_prosecuter = convert_prosecutor_involvement(df['Prosecutor_involved'].iloc[i])
            if involve_prosecuter == 1:
                category.append("prosecutor involvement=1")
            else:
                category.append("prosecutor involvement=0")
        except KeyError:
            print(df['file_name'].iloc[i].replace('rtf', 'txt'))
            continue
    X_train2 = tfidf2.fit_transform(X_train)
    X_train3 = tfidf3.fit_transform(X_train)
    print("LinearSVC()")
    clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("MultinomialNB(fit_prior=True, class_prior=None)")
    clf = OneVsRestClassifier(MultinomialNB(
                 fit_prior=True, class_prior=None))
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("LogisticRegression(solver='sag')")
    clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))

    df = df_original[df_original['reason_closing_flow_chart_numeric']=='lack of V follow up']
    print('-' * 10)
    print('lack of V follow up')
    for i in range(0, len(df)):
        try:
            X_train.append(f_c[df['file_name'].iloc[i].replace('rtf', 'txt')])
            involve_prosecuter = convert_prosecutor_involvement(df['Prosecutor_involved'].iloc[i])
            if involve_prosecuter == 1:
                category.append("prosecutor involvement=1")
            else:
                category.append("prosecutor involvement=0")
        except KeyError:
            print(df['file_name'].iloc[i].replace('rtf', 'txt'))
            continue

    X_train2 = tfidf2.fit_transform(X_train)
    X_train3 = tfidf3.fit_transform(X_train)
    print("LinearSVC()")
    clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("MultinomialNB(fit_prior=True, class_prior=None)")
    clf = OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("LogisticRegression(solver='sag')")
    clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))

    df = df_original[df_original['reason_closing_flow_chart_numeric'] == 'victim no prosecute']
    print('-' * 10)
    print('victim no prosecute')
    for i in range(0, len(df)):
        try:
            X_train.append(f_c[df['file_name'].iloc[i].replace('rtf', 'txt')])
            involve_prosecuter = convert_prosecutor_involvement(df['Prosecutor_involved'].iloc[i])
            if involve_prosecuter == 1:
                category.append("prosecutor involvement=1")
            else:
                category.append("prosecutor involvement=0")
        except KeyError:
            print(df['file_name'].iloc[i].replace('rtf', 'txt'))
            continue

    X_train2 = tfidf2.fit_transform(X_train)
    X_train3 = tfidf3.fit_transform(X_train)
    print("LinearSVC()")
    clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("MultinomialNB(fit_prior=True, class_prior=None)")
    clf = OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("LogisticRegression(solver='sag')")
    clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    df = df_original[df_original['reason_closing_flow_chart_numeric'] == 'V lied or doubted V']
    print('-' * 10)
    print('V lied or doubted V')
    for i in range(0, len(df)):
        try:
            X_train.append(f_c[df['file_name'].iloc[i].replace('rtf', 'txt')])
            involve_prosecuter = convert_prosecutor_involvement(df['Prosecutor_involved'].iloc[i])
            if involve_prosecuter == 1:
                category.append("prosecutor involvement=1")
            else:
                category.append("prosecutor involvement=0")
        except KeyError:
            print(df['file_name'].iloc[i].replace('rtf', 'txt'))
            continue

    X_train2 = tfidf2.fit_transform(X_train)
    X_train3 = tfidf3.fit_transform(X_train)
    print("LinearSVC()")
    clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("MultinomialNB(fit_prior=True, class_prior=None)")
    clf = OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))
    print("LogisticRegression(solver='sag')")
    clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
    print("bi-grams")
    clf.fit(X_train2, category)
    print_top20(tfidf2, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train2, category)))
    print("tri-grams")
    clf.fit(X_train3, category)
    print_top20(tfidf3, clf, category, X_train)
    print('R2 score: {0:.2f}'.format(clf.score(X_train3, category)))





if __name__ == '__main__':
       print('start program')
       #n_grams = 3
       # df = pd.read_spss('MasterAsOf7.sav')
       # c_vec = CountVectorizer(stop_words='english', ngram_range=(n_grams, n_grams), token_pattern=u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b')
       # reasons = [clean_text(t) for t in df['Closing_language']]
       # ngrams = c_vec.fit_transform(reasons)
       # # count frequency of ngrams
       # count_values = ngrams.toarray().sum(axis=0)
       # # list of ngrams
       # vocab = c_vec.vocabulary_
       # df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
       #                         ).rename(columns={0: 'frequency', 1: 'bigram/trigram'})
       # stopwords_bi_tri = df_ngram[df_ngram["frequency"]> 10]["bigram/trigram"].to_list()
       # stopwords_bi_tri = stopwords_bi_tri + ["id rms", "mater id"]
       #combined_main()
       #closing_reason_main()
       prosecutor_involved_main()
       print('end program')