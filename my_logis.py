# my logist 
import pandas as pd
from pandas import DataFrame
import numpy as np
import csv
import os
from sklearn.linear_model import LogisticRegression
import random
from statistics import mean
from math import sqrt
from scipy import stats
from scipy.stats import spearmanr
from sklearn import metrics


if __name__ == '__main__':
    if os.path.exists('weights.csv'):
        os.remove('weights.csv')

    if os.path.exists('df.csv'):
        os.remove('df.csv')



    train_data = pd.read_csv('./train_sak_geo.csv')
    categories = train_data.columns
    #print(categories)

    with open(r'./features.csv', 'r', encoding='utf-8') as f:
        to_explain = f.read().splitlines()
        #print(to_explain)
    # remove the variable does not exist
    for elem in to_explain:
        if elem not in categories:
            print(elem)
            to_explain.remove(elem)

    df = train_data[to_explain]


    #df.info()
    df = df.astype(float)
    df_T = pd.DataFrame(values,columns=to_explain)
    # print(df.info()) # CONVERT DATAFRAME TO NON-NULL FLOAT DATAFRAME
    df = df.replace(' ',np.nan)
    df = df.replace(-999,np.nan)
    df = df.replace(999,np.nan)
    df = df.dropna()

    # extract label 'success_outcome'
    label = df['success_outcome']
    df.drop('success_outcome',axis=1,inplace=True)
    # df.drop(['Average_of_hourly_temperature','report_length','Childprotservices',
    #          'Complshower', 'Began_other_crime'], axis=1, inplace=True)
    df.to_csv('df.csv')

    #convert dataframe to list
    train = df.values.tolist() # convert rows to instances
    label = label.values.tolist()

    # dictionary to record the ID of original train
    dic = {}
    for i in range(len(train)):
        train[i] = list(np.float_(train[i]))
        dic[i] = label[i]

    ###########################T STATISTIC TEST############################
    # print(df_T)
    name = list(df_T.columns)
    p_value = []
    for i in range(df_T.shape[1]):
        col_name = name[i]
        if i != 19:
            sep = 0
            df_sort = df_T.iloc[:,[i,19]]
            df_sort = df_sort.sort_values(by=['success_outcome'])
            #print(df_sort)
            col = df_sort['success_outcome'].to_list()
            for j in range(len(col)):
                if col[j] != 0:
                    sep = j
                    break
            # print(sep)
            df_zero = df_sort.iloc[:sep,:]
            df_one = df_sort.iloc[sep:,:]
        # print(df_one)
        # print(df_zero)
            mean_zero = df_zero.iloc[:,0].mean()
            var_zero = df_zero.iloc[:,0].var()
            num_zero = df_zero.shape[0]
            mean_one = df_one.iloc[:,0].mean()
            var_one = df_one.iloc[:,0].var()
            num_one = df_one.shape[0]
            T_stat = abs(mean_zero-mean_one)/sqrt((var_zero**2/num_zero)+(var_one**2/num_one))
            pval = stats.t.sf(np.abs(T_stat), num_one-1)*2
            p_value.append(round(pval,4))
            # print([mean_zero,var_zero,num_zero],[mean_one,var_one,num_one],col_name,[T_stat,p_value])
            #print(col_name,[T_stat,pval])

    ##################SPEARMAN CORRELATION###################
    to_explain.remove('success_outcome')
    corr = []
    for i in to_explain:
        col = df[i].tolist()
        c,p = spearmanr(col,label)
        corr.append(round(c,2))
        #print(i,'spearman:',round(c,2),'p-value',round(p,2))

    ###########################CROSS VALIDATION############################
    # random sampling the trainID
    trainID = random.sample(range(0,len(train)),800)
    train_sp = np.array([train[ID] for ID in trainID])
    train_label = np.array([dic.get(ID) for ID in trainID])
    # remaing ID behave as testID
    testID = []
    for i in range(len(train)):
        if i not in trainID:
            testID.append(i)
    test_sp = np.array([train[ID] for ID in testID])
    test_label = np.array([dic.get(ID) for ID in testID])

    ###########################LOGISTIC REGRESSION############################
    #fitting model into train_sample and predict the test_sample
    clf = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
    clf.fit(train_sp,train_label)
    w = clf.coef_
    w = [round(num,4) for num in w[0]]
    prediction = clf.predict(test_sp)
    # print('prediction:',prediction)
    #print('weights:',len(w))
    # print(len(prediction),len(test_label))

    # calculate accuracy of logistic_weights of features
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == test_label[i]:
            count += 1
    accuary = count/len(prediction)
    print(format(accuary,'%'))
    ######## List the weight.csv file #######
    weights = pd.DataFrame(w,columns=['logistic_weights'],index=to_explain)
    weights['p_value'] = p_value
    weights['Spearman_Correlation'] = corr
    print(weights)
    weights.to_csv('weights.csv')

    fpr, tpr, thresholds = metrics.roc_curve(test_label, prediction)
    print(metrics.auc(fpr, tpr))