import pandas as pd
import numpy as np
import statsmodels.api as sm


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


def main():
    train_data = pd.read_json('./train_sak.json')
    categories = list(train_data.columns.values)
    # to_explain = []
    # for c in categories:
    #     if len(train_data[train_data[c] == ' ']) < 100:
    #         to_explain.append(c)
    with open('./catagories_binary.csv', 'r', encoding='utf-8') as f:
        to_explain = f.read().splitlines()
    df = train_data[to_explain]
    df.astype(str).replace(' ','999')
    print(df.corr(method='spearman'))
    # x = df.drop(columns=['reason_closing_flow_chart', 'reason_closing_flow_chart_numeric', 'ReasonGiven_Closing_recoded', 'ReasonGiven_Closing',
    #              'Disposition', 'Jurisdiction_Prosecutor'])
    x = df.drop(columns=['reason_closing_flow_chart_numeric', 'ReasonGiven_Closing_recoded'])
    y = df['ReasonGiven_Closing_recoded']

    logit_model = sm.Logit(y, x)
    result = logit_model.fit()
    print(result.summary2())


if __name__ == '__main__':
    print('start program')
    main()
    print('end program')