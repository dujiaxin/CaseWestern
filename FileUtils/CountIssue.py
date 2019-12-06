import json

if __name__ == '__main__':
    count = 0
    rmsList = []
    with open('../data/docs.json', 'r', encoding='utf-8') as file:
        docs = json.load(file)
        for doc in docs:
            if doc['credible_issue']:
                count = count+1
                rmsList.append(doc['rms'])
        print(count)
    with open('../data/issue_rms.csv', 'w') as f:
        f.write('\n'.join(rmsList))
