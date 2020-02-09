import os
import csv
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import docx2txt as d2t
from tqdm import tqdm

blacklist = [
    'document_process.csv'
]
blacklist_words = [
    'ptl',
    'lukes',
    'june',
    'leads////',
    'leads/////',
    'leads//////',
    'leads//////no',
    "''",
    '``',
    "'s"
]
# irrelevant or overused words
stopwords = stopwords.words('english') + blacklist_words + [punc for punc in string.punctuation]
original_corpus = []

def hasNumbers(inp):
    for char in inp:
        if char.isdigit():
            return True
    return False

def buildIdfInfo(dirPath):
    # word : [number of times appearing at least once in file, found in file flag]
    freqDict = {}
    filecount = 0
    for root, dirs, files in os.walk(dirPath, topdown=True):
        # skip any loops that don't result in dir of files
        if not files:
            continue
        # get folder name
        hierarchy = root.split('/')
        folder = ''
        for direc in reversed(hierarchy):
            if direc != '':
                folder = direc
                break
        # iterating over files in this dir
        for filename in tqdm(files):
            if filename in blacklist:
                continue
            filecount += 1
            file = root + '/' + filename
            # read file content
            text = d2t.process(file)
            text = text.lower()
            # extract tokens from text
            tokens = word_tokenize(text)
            # original_corpus is used to for the actual words within wordcloud
            # unsure if below line is needed here
            #original_corpus.append(tokens)
            for i in range(len(tokens - 1)):
                if hasNumbers(tokens[i]):
                    tokens.pop(i)
                    continue
                if tokens[i] in stopwords:
                    tokens.pop(i)
                    continue
                if tokens[i] not in freqDict.keys():
                    freqDict[tokens[i]] = [1, False]
                else:
                    if freqDict[tokens[i]][1] == True:
                        freqDict[tokens[i]][0] += 1
            del text
            del tokens
    with open('wc_wordCountDict.csv', 'w', encoding='utf-8') as fileDict:
        fileDictWriter = csv.writer(fileDict)
        fileDictWriter.writerow(str(filecount))
        for key, value in freqDict.items():
            fileDictWriter.writerow([key, str(value[0])])

def extractIdfInfo(filePath, tfDict):
    idfDict = {}
    wordcount = 0
    with open(filePath, 'r', encoding='utf-8') as fileDict:
        fileDictReader = csv.reader(fileDict)
        first = True
        for row in fileDictReader:
            if first:
                wordcount = int(row[0])
                first = False
                continue
            if row[0] in tfDict.keys():
                idfDict[row[0]] = wordcount / int(row[1])
    return idfDict


def buildTfInfo(filePath):
    # word : [number of times appearing in file]
    freqDict = {}
    # read file content
    text = d2t.process(file)
    text = text.lower()
    # extract tokens from text
    tokens = word_tokenize(text)
    wordcount = len(tokens)
    # add tokens to original_corpus, used to create wordcloud
    original_corpus.append(tokens)
    for i in range(len(tokens - 1)):
        if hasNumbers(tokens[i]):
            tokens.pop(i)
            continue
        if tokens[i] in stopwords:
            tokens.pop(i)
            continue
        if tokens[i] not in freqDict.keys():
            freqDict[tokens[i]] = 1
        else:
            freqDict[tokens[i]] += 1
    del text
    del tokens
    tfDict = {}
    for key, value in freqDict.items():
        tfDict[key] = value / wordcount
    return tfDict

def main():
    