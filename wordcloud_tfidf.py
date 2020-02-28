import os
from os import path
import csv
import string
import nltk
from nltk.corpus import stopwords
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

# TEST_SAMPLE is a dir that only contains a dir named CLEANED_2
# CLEANED_2 only contains a dir named 14
# which only contains 30 .docx police reports that have been extracted and cleaned
idfDirPath = 'C:/Users/Matt/Documents/Data Science/CW/TEST_SAMPLE/CLEANED_2'
wordCountFilePath = 'C:/Users/Matt/Documents/GitHub/CaseWestern/wc_wordCountDict.csv'
docFilePath = 'C:/Users/Matt/Documents/Data Science/CW/TEST_SAMPLE/CLEANED_2/14/M518--3338_RMS98-67176.docx'
# irrelevant or overused words
stopwords = stopwords.words('english') + blacklist_words + [punc for punc in string.punctuation]
original_corpus = []

print("imported, inited, and now defining methods")

def my_tfidf_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl(%d, 80%%, 50%%)' % (360 * kwargs[freqDict][word])

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
        # WARNING: will cause error if one of the files is open in Microsoft Word
        # this is due to the lock file being created while a .docx is open
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
            for i in range(len(tokens) - 1):
                if hasNumbers(tokens[i]) or tokens[i] in stopwords:
                    continue
                if tokens[i] not in freqDict.keys():
                    freqDict[tokens[i]] = [1, False]
                else:
                    if freqDict[tokens[i]][1] == True:
                        freqDict[tokens[i]][0] += 1
            del text
            del tokens
            # resetting the boolean that determines whether or not a word has yet been seen in the document
            for value in freqDict.values():
                value[1] = True
    with open('wc_wordCountDict.csv', 'w', newline='', encoding='utf-8') as fileDict:
        fileDictWriter = csv.writer(fileDict)
        fileDictWriter.writerow(["file count:", str(filecount)])
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


def buildExtractTfInfo(filePath):
    # word : [number of times appearing in file]
    freqDict = {}
    # read file content
    text = d2t.process(filePath)
    text = text.lower()
    # extract tokens from text
    tokens = word_tokenize(text)
    wordcount = len(tokens)
    # add tokens to original_corpus, used to create wordcloud
    original_corpus.append(tokens)
    for i in range(len(tokens) - 1):
        if hasNumbers(tokens[i]) or tokens[i] in stopwords:
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
    print("in main()")
    if not path.exists(wordCountFilePath):
        print("word count per file csv file NOT FOUND, building idf info")
        buildIdfInfo(idfDirPath)
    else:
        print("word count per file csv file found, will extract idf info")
    print("building and extracting tf info for document selected")
    tfDict = buildExtractTfInfo(docFilePath)
    print("extracting idf info")
    idfDict = extractIdfInfo(wordCountFilePath, tfDict)
    tfidfDict = {}
    print("calculating tfidf values")
    for k in tfDict:
        tfidfDict[k] = tfDict[k] * idfDict[k]
    print("creating wordcloud")
    wc = WordCloud(
        background_color='white',
        max_words = 100,
        width=1024,
        height=720
    ).generate_from_frequencies(tfDict)
    wc.recolor(color_func=my_tfidf_color_func(freqDict=tfidfDict))\

main()