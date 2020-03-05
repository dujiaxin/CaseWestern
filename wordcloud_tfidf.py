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

# blacklisted files not to be tokenized and put into word frequency dictionary
blacklist = [
    'document_process.csv'
]
# blacklisted words that are nonsensical and not picked up by stopword detector
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

# tfidf dictionary declared globally so that it may be reached by the recolor method
# which relies on it to color the words
tfidfDict = {}

# TEST_SAMPLE is a dir that only contains a dir named CLEANED_2
# CLEANED_2 only contains a dir named 14
# which only contains 30 .docx police reports that have been extracted and cleaned
# file paths to utilize
# idfDirPath is used to find documents and words to build idf csv from
# test with 1 dir of 30 items so far
idfDirPath = 'C:/Users/Matt/Documents/Data Science/CW/TEST_SAMPLE/CLEANED_2'
# wordCountFilePath is the file path to output the idf csv to
wordCountFilePath = 'C:/Users/Matt/Documents/GitHub/CaseWestern/wc_wordCountDict.csv'
# docFilePath is the document that will be used to calculate tf information
docFilePath = 'C:/Users/Matt/Documents/Data Science/CW/TEST_SAMPLE/CLEANED_2/14/M518--3338_RMS98-67176.docx'
# irrelevant or overused words
stopwords = stopwords.words('english') + blacklist_words + [punc for punc in string.punctuation]
original_corpus = []

print("imported, inited, and now defining methods")

# color function that changes colors of wordcloud depending on the tfidf of the word
def my_tfidf_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl(%d, 80%%, 50%%)' % (360 * tfidfDict[word])

# simple function that determines whether or not numbers exist in a string, causing it to return a boolean value
def hasNumbers(inp):
    for char in inp:
        if char.isdigit():
            return True
    return False

# build idf info from the documents found in the directories at the root CLEANED directory, dirPath
# not run if there is a prior wc_wordCountDict.csv, program goes straight to extracting information from file
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
            # clear large amount of mem for at least a small time
            del text
            del tokens
            # resetting the boolean that determines whether or not a word has yet been seen in the document
            for value in freqDict.values():
                value[1] = True
    # output all information to filepath dictated at top of program to file wc_wordCountDict.csv
    # first row contains "file count: <int file count>"
    # next rows contain "<str word> <int count>"
    with open('wc_wordCountDict.csv', 'w', newline='', encoding='utf-8') as fileDict:
        fileDictWriter = csv.writer(fileDict)
        fileDictWriter.writerow(["file count:", str(filecount)])
        for key, value in freqDict.items():
            fileDictWriter.writerow([key, str(value[0])])

# method used to extract idf info from previously created wc_wordCountDict.csv file
def extractIdfInfo(filePath, tfDict):
    idfDict = {}
    wordcount = 0
    with open(filePath, 'r', encoding='utf-8') as fileDict:
        fileDictReader = csv.reader(fileDict)
        first = True
        for row in fileDictReader:
            if first:
                wordcount = int(row[1])
                first = False
                continue
            if row[0] in tfDict.keys():
                idfDict[row[0]] = wordcount / int(row[1])
    return idfDict

# method that builds and extracts tf info based on the document selected at the top of this program
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
    # no need to store information since there is not much data being utilized just yet
    for key, value in freqDict.items():
        tfDict[key] = value / wordcount
    return tfDict

# main program run flow can be seen here
# build idf info
# build tf info
# extract tf info
# extract idf info
# calculate tfidf
# create wordcloud
# recolor wordcloud <- current problem
def main():
    # use the dictionary that was declared globally to determine colors of words
    global tfidfDict
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
    # TypeError: my_tfidf_color_func() missing 4 required positional arguments: 'word', 'font_size', 'position', and 'orientation'
    # color_func=lambda *args, **kwargs: (255,0,0) should work as a color function
    # it would make the wordcloud entirely red

    wc.recolor(color_func=my_tfidf_color_func, random_state=3)

main()