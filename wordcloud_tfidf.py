import os
from os import path
import csv
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, get_single_color_func
from matplotlib import cm, colors
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
#nevermind, using alternative method explained below

# color_to_words will later have to be a dictionary of colors to list of words that will have those colors
# as such, a gradient of colors has to be made, then mapped to words that fall within a range of tfidf values
# so the colors will effectively be mapped to the tfidf values
# example from website:
'''
color_to_words = {
    # words below will be colord with a green single color function
    '#00ff00': ['beautiful', 'explicit', 'simple', 'sparse', 'readability', 'rules', 'practicality',
                'explicitly', 'one', 'now', 'easy', 'obvious', 'better']
    # will be colored with a red single color function
    'red': ['ugly', 'implicit', 'complex', 'complicated', 'nested', 'dense', 'special', 'errors',
            'silently', 'ambiguity', 'guess', 'hard']
}
'''

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

'''
# color function that changes colors of wordcloud depending on the tfidf of the word
def my_tfidf_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl(%d, 80%%, 50%%)' % (360 * tfidfDict[word])
'''

# https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html
# creates color function object that assigns exact colors to words based on mapping
class SimpleGroupedColorFunc(object):
    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}
        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

# creates color function object which assigns different shades of specified colors to certain words based on mapping
class GroupedColorFunc(object):
    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()
        ]
        self.default_color_func = get_single_color_func(default_color)
    
    # returns a single_color_func associated with the word
    def get_color_func(self, word):
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words
            )
        except StopIteration:
            color_func = self.default_color_func
        return color_func
    
    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)

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
    tfidfDict = {}
    default_color = 'grey'
    color_to_words = {}
    for k in tfDict:
        tfidfDict[k] = tfDict[k] * idfDict[k]
    print("creating wordcloud")
    print(cm.viridis(0))
    print(tfidfDict)
    max = 0
    min = 99999
    for value in tfidfDict.values():
        if value > max:
            max = value
        if value < min:
            min = value
    delta = (max - min)/256
    for key, value in tfidfDict.values():
        if key not in color_to_words.keys():
            idx = (value - min) // delta
            color_to_words[key] = colors.to_hex(cm.viridis(idx))
    wc = WordCloud(
        background_color='white',
        max_words = 100,
        width=1024,
        height=720
    ).generate_from_frequencies(tfDict)
    # TypeError: my_tfidf_color_func() missing 4 required positional arguments: 'word', 'font_size', 'position', and 'orientation'
    # color_func=lambda *args, **kwargs: (255,0,0) should work as a color function
    # it would make the wordcloud entirely red

    '''
    wc.recolor(color_func=my_tfidf_color_func, random_state=3)
    '''

    # Create a color function with single tone
    # grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)
    grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

    # Create a color function with multiple tones
    # grouped_color_func = GroupedColorFunc(color_to_words, default_color)

    # Apply our color function
    # wc.recolor(color_func=grouped_color_func)
    wc.recolor(color_func=grouped_color_func)

main()