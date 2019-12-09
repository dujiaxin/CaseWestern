import sys
import os
import csv
import string
import queue
import multiprocessing
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from wordcloud import WordCloud
import docx2txt as d2t
from natsort import natsorted
from diskcache import Cache

def get_common_surface_form(original_corpus, stemmer):
    counts = defaultdict(lambda: defaultdict(int))
    surface_forms = {}
    for document in original_corpus:
        for token in document:
            stemmed = stemmer.stem(token)
            counts[stemmed][token] += 1
    for stemmed, originals in counts.items():
        surface_forms[stemmed] = max(originals,
                                     key=lambda i: originals[i])
    return surface_forms

def hasNumbers(inp):
    return any(char.isdigit() for char in inp)

def processFiles(filels):
    tot = len(filels)
    print(multiprocessing.current_process().name + ' now processing files')
    global custom_stopwords
    stemmed_corp = []
    original_corp = []
    i = 0
    for filename in filels:
        # read file content
        text = d2t.process(filename)
        text = text.lower()
        print(multiprocessing.current_process().name + ' progress ' + str(i) + '/' + str(tot) + ': processing ' + str(filename))
        # extract tokens from text
        tokens = word_tokenize(text)
        # remove any strings containing numbers from the text
        for j in range(len(tokens) - 1):
            # encoding or windows end line characters result in len(tokens) returning relatively arbitrary values
            try:
                if hasNumbers(tokens[j]):
                    tokens.pop(j)
                if tokens[j] in custom_stopwords:
                    tokens.pop(j)
            except IndexError:
                break
                #print('Token list length: ' + str(len(tokens)))
                #print('Actual length: ' + str(j - 1))
            # stem tokens
            stemmed = [stemmer.stem(token) for token in tokens]
            # store stemmed text
            stemmed_corp.append(stemmed)
            # store original text
            original_corp.append(tokens)
        del text
        del stemmed
        del tokens
        sys.stdout.flush()
        i += 1
    print(multiprocessing.current_process().name + ' has finished processing files')
    sys.stdout.flush()
    q.put([stemmed_corp, original_corp])

text_filepath = 'C:/Users/Matt/Documents/Data Science/CW/CLEANED_3/'
root_cleaned_filepath = 'C:/Users/Matt/Documents/Data Science/CW/WORDCLOUD_3/'
blacklist = [
    'document_process.csv'
]
blacklist_words = [
    'ptl',
    'lukes',
    'june',
    'leads////',
    'leads/////',
    'leads//////'
]
custom_stopwords = stopwords.words('english') + blacklist_words + [punc for punc in string.punctuation]
# stemmer for reducing words
stemmer = PorterStemmer()
# storing stemmed tokens
stemmed_corpus = []
# storing non-stemmed tokens
original_corpus = []
# list of currently running threads
process_list = []
# queue of information processed by threads
q = multiprocessing.Queue()
# testing
# -1 for all files
filesToIter = 2
# -1 for all dirs
dirsToIter = -1
currDir = 0
if __name__ == '__main__':
    # make resulting directory if it does not already exist
    if not os.path.exists(root_cleaned_filepath):
        os.makedirs(root_cleaned_filepath)
    for root, dirs, files in os.walk(text_filepath, topdown=True):
        # skip first directory since it is just a folder of folders containing the actual documents
        if root == text_filepath:
            continue
        if dirsToIter != -1:
            if not (currDir >= dirsToIter):
                currDir += 1
            else:
                break
        # skip any loops that doesn't result in a folder of files
        if not files:
            continue
        # get folder name
        hierarchy = root.split('/')
        folder = ''
        for direc in reversed(hierarchy):
            if direc != '':
                folder = direc
                break
        cleaned_filepath = root_cleaned_filepath + str(folder) + '/'
        if not os.path.exists(cleaned_filepath):
            os.makedirs(cleaned_filepath)
        # creating thread for processing directory of files
        filels = []
        currFile = 0
        for filename in files:
            if filename in blacklist:
                continue
            if filesToIter != -1:
                if currFile < filesToIter:
                    currFile += 1
                else:
                    break
            filels.append(root + '/' + filename)
        print(filels)
        p = multiprocessing.Process(target=processFiles, args=(filels,))
        process_list.append(p)
        p.start()
    print('master process waiting...')
    print(str(len(process_list)))
    for proc in process_list:
        while not q.empty():
            print('emptying queue')
            res = q.get()
            stemmed_corpus.append(res[0])
            original_corpus.append(res[1])
        print('loop' + proc.name)
        proc.join(10)
    print('emptied queue, building frequency dictionary...')
    # build dictionary
    dictionary = Dictionary(stemmed_corpus)
    print('getting surface words from stemmer and original corpus...')
    # get the actual form of each stemmed word
    counts = get_common_surface_form(original_corpus, stemmer)
    print('creating bag of words...')
    # convert to vector corpus
    vectors = [dictionary.doc2bow(text) for text in stemmed_corpus]
    print('training TF-IDF...')
    # train TF-IDF model
    tfidf = TfidfModel(vectors)
    print('getting tfidf weights...')
    # get TF-IDF weights
    weights = tfidf[vectors[0]]
    # replace term IDs with human readable strings
    print('creating dictionary for param...')
    param_dict = {}
    for pair in weights:
        param_dict[counts[dictionary[pair[0]]]] = pair[1]
    #weights = [(counts[dictionary[pair[0]]], pair[1]) for pair in weights]
    print('building wordcloud...')
    # initialize wordcloud
    wc = WordCloud(
        background_color='white',
        max_words=1000,
        width=1024,
        height=720
    )
    # generate cloud
    wc.generate_from_frequencies(param_dict)
    # and save to file
    wc.to_file('word_cloud.png')