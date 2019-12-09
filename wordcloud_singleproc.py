import sys
import os
import csv
import string
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from wordcloud import WordCloud
import docx2txt as d2t
from natsort import natsorted

# Takes command line arguments of three types: none, verb, or noun
# These are used to define which types of words get shown in the wordcloud while it is being created
flag = sys.argv[1]

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
    for char in inp:
        if char.isdigit():
            return True
    return False
    #return any(char.isdigit() for char in inp)

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
    'leads//////',
    'leads//////no'
]
stopwords = stopwords.words('english') + blacklist_words + [punc for punc in string.punctuation]
# stemmer for reducing words
stemmer = PorterStemmer()
# storing stemmed tokens
stemmed_corpus = []
# storing non-stemmed tokens
original_corpus = []
# make resulting directory if it does not already exist
if not os.path.exists(root_cleaned_filepath):
    os.makedirs(root_cleaned_filepath)
# testing purposes:
i = 0
# download nltk package for pos_tagger
nltk.download('averaged_perceptron_tagger')
for root, dirs, files in os.walk(text_filepath, topdown=True):
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
    # iterating over files int this dir
    for filename in files:
        if filename in blacklist:
            continue
        file = root + '/' + filename
        # read file content
        text = d2t.process(file)
        text = text.lower()
        # extract tokens from text
        tokens = word_tokenize(text)
        pos = nltk.pos_tag(tokens)
        # remove any strings containing numbers from the text
        for j in range(len(tokens) - 1):
            # encoding or windows end line characters result in len(tokens) returning relatively arbitrary values
            try:
                if hasNumbers(tokens[j]):
                    tokens.pop(j)
                if tokens[j] in stopwords:
                    tokens.pop(j)
                if flag == 'noun':
                    if pos[j][1][0] == 'N':
                        tokens.pop(j)
                elif flag == 'verb':
                    if pos[j][1][0] == 'R' or pos[j][1][0] == 'V':
                        tokens.pop(j)
            except IndexError as e:
                pass
                #print('Token list length: ' + str(len(tokens)))
                #print('Actual length: ' + str(j - 1))
        # stem tokens
        stemmed = [stemmer.stem(token) for token in tokens]
        # store stemmed text
        stemmed_corpus.append(stemmed)
        # store original text
        original_corpus.append(tokens)
        # clear memory
        del text
        del tokens
        if i >= 10:
            break
    if i >= 10:
        break
# build dictionary
dictionary = Dictionary(stemmed_corpus)
# get the actual form of each stemmed word
counts = get_common_surface_form(original_corpus, stemmer)
# convert to vector corpus
vectors = [dictionary.doc2bow(text) for text in stemmed_corpus]
# train TF-IDF model
tfidf = TfidfModel(vectors)
# get TF-IDF weights
weights = tfidf[vectors[0]]
# replace term IDs with human readable strings
param_dict = {}
for pair in weights:
    param_dict[counts[dictionary[pair[0]]]] = pair[1]
#weights = [(counts[dictionary[pair[0]]], pair[1]) for pair in weights]
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