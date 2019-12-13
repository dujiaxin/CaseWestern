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
# built on python-docx so it should be similar in performance or better than anything I can program myself
import docx2txt as d2t
from natsort import natsorted
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
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

from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

text_filepath = './content/cleaned/'
root_cleaned_filepath = './'
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
stopwords = stopwords.words('english') + blacklist_words + [punc for punc in string.punctuation]
# stemmer for reducing words
stemmer = PorterStemmer()
# storing stemmed tokens
stemmed_corpus = []
# storing non-stemmed tokens
original_corpus = []
# add lemmatizer
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# make resulting directory if it does not already exist
if not os.path.exists(root_cleaned_filepath):
    os.makedirs(root_cleaned_filepath)
# testing purposes:
i = 0
# download nltk package for pos_tagger
# nltk.download('averaged_perceptron_tagger')
# empty file to prepare to append
# TODO: is this a good practice? -Jiaxin
open('word_freq.csv', 'w', encoding='utf-8').close()
# frequency dictionary for words
freq_dict = {}
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
    for filename in tqdm(files):
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
                    continue
                if tokens[j] in stopwords:
                    tokens.pop(j)
                    continue
                if flag == 'noun':
                    #print(pos[j][1][0])
                    if pos[j][1][0] == 'N':
                        if tokens[j] not in freq_dict.keys():
                            freq_dict[tokens[j]] = 1
                        else:
                            freq_dict[tokens[j]] += 1
                    else:
                        tokens.pop(j)
                        continue
                elif flag == 'verb':
                    print('verb')
                    if pos[j][1][0] == 'R' or pos[j][1][0] == 'V':
                        if tokens[j] not in freq_dict.keys():
                            freq_dict[tokens[j]] = 1
                        else:
                            freq_dict[tokens[j]] += 1
                    else:
                        tokens.pop(j)
                        continue
                else:
                    lemma = lemmatizer.lemmatize(tokens[j], get_wordnet_pos(tokens[j]))
                    if lemma not in freq_dict.keys():
                        freq_dict[lemma] = 1
                    else:
                        freq_dict[lemma] += 1
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
dic_keys = list(freq_dict.keys())
postokens = nltk.pos_tag(dic_keys)
with open('word_freq.csv', 'a+', encoding='utf-8') as freq_stat:
    for k in range(len(dic_keys) - 1):
        strtopend = dic_keys[k] + ',' + str(freq_dict[dic_keys[k]]) + ',' + get_wordnet_pos(dic_keys[k]) + '\n'
        freq_stat.write(strtopend)
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