import sys
import os
import csv
import docx2txt as d2t
from docx import Document
from natsort import natsorted
# pip install pyspellchecker
#from spellchecker import SpellChecker
# pip install language-check
import language_check
# pip install nltk

language_check_tool = language_check.LanguageTool('en-US')

import nltk.data
# using sentence tokenizer to separate text into sentences that will then be checked by the language checker

# using pre-trained Punkt tokenizer for English
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

text_filepath = 'C:/Users/Matt/Documents/Data Science/CW/TEST_REPLACE_2/'
root_cleaned_filepath = 'C:/Users/Matt/Documents/Data Science/CW/TEST_CHECK_2/'
blacklist = [
    'document_process.csv'
]
if not os.path.exists(root_cleaned_filepath):
    os.makedirs(root_cleaned_filepath)
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
    # testing
    first = False
    # ^ testing
    for filename in files:
        if filename in blacklist:
            continue
        # testing
        if first:
            break
        else:
            first = True
        # ^ testing
        document = Document()
        file = root + '/' + filename
        #print(file)
        # read file content
        text = d2t.process(file)
        text_sentences = sentence_tokenizer.tokenize(text)
        out_lines = []
        for sentence in text_sentences:
            tmp = language_check.correct(sentence, language_check_tool.check(sentence))
            out_lines.append(tmp)
        for line in out_lines:
            document.add_paragraph(line)
        document.save(cleaned_filepath + filename)
        del text
        del document