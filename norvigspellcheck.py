import sys
import os
import csv
import docx2txt as d2t
from docx import Document
from natsort import natsorted
from spellchecker import SpellChecker

spell = SpellChecker()
text_filepath = 'C:/Users/Matt/Downloads/CW/TEST_REPLACE_2/'
root_cleaned_filepath = 'C:/Users/Matt/Downloads/CW/TEST_CHECK_2/'
abbreviation_doc_filepath = 'C:/Users/Matt/Downloads/CW/SUPPORT/abbre.csv'
blacklist = [
    'document_process.csv'
]
additional_words =[]
with open(abbreviation_doc_filepath, newline='') as csvfile:
    csvrdr = csv.reader(csvfile, delimiter=',')
    for row in csvrdr:
        additional_words.append(row[1])
spell.word_frequency.load_words(additional_words)
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
        if len(text) <= 500000: # maximum length of text for api calls
            res = client.check(text)
        with open('test.txt', 'w+') as testf:
            testf.write(str(res.raw_json))
        text_lines = text.split('\n')
        out_lines = []
        for line in text_lines:
            words = line.split(' ')
            retLine = ''
        for line in out_lines:
            document.add_paragraph(line)
        document.save(cleaned_filepath + filename)
        del text
        del document