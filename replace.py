import sys
import os
import csv
import docx2txt as d2t
from docx import Document
from natsort import natsorted

text_filepath = 'C:/Users/Matt/Downloads/CW/TEST_CLEANED_2/'
root_cleaned_filepath = 'C:/Users/Matt/Downloads/CW/TEST_REPLACE_2/'
abbreviation_doc_filepath = 'C:/Users/Matt/Downloads/CW/SUPPORT/abbre.csv'
abbreviation_dict = {}
blacklist = [
    'document_process.csv'
]
# I could not figure out how to debug so I just removed from the .csv
"""
blacklist_words = [
    'IN',
    'NO',
]
"""
with open(abbreviation_doc_filepath, newline='') as csvfile:
    csvrdr = csv.reader(csvfile, delimiter=',')
    for row in csvrdr:
        if row[0] != '' and row[1] != '':
            #if row[0].lower not in blacklist_words:
            abbreviation_dict[row[0].strip()] = row[1].strip()
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
    for filename in files:
        if filename in blacklist:
            continue
        document = Document()
        file = root + '/' + filename
        #print(file)
        # read file content
        text = d2t.process(file)
        text_lines = text.split('\n')
        out_lines = []
        for line in text_lines:
            words = line.split(' ')
            retLine = ''
            for word in words:
                if word.strip() in abbreviation_dict.keys():
                    stripped_word = word.strip()
                    #print('replaced ' + word.strip(), end=' with ')
                    #word = abbreviation_dict[stripped_word]
                    #print(word)
                    retLine += abbreviation_dict[stripped_word] + ' '
                else:
                    retLine += word + ' '
            out_lines.append(retLine)
        for line in out_lines:
            document.add_paragraph(line)
        document.save(cleaned_filepath + filename)
        del text
        del document