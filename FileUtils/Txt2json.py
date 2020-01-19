# coding: utf-8
import docx
import os
import json
import pandas
import spacy
import sys
from spacy import displacy


class Txt2json():
    original_documents = []  # For storing the documents with titles and rms
    issue_documents = []  # For storing the not credible documents
    para_delimiter = '\n'
    count_rms = 0
    rms_not_here = []
    rms_here = []
    nlp = spacy.load('en_core_web_sm')  # load spacy English moudle

    def __init__(self):
        self.filePath = '../content/cleaned/'
        self.toPath = "../labelled_report.json"
        sav = pandas.read_csv('../policereport.csv')
        self.rms = sav['Standardized_RMS'].to_list()
        self.matter_notcredible = sav['Matter_ID'].to_list()

    def check_rms(self, rms):
        if '-000' in rms:
            rms = rms.replace('000','')
        if '00-00' in rms:
            rms = rms.replace('00-00', '')
        if '-00' in rms:
            rms = rms.replace('00','')
        if rms in self.rms:
            self.count_rms = self.count_rms+1
            self.rms_here.append(rms)
        return rms


    def readDocx(self, filePath):
        file = docx.Document(filePath)
        doc = ""
        for para in file.paragraphs:
            doc = doc + para.text + self.para_delimiter
        return doc

    def docs2json(self):
        for file in os.listdir(self.filePath):  # Iterate over the files
            if file.endswith('.docx'):
                contents = self.readDocx(self.filePath + file)  # Load file contents
            elif file.endswith('.txt'):
                with open(self.filePath + file, 'r') as f:
                    contents = self.para_delimiter.join(f.read().splitlines())
            else:
                continue
            ids = file.replace('RMS','').replace('M','').replace('.docx' , '').replace('.txt' , '').split('_')
            mater_id = ids[0].replace('-', '')
            rms = ids[1].strip()
            doc = {"mater_id": mater_id, "rms": self.check_rms(rms),
                   "document": contents}
            self.original_documents.append(doc)
        with open(self.toPath, 'w') as to:
            json.dump(self.original_documents, to)
        with open("../data/docs_issue.json", 'w') as to:
            json.dump(self.issue_documents, to)


if __name__ == '__main__':
    a = Txt2json()
    a.Txt2json()
    print('000 format end---------------')
    print(a.count_rms)
    print('what not here---------------')
    list = []
    for i in a.rms:
        if i not in a.rms_here:
            list.append(i)
    print('\n'.join(list))

