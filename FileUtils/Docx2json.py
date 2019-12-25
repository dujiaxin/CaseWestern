import docx
import os
import json
import pandas


class Docx2json():
    original_documents = []  # For storing the documents with titles and rms
    issue_documents = []  # For storing the not credible documents
    para_delimiter = '\n\n'
    count_rms = 0
    count_matter = 0
    rms_not_here = []
    rms_here = []

    def __init__(self):
        self.filePath = '../content/auto_2/'
        self.toPath = "../docs.json"
        sav = pandas.read_csv('../check.csv')
        self.rms_notcredible = sav['Standardized_RMS'].to_list()
        self.matter_notcredible = sav['Matter_ID'].to_list()

    def check_rms(self, rms):
        if '-000' in rms:
            print(rms)
            rms = rms.replace('000','')
        if '-00' in rms:
            print(rms)
            rms = rms.replace('00','')
        if rms in self.rms_notcredible:
            self.count_rms = self.count_rms+1
            self.rms_here.append(rms)
        return rms in self.rms_notcredible

    def check_matter(self,matter):
        if int(matter) in self.matter_notcredible:
            self.count_matter = self.count_matter+1
        return int(matter) in self.matter_notcredible

    def readDocx(self, filePath):
        file = docx.Document(filePath)
        doc = ""
        for para in file.paragraphs:
            doc = doc + para.text + self.para_delimiter
        return doc

    def docs2json(self):
        for file in os.listdir(self.filePath):  # Iterate over the files
            if file.endswith('.docx') == False:
                continue
            contents = self.readDocx(self.filePath + file)  # Load file contents
            ids = file.replace('RMS','').replace('M','').replace('.docx' , '').split('_')
            mater_id = ids[0].replace('-', '')
            rms = ids[1]
            if self.check_matter(mater_id):
                issue_doc = {"mater_id": mater_id, "rms": rms, "credible_issue": self.check_rms(rms), "document": contents}
                self.issue_documents.append(issue_doc)
            doc = {"mater_id": mater_id, "rms": rms, "credible_issue": self.check_rms(rms), "document": contents}
            self.original_documents.append(doc)
        with open(self.toPath, 'w') as to:
            json.dump(self.original_documents, to)
        with open("../data/docs_issue.json", 'w') as to:
            json.dump(self.issue_documents, to)


if __name__ == '__main__':
    a = Docx2json()
    a.docs2json()
    print('000 format end---------------')
    print(a.count_matter)
    print(a.count_rms)
    print('what not here---------------')
    list = []
    for i in a.rms_notcredible:
        if i not in a.rms_here:
            list.append(i)
    print('\n'.join(list))

