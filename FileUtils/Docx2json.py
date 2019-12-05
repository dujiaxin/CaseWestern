import docx
import os
import json
import pandas

class Docx2json():
    original_documents = []  # For storing the documents with titles and rms
    issue_documents = []  # For storing the not credible documents
    para_delimiter = '\n\n'
    count = 0

    def __init__(self):
        self.filePath = '../content/auto_2/'
        self.toPath = "../docs.json"
        sav = pandas.read_csv('../check.csv')
        self.rms_notcredible = sav['Standardized_RMS'].to_list()
        self.matter_notcredible = sav['Matter_ID'].to_list()

    def check_rms(self,rms):
        if rms in self.rms_notcredible:
            self.count = self.count+1
        return rms in self.rms_notcredible

    def check_matter(self,matter):
        if int(matter) in self.matter_notcredible:
            self.count = self.count+1
        return matter in self.matter_notcredible

    def readDocx(self, filePath):
        file = docx.Document(filePath)
        doc = ""
        for para in file.paragraphs:
            doc = doc + para.text + self.para_delimiter
        return doc

    def docs2json(self):
        for file in os.listdir(self.filePath):  # Iterate over the files
            contents = self.readDocx(self.filePath + file)  # Load file contents
            ids = file.replace('.docx' , '')
            mater_id = ids[1:8]
            rms = ids.lower().split('rms')[1]
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
    print(a.count)
