import docx
import os


class Docx2txt():
    original_documents = []  # For storing the documents with titles and rms
    para_delimiter = '\n'
    count_rms = 0
    count_matter = 0
    rms_not_here = []
    rms_here = []

    def __init__(self):
        self.filePath = '../CLEANED_3/'
        self.toPath = "../NARRATIVES_BATCH_3/"

    def check_rms(self, rms):
        if '-000' in rms:
            print(rms)
            rms = rms.replace('000','')
        if '-00' in rms:
            print(rms)
            rms = rms.replace('00','')
        return rms

    def readDocx(self, filePath):
        file = docx.Document(filePath)
        doc = ""
        for para in file.paragraphs:
            doc = doc + para.text + self.para_delimiter
        return doc

    def getListOfFiles(self, dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
        return allFiles

    def docs2txt(self):
        allFiles = self.getListOfFiles(self.filePath)
        for file in allFiles:  # Iterate over the files
            if file.endswith('.docx') == False:
                continue
            contents = self.readDocx(file)  # Load file contents
            ids = os.path.basename(file).replace('RMS','').replace('M','').replace('.docx' , '').split('_')
            mater_id = ids[0]
            rms = ids[1]
            name = "M" + mater_id + "_RMS"+ rms + '.txt'
            with open(self.toPath + name, 'w', encoding='utf-8') as to:
                to.write(contents)


if __name__ == '__main__':
    a = Docx2txt()
    a.docs2txt()
    print('end')

