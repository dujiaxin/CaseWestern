import re
import datetime
import json
import spacy
import pandas as pd
import sys
from tqdm import tqdm


nlp = spacy.load('en_core_web_sm')
### regular expression for RMS
rms_pattern = r'\bRMS[;,.:/]?\ [\s]?[0-9]{2}[-/,.]?[0-9]*\b|\bRMS[;,.:/]?[\s]*[0-9]{2}[-/,.]?[0-9]+\b|[_]RMS[;,.:/]?[0-9]{2}[-/,.]?[0-9]+[_]'
### regular expression extract date between 1900-2100
date_slash_Y = r'\b[0-1][0-9]\/[0-3][0-9]\/[1-2][09][0-9]{2}|\b[1-9]\/[0-3][0-9]\/[1-2][09][0-9]{2}|\b[1-9]\/[1-9]\/[1-2][09][0-9]{2}|\b[1-2][09][0-9]{2}\/[0-1][0-9]\/[0-3][0-9]'
date_slash_y = r'\b[0-1][0-9]\/[0-3][0-9]\/[0-9]{2}\b|\b[0-1][0-9]\/[1-9]\/[0-9]{2}\b|\b[1-9]\/[0-3][0-9]\/[0-9]{2}\b|\b[1-9]\/[1-9]\/[0-9]{2}\b'
date_slash_NY = r'\b[1-9]\/[0-3][0-9]\b|\b[0-1][0-9]\/[0-3][0-9]\b|\b[1-9]\/[1-9]\b'
date_dash_Y = r'\b[0-1][0-9]\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[1-9]\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[1-9]\-[1-9]\-[1-2][09][0-9]{2}|\b[A-Z][a-z]{2,8}\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[A-Z][a-z]{2,8}\-[0-3][0-9]\-[0-9]{2}'
date_dash_y = r'\b[0-1][0-9]\-[0-3][0-9]\-[0-9]{2}\b|\b[1-9]\-[0-3][0-9]\-[0-9]{2}\b|\b[1-9]\-[1-9]\-[0-9]{2}\b'
date_dash_NY = r'\b[JFMASOND][aepuco][nbrlgptvcy]\-[0-3][0-9]\b'
date_Noseparate = r'\b[0-1][0-9][0-3][0-9][1-2][09][0-9]{2}'
date_period = r'\b[0-1][0-9]\.[0-3][0-9]\.[0-9]{2}\b|\b[1-9]\.[0-3][0-9]\.[0-9]{2}\b|\b[1-9]\.[1-9]\.[0-9]{2}\b'
date_space = r'\b[0-1][0-9]\ [0-3][0-9]\ [0-9]{2}\b|\b[0-1][0-9]\ [0-3][0-9]\ [1-2][09][0-9]{2}'
date_written = r'\b[A-Z][A-Za-z]{2,8}[.,]?\ [0-3][0-9][.,]?\[\s]?[1-2][09][0-9]{2}|\b[A-Z][A-Za-z]{2,8}\[\s]?[1-9]\[,\s]?[1-2][09][0-9]{2}\b|\b[A-Z][A-Za-z]{2,8}\,[1-2][09][0-9]{2}\b|\b[A-Z][A-za-z]{2,8}.\ [0-3][0-9]\,[\s]?[1-2][09][0-9]{2}\b|\b[0-3][0-9][.,]?\ [A-Z][A-za-z]{2,8}[.,]?\ [1-2][09][0-9]{2}\b|\b[A-Z][A-Za-z]{2,8}\ [0-3]?[0-9][tT][hH][,.]?\ [1-2][09][0-9]{2}\b'
date_written_NY = r'\b[A-Z][a-z]{2,8}\ [0-3][0-9]th\b'
date_written_ND = r'\b[A-Z][a-z]{2,8}\ [1-2][09][0-9]{2}'
### extract hours
hours = r'\b[\d]{4,6}[\s]?\bHRS\b|\b[\d]{4,6}[\s]?hrs\b|\b[\d]{4,6}[\s]?HR\b|\b[AaPp][Mm]\b|\b[\d]{0,2}[:,.]?[\d]+[\s]?[AaPp][Mm]\b'
### re for extracting abbrevations
new_pattern = r'\bA/M\b|\bAgg\.\b|\bAKA|APPROX\b|\bappt\.\[\s]?\b|\bAPT\b|\bARR\b|\bASST\b|\bATM\b|\bATT\.\b|\bAUTO\b|\bAV\.\b|\bave\.\b|\bB\b|\bB\&E|b/f|B/M\b|\bBAC\b|\bBCI\b|\bBF\b|\bBLDG\b|\bblk\b|\bbrn\b|\bC/N\b|\bC\/W\b|\bCAPT\b|\bCCDCFS\b|\bCCS\b|\bCK\b|\bCleve\b|\bCMHA\b|\bCO\.\b|\bCOMM\b|\bCOMP\b|\bCONF\.\b|\bCPD\b|\bCS\b|\bct\.\[\s]?\b|\bCWS\b|\bDEPT\.\b|\bDet\.\b|\bDETS\b|\bDH\b|\bDHS\b|\bDist\b|\bDK\b|\bdob\b|\bDOB\b|\bdr\b|\bDR\.\[\s]?\b|\bDR\.\[\s]?\b|\bDVD\b|\bE\.\b|\bE/B\b|\bEMS\b|\bER\b|\bETA\b|\bFBI\b|\bFel\.\b|\bFIR\b|\bFT\b|\bFYI\b|\bGOA\b|\bGSI\b|\bGTMV\b|\bH/M\b|\bHGT\b|\bhosp\.\b|\bHR\b|\bhrs\b|\bHS\b|\bHT\b|\bHTS\b|\bI\-\b|\bID\'D\b|\binfo\b|\bINT\b|\bINTOX\b|\bINTOX\b|\bINVEST\b|\bJC\b|\bjc\b|\bJO\b|\bJr\.\[\s]?\b|\blbs\.\b|\bLCI\b|\bLIC\b|\bLIEUT\b|\bLN\.\b|\bLt\.\b|\bM\b|\bm/t/E\b|\bM/T/E/S\b|\bMED\b|\bMGS\b|\bMHMC\b|\bMIN\b|\bMISC\b|\bMLK\b|\bMO\b|\bMTE\b|\bMV\b|\bN\.S\.\b|\bN/B\b|\bN/E\b|\bN/S\b|\bN/W\b|\bNARC\.\ POSS\b|\bNB\b|\bNFI\b|\bNFIL\b|\bNMD\b|\bNR\b|\bOFF\.\b|\bOH\b|\bOIC\b|\bORC\b|\bP\.A\.\b|\bP\.O\.\b|\bP\.O\.|PC\b|\bPCS\b|\bPD\b|\bpg\b|\bPH\b|\bPIO\b|\bPOSS\b|\bPROP\b|\bPROP\b|\bPROS\b|\bPROSC\b|\bPTL\b|\bPTLM\b|\bR/P\b|\bRD\b|\bREC\b|\bREC\'D\b|\bRECV\'D\b|\bREP\b|\bREPTS\b|\bRN\b|\bRP\b|\bRPT\b|\bs\.|S\.O\.\b|\bS/B\b|\bS/W\b|\bS/W/F\b|\bS/W/M\b|\bSANE\b|\bSCU\b|\bSGT\b|\bSIO\b|\bSLMC\b|\bSS\#\b|\bST\b|\bSUB\b|\bSUBJ\.\b|\bSUS\b|\bSUSP\b|\bSVCH\b|\bT\ \&\ R\b|\bT\ AND\ R\b|\bT\.V\.\b|\bT/A\b|\bT/R\b|\bT/R\'D\b|\bTHURS\b|\bTRAFF\b|\bUH\b|\bUSPS\b|\bUTL\b|\bV\[\s]?\b|\bVEH\b|\bVIC\b|\bVict\.\b|\bVICTS\b|\bVICT\'S\[\s]?\b|\bVS\b|\bW\.\b|\bW/\b|\bW/B\b|\bW/M\b|\bWGT\b|\bWIT\b|\bWITN\b|\bWM\b|\bWTS\b|\bYR\b|\bYRS\.\b|\bZ/C\b|ZC\b'

def ner(myfile): # load spacy to formate the context and split the sentence
    sentence = []
    for num,sen in enumerate(myfile.sents):
        sentence.append(str(sen))
    return(sentence)

def getDate(report_text): # retrive datetime to 'DD-MM-YYYY HH:MM:SS'
    doc = nlp(report_text)
    #print(doc)
    rms = re.search(rms_pattern,report_text)
    sen = ner(doc)
    date = []
    #pattern_hidden = [date_slash_NY,date_dash_NY,date_Noseparate,date_written_NY]
    pattern = [date_dash_Y,date_slash_y,date_slash_Y,date_slash_y,date_dash_y,date_period,date_space,date_written,date_slash_NY,date_dash_NY,date_Noseparate,date_written_NY,date_written_ND]
    for i in range(len(sen)):
        for j in range(len(pattern)):
            date_time = re.search(pattern[j],sen[i])
            if date_time != None:
                hour = re.search(hours,sen[i])
                if hour != None:
                    date_time = date_time.group() + ' ' + hour.group()
                    united = pd.to_datetime(date_time,errors='ignore')
                    if type(united) != str:
                        if rms != None:
                            date.append([date_time,datetime.datetime.strftime(united,'%d-%m-%Y %H:%M:%S'),str(sen[i]),rms.group()])
                        else:
                            date.append([date_time,datetime.datetime.strftime(united,'%d-%m-%Y %H:%M:%S'),str(sen[i]),'None'])
                    #print('original:',date_time,'formated:',united)
                else:
                    united = pd.to_datetime(date_time.group(),errors='ignore')
                    if type(united) != str:
                        if rms != None:
                            date.append([date_time.group(),datetime.datetime.strftime(united,'%d-%m-%Y %H:%M:%S'),str(sen[i]),rms.group()])
                        else:
                            date.append([date_time.group(),datetime.datetime.strftime(united,'%d-%m-%Y %H:%M:%S'),str(sen[i]),'None'])
                    #print('original:',date_time.group(),'formated:',united)
    return(date)



# regular expression pattern converting abbreviation to full name
def fullName(text_report):
    rms = re.search(rms_pattern,text_report)
    full_name = []
    doc = nlp(text_report)
    draft_sen = ner(doc)
    for i in range(len(draft_sen)):
        cut_sen = draft_sen[i]
        find = re.search(new_pattern,cut_sen)
        while find != None:
            if rms != None:
                full_name.append([find.group(),full_dic.get(find.group()),str(draft_sen[i]),rms.group()])
                cut_sen = cut_sen.replace(find.group(),'')
                find = re.search(new_pattern,cut_sen)
            else:
                full_name.append([find.group(),full_dic.get(find.group()),str(draft_sen[i]),'None'])
                cut_sen = cut_sen.replace(find.group(),'')
                find = re.search(new_pattern,cut_sen)                
    return(full_name)



##### open document json
with open('./train_sak.json','r') as f: # input the report file.json
    reports = []
    file = json.load(f)
    for elem in file:
        if elem['document']:
            reports.append(elem['document'])

#####Convert Abbreviations to full names####
with open('./NIJ_AbbreviationList.xlsx','rb') as f: # input the Abbreviation file.xlsx
    abbr = []
    full = []
    df = pd.read_excel(f,'Abbreviations',encoding = 'utf-8')
    df = df.iloc[:,0:2] # input dateframe with first column as abbreviations and second column as full names

    for i in range(df.shape[0]):
        if type(df.iloc[i,1]) != float:
                abbr.append(df.iloc[i,0])
                full.append(df.iloc[i,1])
# form a dictionary to retrive full names
    full_dic = {}
    for i in range(len(full)):
        full_dic[abbr[i]] = full[i]
    full_dic['DOB'] = 'DATE OF BIRTH' # add 'DOB' as 'DATE OF BIRTH'

df_all = pd.DataFrame([], columns =['Original', 'Modified', 'Sentence','RMS'], dtype = str)
df_all_date = pd.DataFrame([], columns=['Original', 'Modified', 'Sentence', 'RMS'], dtype=str)
for i in tqdm(reports):
    date = getDate(i)
    full_name = fullName(i)
    df1 = pd.DataFrame(date, columns =['Original', 'Modified', 'Sentence','RMS'], dtype = str)
    df2 = pd.DataFrame(full_name, columns =['Original', 'Modified', 'Sentence','RMS'], dtype = str)
    # df = df1.append(df2)
    df_all_date = df_all_date.append(df1)
    df_all = df_all.append(df2)
    # print(*date, sep = "\n")
    # print(*full_name, sep = "\n")
    # print('##############################:',i)
df_all.to_csv('abbreviation_convert_log.csv')
df_all_date.to_csv('date_convert_log.csv')
