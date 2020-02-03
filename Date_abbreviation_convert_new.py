import re
import datetime
import json
import spacy
import pandas as pd
import sys
from tqdm import tqdm
import os
import sklearn

# check if write out file already exist
if os.path.exists('date_abbreviation_convert.csv'):
    os.remove('date_abbreviation_convert.csv')

# load in spacy English module
nlp = spacy.load('en_core_web_sm')

### regular expression for RMS
rms_pattern = r'\bRMS[;,.:/]?\ [\s]?[0-9]{2}[-/,.]?[0-9]*\b|\bRMS[;,.:/]?[\s]*[0-9]{2}[-/,.]?[0-9]+\b|[_]RMS[;,.:/]?[0-9]{2}[-/,.]?[0-9]+[_]'

### regular expression extract date between 1900-2100
date_slash_Y = r'\b[0-1][0-9]\/[0-3][0-9]\/[1-2][09][0-9]{2}|\b[1-9]\/[0-3][0-9]\/[1-2][09][0-9]{2}|\b[1-9]\/[1-9]\/[1-2][09][0-9]{2}|\b[1-2][09][0-9]{2}\/[0-1][0-9]\/[0-3][0-9]'
date_slash_y = r'|\b[0-1][0-9]\/[0-3][0-9]\/[0-9]{2}\b|\b[0-1][0-9]\/[1-9]\/[0-9]{2}\b|\b[1-9]\/[0-3][0-9]\/[0-9]{2}\b|\b[1-9]\/[1-9]\/[0-9]{2}\b'
date_slash_NY = r'|\b[1-9]\/[0-3][0-9]\b|\b[0-1][0-9]\/[0-3][0-9]\b|\b[1-9]\/[1-9]\b'
date_dash_Y = r'|\b[0-1][0-9]\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[1-9]\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[1-9]\-[1-9]\-[1-2][09][0-9]{2}|\b[A-Z][a-z]{2,8}\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[A-Z][a-z]{2,8}\-[0-3][0-9]\-[0-9]{2}'
date_dash_y = r'|\b[0-1][0-9]\-[0-3][0-9]\-[0-9]{2}\b|\b[1-9]\-[0-3][0-9]\-[0-9]{2}\b|\b[1-9]\-[1-9]\-[0-9]{2}\b'
date_dash_NY = r'|\b[JFMASOND][aepuco][nbrlgptvcy]\-[0-3][0-9]\b'
date_Noseparate = r'|\b[0-1][0-9][0-3][0-9][1-2][09][0-9]{2}'
date_period = r'|\b[0-1][0-9]\.[0-3][0-9]\.[0-9]{2}\b|\b[1-9]\.[0-3][0-9]\.[0-9]{2}\b|\b[1-9]\.[1-9]\.[0-9]{2}\b'
date_space = r'|\b[0-1][0-9]\ [0-3][0-9]\ [0-9]{2}\b|\b[0-1][0-9]\ [0-3][0-9]\ [1-2][09][0-9]{2}'
date_written = r'|\b[A-Z][A-Za-z]{2,8}[.,]?\ [0-3][0-9][.,]?[\s]?[1-2][09][0-9]{2}|\b[A-Z][A-Za-z]{2,8}[.]?[\s]?[1-9],[\s]?[1-2][09][0-9]{2}\b|\b[A-Z][A-Za-z]{2,8}\,[1-2][09][0-9]{2}\b|\b[A-Z][A-za-z]{2,8}.\ [0-3][0-9]\,[\s]?[1-2][09][0-9]{2}\b|\b[0-3][0-9][.,]?\ [A-Z][A-za-z]{2,8}[.,]?\ [1-2][09][0-9]{2}\b|\b[A-Z][A-Za-z]{2,8}\ [0-3]?[0-9][tT][hH][,.]?\ [1-2][09][0-9]{2}\b'
date_written_NY = r'|\b[A-Z][a-z]{2,8}\ [0-3][0-9]th\b'
date_written_ND = r'|\b[0-9]?[0-9]?[\s]?[JFMASOND][aepuco][nbrlgptvcy]\w*\ [1-2][09][0-9]{2}'
hours = r'[\d]{4,6}[\s]?[H][O]?[U]?[R][S]?\b|[\d]{0,2}[:]?[\d]+[\s]?[AaPp][Mm]\b'
### combine date pattern (must follow the sequence above)
date_pattern = date_slash_Y + date_slash_y + date_slash_NY + date_dash_Y + date_dash_y + date_dash_NY + date_Noseparate + date_period + date_space + date_written + date_written_NY + date_written_ND
#date_pattern = r'\b[0-1][0-9]\/[0-3][0-9]\/[1-2][09][0-9]{2}|\b[1-9]\/[0-3][0-9]\/[1-2][09][0-9]{2}|\b[1-9]\/[1-9]\/[1-2][09][0-9]{2}|\b[0-1][0-9]\/[0-3][0-9]\/[0-9]{2}|\b[0-1][0-9]\/[1-9]\/[0-9]{2}|\b[1-9]\/[0-3][0-9]\/[0-9]{2}|\b[1-2][09][0-9]{2}\/[0-1][0-9]\/[0-3][0-9]|\b[1-9]\/[1-9]\/[0-9]{2}|\b[1-9]\/[0-3][0-9]\b|\b[0-1][0-9]\/[0-3][0-9]|\b[1-9]\/[1-9]|\b[0-1][0-9]\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[1-9]\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[0-1][0-9]\-[0-3][0-9]\-[0-9]{2}|\b[1-9]\-[0-3][0-9]\-[0-9]{2}|\b[1-9]\-[1-9]\-[0-9]{2}|\b[JFMASOND][aepuco][nbrlgptvcy]\-[0-3][0-9]\b|\b[A-Z][a-z]{2,8}\-[0-3][0-9]\-[1-2][09][0-9]{2}|\b[A-Z][a-z]{2,8}\-[0-3][0-9]\-[0-9]{2}|\b[0-1][0-9][0-3][0-9][1-2][09][0-9]{2}|\b[0-1][0-9]\.[0-3][0-9]\.[0-9]{2}\b|\b[1-9]\.[0-3][0-9]\.[0-9]{2}\b|\b[1-9]\.[1-9]\.[0-9]{2}\b|\b[0-1][0-9]\ [0-3][0-9]\ [0-9]{2}\b|\b[0-1][0-9]\ [0-3][0-9]\ [1-2][09][0-9]{2}|\b[A-Z][a-z]{2,8}\ [0-3][0-9]\,[1-2][09][0-9]{2}\b|\b[A-Z][a-z]{2,8}\ [1-9]\,[1-2][09][0-9]{2}\b|\b[A-Z][a-z]{2,8}\ [0-3][0-9]\b|\b[A-Z][a-z]{2,8}\ [0-3][0-9]th\b|\b[A-Z][a-z]{2,8}\,[1-2][09][0-9]{2}\b|\b[A-Z][a-z]{2,8}.\ [0-3][0-9]\,[1-2][09][0-9]{2}\b'

### re for extracting abbrevations
A_pattern = r'\bA/M\b|\bAgg\.|\bAKA\b|\bAPPROX\b|\bappt\.|\bAPT\b|\bARR\b|\bASST\b|\bATM\b|\bATT\.|\bAUTO\b|\bAV\.|\bave\.'
B_pattern = r'|\b B \b|\bB\&E\b|\bb\/f\b|\bB\/M\b|\bBAC\b|\bBCI\b|\bBF\b|\bBLDG\b|\bblk\b|\bbrn\b'
C_pattern = r'|\bC\/N\b|\bC\/W\b|\bCAPT\b|\bCCDCFS\b|\bCCS\b|\bCK\b|\bCleve\b|\bCMHA\b|\bCO\.|\bCOMM\b|\bCOMP\b|\bCONF\.|\bCPD\b|\bCS\b|\bct\.|\bCWS\b'
D_pattern = r'|\bDEPT\.|\bDet\.|\bDETS\b|\bDH\b|\bDHS\b|\bDist\b|\bDK\b|\bdob\b|\bDOB\b|\bdr\b|\bDR\.|\bDVD\b'
E_pattern = r'|\bE\.|\bE\/B\b|\bEMS\b|\bER\b|\bETA\b'
F_pattern = r'|\bFBI\b|\bFel\.|\bFIR\b|\bFT\b|\bFYI\b'
G_pattern = r'|\bGOA\b|\bGSI\b|\bGTMV\b'
H_pattern = r'|\bH\/M\b|\bHGT\b|\bhosp\.|\bHR\b|\bhrs\b|\bHS\b|\bHT\b|\bHTS\b'
I_pattern = r'|\bI\-|\bID\'D\b|\binfo\b|\bINT\b|\bINTOX\b|\bINVEST\b'
J_pattern = r'|\bJC\b|\bjc\b|\bJr\.'
L_pattern = r'|\blbs\.|\bLCI\b|\bLIC\b|\bLIEUT\b|\bLN\.|\bLt\.'
M_pattern = r'|\bM\b|\bm\/t\/E\b|\bM\/T\/E\/S\b|\bMED\b|\bMGS\b|\bMHMC\b|\bMIN\b|\bMISC\b|\bMLK\b|\bMO\b|\bMTE\b|\bMV\b|\bM\'S\b'
N_pattern = r'|\bN\.S\.|\bN\/B\b|\bN\/E\b|\bN\/S\b|\bN\/W\b|\bNARC.\ POSS\b|\bNB\b|\bNFI\b|\bNFIL\b|\bNMD\b'
O_pattern = r'|\bOFF\.|\bOH\b|\bOIC\b|\bORC\b' # there is a little space bwt 'PIC' and 'OIC'
P_pattern = r'|\bP\.A\.|\bP\.O\.|\bPC\b|\bPCS\b|\bPD\b|\bpg\b|\bPH\b|\bPIO\b|\bPOSS\b|\bPROP\b|\bPROS\b|\bPROSC\b'
R_pattern = r'|\bR\/P\b|\bRD\b|\bREC\b|\bREC\'D\b|\bRECV\'D\b|\bREP\b|\bREPTS\b|\bRN\b|\bRP\b|\bRPT\b'
S_pattern = r'|\bs\.|S\.O\.|\bS\/B\b|\bS\/W\b|\bS\/W\/F\b|\bS\/W\/M\b|\bSANE\b|\bSCU\b|\bSGT\b|\bSIO\b|\bSLMC\b|\bSS\#|\bST\b|\bSUB\b|\bSUBJ\.|\bSUS\b|\bSUSP\b|\bSVCH\b'
T_pattern = r'|\bT\ \[&]\ R\b|\bT\ AND\ R\b|\bT\.V\.|\bT\/R\b|\bTHURS\b|\bTRAFF\b'
U_pattern = r'|\bUH\b|\bUSPS\b|\bUTL\b'
V_pattern = r'|\b V \b|\bVEH\b|\bVIC\b|\bVict\.|\bVICTS\b|\bVICT\'S\b|\bVS\b'
W_pattern = r'|\bW\.|\bW\/B\b|\bW[/]|\bW\/M\b|\bWGT\b|\bWIT\b|\bWITN\b|\bWM\b|\bWTS\b'
Y_pattern = r'|\bYR\b|\bYRS\.'
Z_pattern = r'|\bZ\/C\b|\bZC\b'
new_pattern = A_pattern + B_pattern + C_pattern + D_pattern + E_pattern + F_pattern + G_pattern + H_pattern + I_pattern + J_pattern + L_pattern + M_pattern + N_pattern + O_pattern + P_pattern + R_pattern + S_pattern + T_pattern + U_pattern + V_pattern + W_pattern + Y_pattern + Z_pattern

def ner(nlp_text_file): # load spacy to formate the context and split the sentence
    sentence = []
    for num,sen in enumerate(nlp_text_file.sents):
        sentence.append(str(sen))
    return(sentence)

def getDate(sentence_text): # retrive datetime to 'DD-MM-YYYY HH:MM:SS'
    date_modified = []
# pattern_hidden = [date_slash_NY,date_dash_NY,date_Noseparate,date_written_NY]
    date_time = re.search(date_pattern,sentence_text)
    if date_time != None:
        hour = re.search(hours,sentence_text)
        if hour != None:
            united = date_time.group() + ' ' + hour.group()
            united = pd.to_datetime(united,errors='ignore')
            if type(united) != str:
                united = datetime.datetime.strftime(united,'%m-%d-%Y %H:%M:%S')
                date_modified = sentence_text.replace(str(date_time.group()),united)
            else:
                date_modified = sentence_text #### Some pattern of dates pandas can not detect, return original sentence
        else:
            united = pd.to_datetime(date_time.group(),errors='ignore')
            if type(united) != str:
                united = datetime.datetime.strftime(united,'%m-%d-%Y %H:%M:%S')
                date_modified = sentence_text.replace(str(date_time.group()),' ' + united + ' ')
            else:
                date_modified = sentence_text #### Some pattern of dates pandas can not detect, return original sentence
    else:
        date_modified = sentence_text #### return if there is no date detected in sentence
    return(date_modified)

# regular expression pattern converting abbreviation to full name
def fullName(sentence_text):
    cut_sen = sentence_text # cut the abbreviation sentence if it is found
    rec_sen = sentence_text # replace the abbreviation in sentence
    find = re.search(new_pattern,sentence_text)
    while find != None:
        cut_sen = cut_sen.replace(find.group(),'')
        rec_sen = rec_sen.replace(find.group(),full_dic.get(str(find.group())))
        find = re.search(new_pattern,cut_sen)             
    return(rec_sen)

##### open document json
with open(r'C:\Users\Administrator\Desktop\train_sak.json','r') as f: # input the report file.json
    reports = []
    file = json.load(f)
    for elem in file:
        if elem['document']:
            reports.append(elem['document'])

#####Convert Abbreviations to full names####
with open(r'C:\Users\Administrator\Desktop\NIJ_AbbreviationList.xlsx','rb') as f: # input the Abbreviation file.xlsx
    abbr = []
    full = []
# top priority abbreviation
    df_top = pd.read_excel(f,'Top priority',encoding = 'utf-8').iloc[:,0:2]
    df_less = pd.read_excel(f,'less important',encoding = 'utf-8').iloc[:,0:2]
    df_org = pd.read_excel(f,'Organizations.Locations',encoding = 'utf-8').iloc[:,0:2]
    df = df_top.append(df_less)
    df = df.append(df_org)
# input dateframe with first column as abbreviations and second column as full names
    for i in range(df.shape[0]):
        if type(df.iloc[i,1]) != float:
                abbr.append(df.iloc[i,0])
                full.append(df.iloc[i,1])
    #print(full)                
# form a dictionary to retrive full names
    full_dic = {}
    for i in range(len(full)):
        if full[i] != None:
            full_dic[abbr[i].strip()] = full[i]
# fill the dictionary with special abbreviations             
    full_dic['DOB'] = 'DATE OF BIRTH' # add 'DOB' as 'DATE OF BIRTH'
    full_dic['C/W'] = 'cwith' # three same 'C/W' as 'cwith'
    full_dic['DR.'] = 'DRIVE OR DOCTOR'
    full_dic['Jr.'] = 'Junior'
    full_dic[' B '] = 'Black'
    full_dic[' V '] = 'VICTIM'
    full_dic['VICT\'S'] = 'VICTIM\'S'

# open document json
with open(r'C:\Users\Administrator\Desktop\train_sak.json','r') as f: # input the report file.json
    reports = []
    file = json.load(f)
    for elem in file:
        if elem['document']:
            reports.append(elem['document'])
            
# output the modification
output = []
for i in tqdm(reports):
    doc = nlp(i)
    sen = ner(doc)
    d = ''
    for j in range(len(sen)):
        date_modified = getDate(sen[j])
        abbr_modified = fullName(str(date_modified))
        d = d + abbr_modified
    #print(d)
    output.append([i,d])

df_out = pd.DataFrame(output,columns=['original','modified'])
df_out.to_csv('date_abbreviation_convert.csv')