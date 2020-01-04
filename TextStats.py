import nltk
from nltk.tokenize import word_tokenize,RegexpTokenizer
import os
import docx
from tqdm import tqdm
import re
from nltk.corpus import stopwords


def readDocx(filePath):
    file = docx.Document(filePath)
    doc = ""
    for para in file.paragraphs:
        doc = doc + para.text + '\n'
    return doc


def prepare_file(text_filepath, to_filepath):
    print('prepare file')
    content = ''
    for file in tqdm(os.listdir(text_filepath)):  # Iterate over the files
        if file.endswith('.docx') == False:
            continue
        contents = readDocx(text_filepath + file)  # Load file contents
        ids = file.replace('RMS', '').replace('M', '').replace('.docx', '').split('_')
        mater_id = ids[0].replace('-', '')
        rms = ids[1]
        content = content + '\n\n' + contents
    with open(to_filepath, mode='w', encoding='utf-8') as f:
        f.write(content)
    return content

def lexical_diversity(text):
    return len(set(text)) / len(text)

def percentage(count, total):
    return 100 * count / total


def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)


def replace_marks(string, maxsplit=0):
    # replace special marks
    # string.replace('\\n','').replace('\\t','')
    markers = "*", "/"
    regexPattern = '|'.join(map(re.escape, markers))
    return re.sub(regexPattern, ' ', string)


def main():
    nltk.download('punkt')
    nltk.download('words')
    text_filepath = './content/cleaned/'
    to_filepath = './content/all.json'
    content = ''
    #content = prepare_file(text_filepath, to_filepath)
    with open(to_filepath, mode='r', encoding='utf-8') as f:
        content = f.read()
    tokens = word_tokenize(replace_marks(content.lower()))
    report = nltk.Text(tokens)
    print('unusual_words')
    print(unusual_words(tokens))
    print('You can search words here')
    print(report.concordance("drug"))
    print(report.similar("drunk")) #nothing
    print(report.common_contexts(["male", "tall"])) #nothing
    print(report.dispersion_plot(["drug", "drunk"]))
    print('How many tokens in reports:')
    print(len(tokens))
    print('In average, every report have ... words')
    print(len(tokens)/3168)

    print('How many unique tokens in reports')
    print(len(set(tokens)))
    print('And these tokens are:')
    print(sorted(set(tokens)))

    print(' frequency distribution, it tells us the frequency of each vocabulary item in the text')
    fdist1 = nltk.FreqDist(tokens)
    print(fdist1.most_common(50))
    fdist1.plot(50, cumulative=True)

    print('lexical diversity:')
    print(lexical_diversity(tokens))

    print('lets look at the long words of a text; perhaps these will be more characteristic and informative. ' +
          'Have we succeeded in automatically extracting words that typify a text? Well, these very long words are often hapaxes (i.e., unique) ' +
          'and perhaps it would be better to find frequently occurring long words. ' +
          'This seems promising since it eliminates frequent short words (e.g., the) and infrequent long words (e.g. antiphilosophists). '+
          'Here are all words from the chat corpus that are longer than seven characters, that occur more than seven times:')
    print(sorted(w for w in set(fdist1) if len(w) > 7 and fdist1[w] > 7))
    print(sorted(w for w in set(fdist1) if len(w) > 15 and fdist1[w] > 0))

    print('A collocation is a sequence of words that occur together unusually often.')
    bi = list(nltk.bigrams(tokens))
    fdist_bigrams = nltk.FreqDist(bi)
    fdist_bigrams.plot(50, cumulative=True)

    print('the distribution of word lengths in a text')
    fdist_wordLength = nltk.FreqDist(len(w) for w in tokens)
    print(fdist_wordLength.most_common())
    fdist_wordLength.max()
    # 3
    # >> > fdist[3]
    # 50223
    # >> > fdist.freq(3)
    # 0.19255882431878046


    return 0



if __name__ == '__main__':
    main()
    print('end program')