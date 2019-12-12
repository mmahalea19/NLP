import re
from spellchecker import SpellChecker
from nltk .tokenize import TweetTokenizer

DATASET_DIR = "./Dataset/Enron_PRE/"


def find_urls(filename):
    no_url = 0
    with open(filename) as fi:
        for i, line in enumerate(fi):
            ## curently we accept spaces between http : //. Correct or not?
            url = re.findall('http[s]?\s?:\s?/', line) # 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            if len(url) != 0:
                no_url += 1
    return no_url

def find_mistakes(filename):
    no_mistakes = 0
    spell = SpellChecker()
    tknzr = TweetTokenizer()
    with open(filename) as fi:
        for i, line in enumerate(fi):
            line_tok = tknzr.tokenize(line)
            for w in line_tok:
                correction = spell.correction(w)
                if correction == w:
                    no_mistakes += 1
    return no_mistakes




filename = DATASET_DIR + 'enron1/spam/0103.2003-12-29.GP.spam.txt'
no_url = find_urls(filename)
no_mistakes = find_mistakes(filename)
print("There are " + str(no_url) + " urls")
print("There are " + str(no_mistakes) + " mistakes")

