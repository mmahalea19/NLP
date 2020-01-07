import re
from spellchecker import SpellChecker
from nltk .tokenize import TweetTokenizer
import nltk
from textblob import TextBlob

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

def find_words(filename):
    no_words = 0
    tknzr = TweetTokenizer()
    with open(filename) as fi:
        for i, line in enumerate(fi):
            line_tok = tknzr.tokenize(line)
            no_words += len(line_tok)
    return no_words


def find_entities(filename):
    no_entities = 0
    tknzr = TweetTokenizer()
    with open(filename) as fi:
        for i, line in enumerate(fi):
            line_tok = tknzr.tokenize(line)
            ne_tree = nltk.ne_chunk(nltk.pos_tag(line_tok), binary=True)
            named_entities = []
            for tagged_tree in ne_tree:
                if hasattr(tagged_tree, 'label'):
                    entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  #
                    entity_type = tagged_tree.label()  # get NE category
                    named_entities.append((entity_name, entity_type))

            no_entities += len(named_entities)
    return no_entities

def find_pronouns(filename):
    no_pronouns = 0
    with open(filename) as fi:
        for i, line in enumerate(fi):
            blob = TextBlob(line)
            blob.parse()
            for w in blob.tags:
                print(w[1])
                if 'PRP' in w[1]:
                    no_pronouns += 1
    return no_pronouns

def find_repetitions(filename):
    no_repetitions = 0
    tknzr = TweetTokenizer()
    d = dict()
    with open(filename) as fi:
        for i, line in enumerate(fi):
            line_tok = tknzr.tokenize(line)
            for w in line_tok:
                if w in d:
                    no_repetitions += 1
                else:
                    d[w] = 1
    return no_repetitions

#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('punkt')
#filename = DATASET_DIR + 'enron1/spam/0103.2003-12-29.GP.spam.txt'
filename = 'test.txt'
#no_url = find_urls(filename)
#no_mistakes = find_mistakes(filename)
#no_entities = find_entities(filename)
#no_pronouns = find_pronouns(filename)
no_repetitions = find_repetitions(filename)
#print("There are " + str(no_url) + " urls")
#print("There are " + str(no_mistakes) + " mistakes")
#print("There are " + str(no_entities) + " entities")
#print("There are " + str(no_pronouns) + " pronouns")
print("There are " + str(no_repetitions) + " repetitions")
