# %%


import nltk
import re


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import string as string

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import permutations
from random import randint

import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import re
from spellchecker import SpellChecker
from nltk.tokenize import TweetTokenizer
import nltk
from textblob import TextBlob


nltk.download('stopwords')
nltk.download('punkt')

def removeWords(phrases, options):
    # print("Initial")
    # print(phrases)
    if ("link" in options):
        phrases = [re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            " ", phrase) for phrase in phrases]
        # print("After link")
        # print(phrases)
    if "symbol" in options:
        whitelist = string.ascii_letters + string.digits + ' '
        symbolRemoved = []
        for phrase in phrases:
            phrase1 = ''.join(list(map(lambda cha: cha if cha in whitelist else ' ', phrase)))
            symbolRemoved.append(phrase1)
        # print("After symbol removal")

        phrases = symbolRemoved
        #print(phrases)

    tokenized = [word_tokenize(phrase) for phrase in phrases]  # split in individual words
    # print(tokenized)
    if "stopword" in options:
        stop_words = set(stopwords.words('english'))  # get set of stopwords
        filtered = list(map(lambda phrase: [w for w in phrase if w not in stop_words],
                            tokenized))  # remove words that appear in the stopwords set
        # print("After stopword removal")

        tokenized = filtered
        # phrases=''.join(filtered)
        phrases = [' '.join(x) for x in filtered]
        #print(phrases)
    return phrases


def make_Dictionary(train_dir):
    ham_emails = [os.path.join(train_dir + "/ham", f) for f in os.listdir(train_dir + "/ham")]
    spam_emails = [os.path.join(train_dir + "/spam", f) for f in os.listdir(train_dir + "/spam")]
    all_words = []
    for mail in ham_emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                #  if i == 2:  # Body of email is only 3rd line of text file. Available for ling-spam only!!!
                words = line.split()
                all_words += words

    for mail in spam_emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                #  if i == 2:  # Body of email is only 3rd line of text file. Available for ling-spam only!!!
                words = line.split()
                all_words += words

    dictionary1 = Counter(all_words)
    # Paste code for non-word removal here(code snippet is given below)
    list_to_remove = list(dictionary1)

    for item in list_to_remove:
        if not item.isalpha():
            del dictionary1[item]
        elif len(item) == 1:
            del dictionary1[item]
    dictionary1 = dictionary1.most_common(3000)
    return dictionary1


def reduceFeatures(raw, nrFeats):
    # min max strategy, cut from both ends until the required number of features
    x = True
    while (raw.shape[1] > nrFeats):
        sums = [sum(x) for x in zip(*raw)]
        index = -1
        if x:
            index = np.argmin(sums)
        else:
            index = np.argmax(sums)
        x = not x
        raw = np.delete(raw, index, 1)
    return raw


def extract_my_features(mail_dir, filter):  # idf features
    ham_files = [os.path.join(mail_dir + "/ham", fi) for fi in os.listdir(mail_dir + "/ham")]
    spam_files = [os.path.join(mail_dir + "/spam", fi) for fi in os.listdir(mail_dir + "/spam")]
    all_files = ham_files + spam_files
    #labels = np.zeros(len(ham_files) + len(spam_files))
    #labels[len(ham_files):] = 1  ## ham=0 and spam=1. Should it be vice-versa?

    lines = []
    for fil in all_files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                    lines.append(line)
    # lines=["the house had a tiny little mouse",
    #   "the cat saw the mouse",
    #   "the mouse ran away from the house",
    #   "the cat finally ate the mouse",
    #   "the end of the mouse story"]
    lines = removeWords(lines, filter)
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # just send in all your docs here
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(lines)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                      columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)

    return tfidf_vectorizer_vectors


def LDA_PCA(raw, option, nr_components, labels=None):
    if option is "pca":
        pca = PCA(n_components=nr_components)
        x_pca = pca.fit_transform(raw)
        return x_pca
    if option is "lda":
        lda = LDA(n_components=nr_components)
        x_lda = lda.fit_transform(raw, labels)
        return x_lda


def extract_features(mail_dir, dictio, filter):  # based of number of occurences of words from dictionary

    ham_files = [os.path.join(mail_dir + "/ham", fi) for fi in os.listdir(mail_dir + "/ham")]
    spam_files = [os.path.join(mail_dir + "/spam", fi) for fi in os.listdir(mail_dir + "/spam")]
    features_matrix = np.zeros((len(ham_files) + len(spam_files), 3000))
    labels = np.zeros(len(ham_files) + len(spam_files))
    labels[len(ham_files):] = 1 ## ham=0 and spam=1. Should it be vice-versa?
    all_files = ham_files + spam_files
    docID = 0
    for fil in all_files:
        with open(fil) as fi:
            content = fi.read().splitlines()
            content = removeWords(content, filter)

            for line in content:
                #if i == 2:
                words = line.split()
                for word in words:
                    for i, d in enumerate(dictio):
                        if d[0] == word:
                            wordID = i
                            features_matrix[docID, wordID] = words.count(word)

            docID = docID + 1
    return features_matrix, labels


# Training SVM and Naive bayes classifier
def train_classifiers(classifiers, train_matrix, train_labels):
    for classifier in classifiers:
        classifier.fit(train_matrix, train_labels)
    return classifiers


def majority_voting(classifiers, classifier_labels, sample, true_labels):
    results = []
    classifier_order = range(0, len(classifiers) - 1);
    k = 3
    combinations = permutations(classifier_order, k)
    for index, i in enumerate(list(combinations)):
        majority_vote = classifiers[i[0]].predict(sample) + classifiers[i[1]].predict(sample) + classifiers[
            i[2]].predict(sample);
        final_result = np.round(majority_vote / k);
        results.append(final_result)
        print(
            "The final result for the classifiers {}|{}|{}={}".format(classifier_labels[i[0]], classifier_labels[i[1]],
                                                                      classifier_labels[i[2]],
                                                                      confusion_matrix(true_labels, final_result)))


def find_urls(filename):
    no_url = 0
    with open(filename) as fi:
        for i, line in enumerate(fi):
            ## curently we accept spaces between http : //. Correct or not?
            url = re.findall('http[s]?\s?:\s?/',
                             line)  # 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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


def printConfusion(results, true_labels, testName):
    print("The confusion matrix for test {} is".format(testName))

    print(confusion_matrix(true_labels, results))


def doStatistics(classifiers, classifierLabels, samples, true_labels, testName):
    print("Currently running test {}".format(testName))
    for index, classifier in enumerate(classifiers):
        results = classifier.predict(samples)
        print("Data for {} classifier".format(classifierLabels[index]))
        printConfusion(results, true_labels, testName)
    majority_voting(classifiers, classifierLabels, samples, true_labels)


def run_with_filter(train_dir, test_dir, classifierArray, classifierLabels, filters, feature_extractor):
    dictionary = make_Dictionary(train_dir)

    [train_matrix, train_labels] = feature_extractor(train_dir, dictionary, filters)

    classifiers = train_classifiers(classifierArray, train_matrix, train_labels)

    # Test the unseen mails for Spam
    [test_matrix, test_labels] = feature_extractor(test_dir, dictionary, filters)
    # majority_voting(classifierArray, classifierLabels, test_matrix, test_labels)
    doStatistics(classifiers, classifierLabels, test_matrix, test_labels, "Normal")


def main():

    # Create classifiers

    model1 = MultinomialNB()
    model2 = LinearSVC()
    model3 = tree.DecisionTreeClassifier()
    model4 = RandomForestClassifier(max_depth=2, random_state=0)

    # Add them to a list
    classifierArray = [model1, model2, model3, model4]
    classifierLabels = ["Muntinomial", "LinearSVC", "Decision Tree", "Random Forest"]

    # Directory selection
    train_dir = 'Datasets/enron1'
    test_dir = './Datasets/enron2'

    # Prepare feature vectors per training mail and its labels

    filters = ["stopword"]
    run_with_filter(train_dir, test_dir, classifierArray, classifierLabels, filters, extract_features)


main()

