# %%


import nltk
import re

import pickle
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
import time


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

# def train_idf_vectorizer(lines):
#     tfidf_vectorizer = TfidfVectorizer(use_idf=True)
#
#     # just send in all your docs here
#     tfidf_vectorizer = tfidf_vectorizer.fit(lines)
#     # first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]
#
#     # place tf-idf values in a pandas data frame
#     # df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
#     #                   columns=["tfidf"])
#     # df.sort_values(by=["tfidf"], ascending=False)
#     vocabulary = tfidf_vectorizer.get_feature_names()
#     return tfidf_vectorizer

def run_vectorizer_idf(vectorizer,mail_dir, filter,train=False):  # idf features

    all_files ,true_labels=get_ham_spam_files(mail_dir)
    #labels = np.zeros(len(ham_files) + len(spam_files))
    #labels[len(ham_files):] = 1  ## ham=0 and spam=1. Should it be vice-versa?

    lines = []
    for fil in all_files:
        with open(fil) as fi:
            preprocessed=removeWords([fi.read()], filter)
            lines.append(preprocessed[0])
    if train:
        vectorizer.fit(lines)

    vocabulary = vectorizer.get_feature_names()

    return  vectorizer,vectorizer.transform(lines).todense().getA(),true_labels



def LDA_PCA(raw, option, nr_components, labels=None):
    if option is "pca":
        pca = PCA(n_components=nr_components)
        x_pca = pca.fit_transform(raw)
        return x_pca
    if option is "lda":
        lda = LDA(n_components=nr_components)
        x_lda = lda.fit_transform(raw, labels)
        return x_lda
    return raw


def get_ham_spam_files(mail_dir):

    ham_files = [os.path.join(mail_dir + "/ham", fi) for fi in os.listdir(mail_dir + "/ham")]
    spam_files = [os.path.join(mail_dir + "/spam", fi) for fi in os.listdir(mail_dir + "/spam")]
    labels = np.zeros(len(ham_files) + len(spam_files))
    labels[len(ham_files):] = 1  ## ham=0 and spam=1. Should it be vice-versa?
    all_files = ham_files + spam_files
    return all_files, labels

def extractNormalFeatures(filename,features_matrix,docID,dictio):
    with open(filename) as fi:
        content = fi.read().splitlines()
        content = removeWords(content, filter)

        for line in content:
            # if i == 2:
            words = line.split()
            for word in words:
                for i, d in enumerate(dictio):
                    if d[0] == word:
                        wordID = i
                        features_matrix[docID, wordID] = words.count(word)
def extract_features(mail_dir, dictio, filter):  # based of number of occurences of words from dictionary

    all_files,true_labels=get_ham_spam_files(mail_dir)
    features_matrix = np.zeros((len(all_files), 3000))

    docID = 0
    for fil in all_files:
        extractNormalFeatures(fil,features_matrix,docID,dictio)

        docID = docID + 1
    return features_matrix, true_labels


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
        conf_mat = confusion_matrix(true_labels, final_result)



        print(
            "The final result for the classifiers {}|{}|{}=\n{}".format(classifier_labels[i[0]], classifier_labels[i[1]],
                                                                      classifier_labels[i[2]],
                                                                        conf_mat ))
        print("The acc. is {}".format((conf_mat[0][0] + conf_mat[1][1]) / conf_mat.sum()))


def find_urls(filename):
    filename.seek(0)
    no_url = 0
    for i, line in enumerate(filename):
        ## curently we accept spaces between http : //. Correct or not?
        url = re.findall('http[s]?\s?:\s?/',
                         line)  # 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        if len(url) != 0:
            no_url += 1
    return no_url


import pkg_resources
from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
def find_mistakes2(filename):
    no_mistakes = 0
    tknzr = TweetTokenizer()

    with open(filename) as fi:
        for i, line in enumerate(fi):

            suggestions = sym_spell.lookup_compound(line, max_edit_distance=2, transfer_casing=True)
            print(suggestions)
            no_mistakes+=suggestions[0].distance
    return no_mistakes


def find_mistakes(filename):
    filename.seek(0)
    no_mistakes = 0

    spell = SpellChecker()
    tknzr = TweetTokenizer()
    no_mistakes = 0
    for i, line in enumerate(filename):
        line_tok = tknzr.tokenize(line)
        for w in line_tok:
            correction = spell.correction(w)
            if correction == w:
                no_mistakes += 1
    return no_mistakes


def find_words(filename):
    filename.seek(0)
    no_words = 0
    tknzr = TweetTokenizer()
    for i, line in enumerate(filename):
        line_tok = tknzr.tokenize(line)
        no_words += len(line_tok)
    return no_words


def find_entities(filename):
    filename.seek(0)
    no_entities = 0
    tknzr = TweetTokenizer()
    for i, line in enumerate(filename):
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
    filename.seek(0)
    no_pronouns = 0
    for i, line in enumerate(filename):
        blob = TextBlob(line)
        blob.parse()
        for w in blob.tags:
            if 'PRP' in w[1]:
                no_pronouns += 1
    return no_pronouns

from collections import Counter

def find_pronouns2(filename):
    from collections import Counter
    no_pronouns = 0
    with open(filename) as fi:
        contet=fi.read()

        tokens = nltk.word_tokenize(contet.lower())
        text = nltk.Text(tokens)
        tags = nltk.pos_tag(text)
        counts = Counter(tag for word, tag in tags)
        no_pronouns+=counts["PRP"]
        print(1)
    return no_pronouns


def find_repetitions(filename):
    filename.seek(0)
    no_repetitions = 0
    tknzr = TweetTokenizer()
    d = dict()
    for i, line in enumerate(filename):
        line_tok = tknzr.tokenize(line)
        for w in line_tok:
            if w in d:
                no_repetitions += 1
            else:
                d[w] = 1
    return no_repetitions


def enhanced_feature_vector(train_files):
    train_matrix = []

    tknzr = TweetTokenizer()
    spell = SpellChecker()

    for file in train_files:

        no_repetitions = 0
        no_pronouns = 0
        no_entities = 0
        no_words = 0
        no_mistakes = 0
        no_url = 0

        with open(file) as fi:
            d = dict()
            for i, line in enumerate(fi):
                blob = TextBlob(line)
                blob.parse()
                url = re.findall('http[s]?\s?:\s?/',
                                 line)  # 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                if len(url) != 0:
                    no_url += 1

                line_tok = tknzr.tokenize(line)
                no_words = len(line_tok)
                for word in line_tok:

                    correction = spell.correction(word)
                    if correction != word:
                        no_mistakes += 1

                    if word in d:
                        no_repetitions += 1
                    else:
                        d[word] = 1

                for w in blob.tags:
                    if 'PRP' in w[1]:
                        no_pronouns += 1

                ne_tree = nltk.ne_chunk(nltk.pos_tag(line_tok), binary=True)
                named_entities = []
                for tagged_tree in ne_tree:
                    if hasattr(tagged_tree, 'label'):
                        entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  #
                        entity_type = tagged_tree.label()  # get NE category
                        named_entities.append((entity_name, entity_type))
                no_entities += len(named_entities)

        train_matrix.append([no_url, no_mistakes, no_words, no_entities, no_repetitions, no_pronouns])

    return train_matrix

def printConfusion(results, true_labels, testName):
    print("The confusion matrix for test {} is".format(testName))
    conf_mat=confusion_matrix(true_labels, results)
    print(conf_mat)
    print("The acc. is {}".format((conf_mat[0][0]+conf_mat[1][1])/conf_mat.sum()))


def doStatistics(classifiers, classifierLabels, samples, true_labels, testName):
    print("Currently running test {}".format(testName))
    for index, classifier in enumerate(classifiers):
        results = classifier.predict(samples)
        print("Data for {} classifier".format(classifierLabels[index]))
        printConfusion(results, true_labels, testName)
    majority_voting(classifiers, classifierLabels, samples, true_labels)



def run_with_filter(train_dir, test_dir, filters, feature_extractor,nrfeatures=None,extra_feature_reduction=None):
    model1 = MultinomialNB()
    model2 = LinearSVC()
    model3 = tree.DecisionTreeClassifier()
    model4 = RandomForestClassifier(max_depth=2, random_state=0)

    classifierArray = [model1, model2, model3, model4]
    classifierLabels = ["Muntinomial", "LinearSVC", "Decision Tree", "Random Forest"]
    dictionary = make_Dictionary(train_dir)

    [train_matrix, train_labels] = feature_extractor(train_dir, dictionary, filters)
    if nrfeatures is not None:
        train_matrix=reduceFeatures(train_matrix,nrfeatures)
    if extra_feature_reduction is not None:
         train_matrix=LDA_PCA(train_matrix,extra_feature_reduction,10)
    classifiers = train_classifiers(classifierArray, train_matrix, train_labels)
    filename="./Out/"+"_".join(filters)+"_"+str(nrfeatures)+"_"+str(extra_feature_reduction);
    with open(filename,'wb') as handler:
        pickle.dump(classifiers,handler,protocol=pickle.HIGHEST_PROTOCOL)

    # Test the unseen mails for Spam
    [test_matrix, test_labels] = feature_extractor(test_dir, dictionary, filters)
    if nrfeatures is not None:
        test_matrix=reduceFeatures(test_matrix,nrfeatures)
    if extra_feature_reduction is not None:
        test_matrix = LDA_PCA(test_matrix, extra_feature_reduction, 10)

    # majority_voting(classifierArray, classifierLabels, test_matrix, test_labels)
    doStatistics(classifiers, classifierLabels, test_matrix, test_labels, "Normal")
def run_with_filter_idf(train_dir, test_dir, filters,nrfeatures=None,extra_feature_reduction=None):
    model1 = GaussianNB()
    model2 = LinearSVC()
    model3 = tree.DecisionTreeClassifier()
    model4 = RandomForestClassifier(max_depth=2, random_state=0)

    classifierArray = [model1, model2, model3, model4]
    classifierLabels = ["Muntinomial", "LinearSVC", "Decision Tree", "Random Forest"]
    dictionary = make_Dictionary(train_dir)
    transforme_dic=[item[0] for item in dictionary]
    vectorizer = TfidfVectorizer(use_idf=True,vocabulary=transforme_dic)

    vectorizer,train_matrix, train_labels = run_vectorizer_idf(vectorizer,train_dir, filters,train=True)
    if nrfeatures is not None:
        train_matrix=reduceFeatures(train_matrix,nrfeatures)
    for row in train_matrix:
        for i in row:
            if (i < 0 ):
                print(i)
    if extra_feature_reduction is not None:
         train_matrix=LDA_PCA(train_matrix,extra_feature_reduction,10)
    for row in train_matrix:
        for i in row:
            if (i < 0 ):
                print(i)
    classifiers = train_classifiers(classifierArray, train_matrix, train_labels)
    filename="./Out/"+"_".join(filters)+"_"+str(nrfeatures)+"_"+str(extra_feature_reduction);
    with open(filename,'wb') as handler:
        pickle.dump(classifiers,handler,protocol=pickle.HIGHEST_PROTOCOL)

    # Test the unseen mails for Spam
    vectorizer,test_matrix, test_labels = run_vectorizer_idf(vectorizer,test_dir, filters)
    if nrfeatures is not None:
        test_matrix=reduceFeatures(test_matrix,nrfeatures)
    if extra_feature_reduction is not None:
        test_matrix = LDA_PCA(test_matrix, extra_feature_reduction, 10)

    # majority_voting(classifierArray, classifierLabels, test_matrix, test_labels)
    doStatistics(classifiers, classifierLabels, test_matrix, test_labels, "Normal")

def run_with_new_features(train_dir, test_dir):
    model1 = MultinomialNB()
    model2 = LinearSVC()
    model3 = tree.DecisionTreeClassifier()
    model4 = RandomForestClassifier(max_depth=2, random_state=0)

    classifierArray = [model1, model2, model3, model4]
    classifierLabels = ["Multinomial", "LinearSVC", "Decision Tree", "Random Forest"]
    train_files, train_labels = get_ham_spam_files(train_dir)
    #train_matrix = [[find_urls(file), find_mistakes(file), find_words(file),
    #                 find_entities(file), find_repetitions(file), find_pronouns(file)] for file in train_files]
    train_matrix = enhanced_feature_vector(train_files)

    classifiers = train_classifiers(classifierArray, train_matrix, train_labels)

    # Test the unseen mails for Spam
    [test_files, test_labels] = get_ham_spam_files(test_dir)

    test_matrix = enhanced_feature_vector(test_files)
    #for file in test_files:
    #    with open(file) as fi:
    #        test_matrix.append([find_urls(fi), find_mistakes(fi), find_words(fi),
    #                 find_entities(fi), find_repetitions(fi), find_pronouns(fi)])

    # majority_voting(classifierArray, classifierLabels, test_matrix, test_labels)
    doStatistics(classifiers, classifierLabels, test_matrix, test_labels, "Normal")


def main():

    # Create classifiers

    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    # Add them to a list
    nltk.download('words')

    # Directory selection
    train_dir = './Datasets/enron1'
    test_dir = './Datasets/enron2'

    # Prepare feature vectors per training mail and its labels

    # filters=[]
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=3000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=2000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=1000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=500)
    # filters = ["stopword"]
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=3000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=2000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=1000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=500)
    # filters = ["link"]
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=3000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=2000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=1000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=500)
    # filters = ["symbol"]
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=3000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=2000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=1000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=500)
    # filters = ["stopword","link","symbol"]
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=3000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=2000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=1000)
    # run_with_filter_idf(train_dir, test_dir, filters, nrfeatures=500)

    filters = []
    run_with_filter_idf(train_dir, test_dir, filters, extra_feature_reduction="pca")






    run_with_new_features(train_dir, test_dir)




main()

