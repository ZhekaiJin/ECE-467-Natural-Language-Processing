# -*- coding: utf-8 -*-
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from math import log
import string
import os
import sys


def text_categorizer (trainning_data, testing_data, output_file, k = 0.06):
    #Training
    k = float(k)
    stemer = PorterStemmer()
    total_words_per_category = dict() # the total count of all words that have been mapped to this class.
    doc_count_per_category = dict()   #the count of documents that have been mapped to this category Nc
    word_count_per_category = dict()  # the frequency with which each word in the document has been mapped to this category

    train_dir = open(trainning_data, 'r')
    train_dir_lines = train_dir.read().splitlines()

    for line in train_dir_lines:
        parsed_line = line.split()
        train_file_label = parsed_line[1]
        # update file counting per category
        if train_file_label in doc_count_per_category:
            doc_count_per_category[train_file_label] += 1.
        else:
            doc_count_per_category[train_file_label] = 1.

        train_file = open(parsed_line[0], 'r')
        train_file_tokenized = word_tokenize(train_file.read())
        for token in train_file_tokenized:
            stems = stemer.stem(token); #stemming

            # update word counting per category
            if (stems, train_file_label) in word_count_per_category:
                word_count_per_category[(stems, train_file_label)] += 1.
            else:
                word_count_per_category[(stems, train_file_label)] = 1.

            # update token counting per category
            if train_file_label in total_words_per_category:
                total_words_per_category[train_file_label] += 1.
            else:
                total_words_per_category[train_file_label] = 1.

    total_doc_number = sum(doc_count_per_category.values()) #the count of total documents we have learned from N
    category_list = total_words_per_category.keys()

    #Stastics --> MLE
    #P ( wi | cj ) = [ count( wi, cj ) + 1 ] / [ Σw∈V( count ( w, cj ) + 1 ) ]
    # ==> P( w | c ) = [ count( w, c ) + 1 ] / [ count( c ) + |V| ]

    #P ( ci ) = [ Num documents that have been classified as ci ] / [ Num documents ]


    test_dir = open(testing_data, 'r')
    test_dir_lines = test_dir.read().splitlines()
    result = []

    for line in test_dir_lines:
        test_file = open(line, 'r')
        test_file_tokenized = word_tokenize(test_file.read())

        test_token_cout = dict()

        for token in test_file_tokenized:
            stems = stemer.stem(token); #stemming

            if stems not in list(string.punctuation):
                if stems in test_token_cout:
                    test_token_cout[stems] += 1.
                else:
                    test_token_cout[stems] = 1.

        V = len(test_token_cout)
    #Prediction
        category_probabilities = dict()
        for category in category_list:
            MLE = 0.
            P_Ci = doc_count_per_category[category]/total_doc_number
            divisor = total_words_per_category[category] + V * k

            for word, count in test_token_cout.items():
                if (word, category) in word_count_per_category:
                    word_count_given_category = word_count_per_category[(word, category)] + k
                else:
                    word_count_given_category = k

                MLE += count * log(word_count_given_category / divisor)  #taking log turn mutiplication to addition

            category_probabilities[category] = MLE + log(P_Ci)


        decision = max(category_probabilities, key = category_probabilities.get)

        str = line + ' ' + decision + '\n'
        result.append(str)

    with open(output_file, 'w') as f:
        for line in result:
            f.write(line)

    return

if __name__ == "__main__":
    trainning_data = input('Please specify the name of the trainning_data: ')
    testing_doc = input('Please specify the name of the testing_doc: ')
    output_file = input('Please specify the output file: ')
    if (len(sys.argv) == 2):
        k = sys.argv[1]
        print("Processing")
        text_categorizer(trainning_data.strip(), testing_doc.strip(), output_file, k)
    else:
        print("Processing with default Laplace smoothing factor")
        text_categorizer(trainning_data.strip(), testing_doc.strip(), output_file)
    print("Thanks for use.")