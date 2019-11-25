import nltk  
import numpy as np  
import random  
import string
import re
# import pdb; pdb.set_trace()
article_text = open("charlotte.txt","r").readlines()

article_text = re.sub(r'[^A-Za-z. ]', '', article_text[0]) 

ngrams = {}  
words = 2


words_tokens = nltk.word_tokenize(article_text)  
for i in range(len(words_tokens)-words):  
    seq = ' '.join(words_tokens[i:i+words])
    # print(seq)
    if  seq not in ngrams.keys():
        ngrams[seq] = []
    ngrams[seq].append(words_tokens[i+words])
# import pdb; pdb.set_trace()
def predict_word(word1,word2):
    curr_sequence = ' '.join(word1,word2)  
    output = curr_sequence  
    for i in range(1):  
        if curr_sequence not in ngrams.keys():
            break
        possible_words = ngrams[curr_sequence]
        next_word = possible_words[random.randrange(len(possible_words))]
        output += ' ' + next_word
        seq_words = nltk.word_tokenize(output)
        curr_sequence = ' '.join(seq_words[len(seq_words)-words:len(seq_words)])

    print(output)
    return next_word