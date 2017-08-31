#-*- coding:utf-8 -*-
"""
@Author: Jeff Zhang
@Time:   17-8-31 上午9:34
@File:   test_NaiveBayes.py
"""

import re

def textParse(text):
    text_ = re.split(r'\W*', text)
    return [tok.lower() for tok in text if len(tok) > 2]


docList = []
classList = []
fullText = []

for i in range(1, 26):
    wordList = open('../Data/email4bayes/spam/%d.txt' % i, 'r').read()
    print(wordList)