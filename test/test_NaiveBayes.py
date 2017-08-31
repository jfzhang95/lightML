#-*- coding:utf-8 -*-
"""
@Author: Jeff Zhang
@Time:   17-8-31 上午9:34
@File:   test_NaiveBayes.py
"""

import numpy as np
import sys
import random
sys.path.append('..')
from SupervisedLearning.NaiveBayes import NaiveBayes

def textParser(text):
    import re
    regEx = re.compile(r'[^a-zA-Z]|\d')
    words = regEx.split(text)
    words = [word.lower() for word in words if len(word) > 0]
    return words


def loadSMSData(fileName):
    f = open(fileName)
    classCategory = []  # 类别标签，1表示是垃圾SMS，0表示正常SMS
    smsWords = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'ham':
            classCategory.append(0)
        elif linedatas[0] == 'spam':
            classCategory.append(1)
        # 切分文本
        words = textParser(linedatas[1])
        smsWords.append(words)
    return smsWords, classCategory


def createVocabularyList(smsWords):
    vocabularySet = set([])
    for words in smsWords:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList


def getVocabularyList(fileName):
    fr = open(fileName)
    vocabularyList = fr.readline().strip().split('\t')
    fr.close()
    return vocabularyList


def setOfWordsToVecTor(vocabularyList, smsWords):
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in smsWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return vocabMarked


def setOfWordsListToVecTor(vocabularyList, smsWordsList):
    vocabMarkedList = []
    for i in range(len(smsWordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, smsWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList


# 导入数据
filename = '../Data/emails/training/SMSCollection.txt'
smsWords, classLables = loadSMSData(filename)

# 交叉验证
testWords = []
testWordsType = []

testCount = 1000
for i in range(testCount):
    randomIndex = int(random.uniform(0, len(smsWords)))
    testWordsType.append(classLables[randomIndex])
    testWords.append(smsWords[randomIndex])
    del (smsWords[randomIndex])
    del (classLables[randomIndex])

vocabularyList = createVocabularyList(smsWords)
print("生成语料库！")
trainMarkedWords = setOfWordsListToVecTor(vocabularyList, smsWords)
print("数据标记完成！")
# 转成array向量
trainMarkedWords = np.array(trainMarkedWords)
print( "数据转成矩阵！")


# 训练模型
naiveBayes = NaiveBayes()
naiveBayes.fit(trainMarkedWords, classLables)

errorCount = 0.0
for i in range(testCount):
    smsType = naiveBayes.predict(vocabularyList, testWords[i])
    print('预测类别：', smsType, '实际类别：', testWordsType[i])
    if smsType != testWordsType[i]:
        errorCount += 1

print('错误个数：', int(errorCount), '错误率：', errorCount / testCount)


# 加载测试数据
print('加载测试数据......')
filename = '../Data/emails/test/test.txt'
smsWords_test, classLables_test = loadSMSData(filename)

smsType = naiveBayes.predict(vocabularyList,smsWords_test[0])
print('测试数据')
print('预测类别：', smsType, '实际类别：', classLables_test[0])
