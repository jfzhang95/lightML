#-*- coding:utf-8 -*-
"""
@Author: Jeff Zhang
@Time:   17-8-31 上午9:29
@File:   NaiveBayes.py
"""


import numpy as np


class NaiveBayes(object):
    def __init__(self):
        pass

    def setOfWordsToVecTor(self, vocabularyList, smsWords):
        """
        SMS内容匹配预料库，标记预料库的词汇出现的次数
        :param vocabularyList:
        :param smsWords:
        :return:
        """
        vocabMarked = [0] * len(vocabularyList)
        for smsWord in smsWords:
            if smsWord in vocabularyList:
                vocabMarked[vocabularyList.index(smsWord)] += 1
        return vocabMarked

    def setOfWordsListToVecTor(self, vocabularyList, smsWordsList):
        """
        将文本数据的二维数组标记
        :param vocabularyList:
        :param smsWordsList:
        :return:
        """
        vocabMarkedList = []
        for i in range(len(smsWordsList)):
            vocabMarked = self.setOfWordsToVecTor(vocabularyList, smsWordsList[i])
            vocabMarkedList.append(vocabMarked)
        return vocabMarkedList

    def fit(self, trainMarkedWords, trainCategory):
        numTrainDoc = len(trainMarkedWords)
        numWords = len(trainMarkedWords[0])
        pSpam = sum(trainCategory) / float(numTrainDoc)
        wordsInSpamNum = np.ones(numWords)
        wordsInHealthNum = np.ones(numWords)
        SpamWordsNum = 2.0
        HealthWordsNum = 2.0
        for i in range(0, numTrainDoc):
            if trainCategory[i] == 1:
                wordsInSpamNum += trainMarkedWords[i]
                SpamWordsNum += sum(trainMarkedWords[i])  # 统计Spam中语料库中词汇出现的总次数
            if trainCategory[i] == 0:
                wordsInHealthNum += trainMarkedWords[i]
                HealthWordsNum += sum(trainMarkedWords[i])

        pWordsSpamicity = np.log(wordsInSpamNum / SpamWordsNum)
        pWordsHealthy = np.log(wordsInHealthNum / HealthWordsNum)

        self.pWordsHealthy_, self.pWordsSpamicity_, self.pSpam_ = pWordsHealthy, pWordsSpamicity, pSpam
        return self



    def predict(self, vocabularyList, testWords):
        testWordsCount = self.setOfWordsToVecTor(vocabularyList, testWords)
        testWordsMarkedArray = np.array(testWordsCount)
        # 计算P(Ci|W)，W为向量。P(Ci|W)只需计算P(W|Ci)P(Ci)
        p1 = sum(testWordsMarkedArray * self.pWordsSpamicity_) + np.log(self.pSpam_)
        p0 = sum(testWordsMarkedArray * self.pWordsHealthy_) + np.log(1 - self.pSpam_)
        if p1 > p0:
            return 1
        else:
            return 0