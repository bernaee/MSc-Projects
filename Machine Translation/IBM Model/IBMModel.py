import string
import re
import numpy as np
from math import factorial
import math

MAX_FERTILITY = 10
MAX_SENT_LEN = 5
MIN_PROB = 1.0e-12
np.seterr(divide='ignore', invalid='ignore')
#


class IBMModel:
    def __init__(self, forCorpus, engCorpus):
        self.forCorpus = forCorpus
        self.engCorpus = engCorpus
        self.engLexicon, self.maxLenEngSent = self.getWords(self.engCorpus)
        self.forLexicon, self.maxLenForSent = self.getWords(self.forCorpus)
        print('%s unique words in english corpus.' % (len(self.engLexicon)))
        print('Max length of english sentence is %s .' % self.maxLenEngSent)
        print('%s unique words in foreign corpus.' % (len(self.forLexicon)))
        print('Max length of foreign sentence is %s .' % self.maxLenForSent)

    def getWords(self, corpus):
        words = []
        maxLengthOfSent = 0
        for sentence in corpus:
            splittedSent = sentence.split()
            lenOfSent = len(splittedSent)
            if lenOfSent >= maxLengthOfSent:
                maxLengthOfSent = lenOfSent
            words = words + splittedSent
        words = list(set(words))
        return words, maxLengthOfSent

    def initTranslationProbabilities(self):
        return np.full([len(self.engLexicon), len(self.forLexicon)], 1 / len(self.engLexicon), dtype=float)
        # transProb[:,0].sum()

    def initAlignmentProbabilities(self):
        return np.full([self.maxLenForSent, self.maxLenEngSent, self.maxLenEngSent, self.maxLenForSent],
                       1 / (self.maxLenForSent), dtype=float)

    def initFertilityProbabilities(self):
        return np.full([MAX_FERTILITY, len(self.forLexicon)], 1 / MAX_FERTILITY, dtype=float)

    def IBMModel1(self, converge):
        print('IBM Model 1 is being trained...')
        self.transProb1 = self.initTranslationProbabilities()
        for con in range(0, converge):
            sTotal = np.zeros([len(self.engCorpus), len(self.engLexicon)], dtype=float)
            count = np.zeros([len(self.engLexicon), len(self.forLexicon)], dtype=float)
            totalF = np.zeros([len(self.forLexicon)], dtype=float)
            for forSent, engSent in zip(self.forCorpus, self.engCorpus):
                eSentIdx = self.engCorpus.index(engSent)
                for engWord in engSent.split():
                    eWordIdx = self.engLexicon.index(engWord)
                    for forWord in forSent.split():
                        fWordIdx = self.forLexicon.index(forWord)
                        sTotal[eSentIdx, eWordIdx] += self.transProb1[eWordIdx, fWordIdx]
                for engWord in engSent.split():
                    eWordIdx = self.engLexicon.index(engWord)
                    for forWord in forSent.split():
                        fWordIdx = self.forLexicon.index(forWord)
                        count[eWordIdx, fWordIdx] += self.transProb1[eWordIdx, fWordIdx] / sTotal[eSentIdx, eWordIdx]
                        totalF[fWordIdx] += self.transProb1[eWordIdx, fWordIdx] / sTotal[eSentIdx, eWordIdx]
            for j in range(0, len(self.engLexicon)):
                self.transProb1[j] = count[j] / totalF
                self.transProb1[np.isnan(self.transProb1)] = MIN_PROB
            print('%s th iteration is completed...' % (con + 1))
        return self.transProb1

    def IBMModel2(self, transProb, converge):
        print('IBM Model 2 is being trained...')
        self.transProb2 = transProb
        self.alignProb2 = self.initAlignmentProbabilities()
        for con in range(0, converge):
            sTotal = np.zeros([len(self.engCorpus), len(self.engLexicon)], dtype=float)
            count = np.zeros([len(self.engLexicon), len(self.forLexicon)], dtype=float)
            totalF = np.zeros([len(self.forLexicon)], dtype=float)
            countAlign = np.zeros([self.maxLenForSent, self.maxLenEngSent, self.maxLenEngSent, self.maxLenForSent],
                                  dtype=float)
            totalAlignF = np.zeros([self.maxLenEngSent, self.maxLenEngSent, self.maxLenForSent], dtype=float)
            for forSent, engSent in zip(self.forCorpus, self.engCorpus):
                eSentIdx = self.engCorpus.index(engSent)
                for j, engWord in enumerate(engSent.split()):
                    eWordIdx = self.engLexicon.index(engWord)
                    for i, forWord in enumerate(forSent.split()):
                        fWordIdx = self.forLexicon.index(forWord)
                        sTotal[eSentIdx, eWordIdx] += self.transProb2[eWordIdx, fWordIdx] * self.alignProb2[
                            i, j, len((engSent.split())) - 1, len((forSent.split())) - 1]
                for j, engWord in enumerate(engSent.split()):
                    eWordIdx = self.engLexicon.index(engWord)
                    for i, forWord in enumerate(forSent.split()):
                        fWordIdx = self.forLexicon.index(forWord)
                        c = self.transProb2[eWordIdx, fWordIdx] * self.alignProb2[
                            i, j, len((engSent.split())) - 1, len((forSent.split())) - 1] / sTotal[eSentIdx, eWordIdx]
                        count[eWordIdx, fWordIdx] += c
                        totalF[fWordIdx] += c
                        countAlign[i, j, len((engSent.split())) - 1, len((forSent.split())) - 1] += c
                        totalAlignF[j, len((engSent.split())) - 1, len((forSent.split())) - 1] += c
            self.transProb2 = np.zeros([len(self.engLexicon), len(self.forLexicon)], dtype=float)
            self.alignProb2 = np.zeros([self.maxLenForSent, self.maxLenEngSent, self.maxLenEngSent, self.maxLenForSent],
                                       dtype=float)
            for j in range(0, len(self.engLexicon)):
                self.transProb2[j, :] = count[j, :] / totalF
                self.transProb2[np.isnan(self.transProb2)] = MIN_PROB
            for l in range(0, self.maxLenForSent):
                self.alignProb2[l] = countAlign[l] / totalAlignF
                self.alignProb2[np.isnan(self.alignProb2)] = MIN_PROB
            print('%s th iteration is completed...' % (con + 1))
        return self.transProb2, self.alignProb2

    def IBMModel3(self, transProb, distProb, converge):
        print('IBM Model 3 is being trained...')
        self.transProb3 = transProb
        self.distProb3 = distProb
        self.fertProb3 = self.initFertilityProbabilities()
        self.p1 = 0.5

        for con in range(0, converge):
            count = np.zeros([len(self.engLexicon), len(self.forLexicon)], dtype=float)
            totalF = np.zeros([len(self.forLexicon)], dtype=float)
            countDis = np.zeros([self.maxLenForSent, self.maxLenEngSent, self.maxLenEngSent, self.maxLenForSent],
                                dtype=float)
            totalDisF = np.zeros([self.maxLenEngSent, self.maxLenEngSent, self.maxLenForSent], dtype=float)
            countFer = np.zeros([MAX_FERTILITY, len(self.forLexicon)], dtype=float)
            totalFerF = np.zeros([len(self.forLexicon)], dtype=float)
            countP0 = 0
            countP1 = 0
            for forSent, engSent in zip(self.forCorpus, self.engCorpus):
                A = self.sample(engSent, forSent)
                cTotal = 0
                for a in A:
                    cTotal += self.findTranslationProbabilityModel3(engSent, forSent, a)
                for a in A:
                    c = self.findTranslationProbabilityModel3(engSent, forSent, a) / cTotal
                    null = 0.0
                    for i, engWord in enumerate(engSent.split()):
                        eWordIdx = self.engLexicon.index(engWord)
                        forWordPos = a[i]
                        fWordIdx = self.forLexicon.index(forSent.split()[forWordPos])
                        count[eWordIdx, fWordIdx] += c
                        totalF[fWordIdx] += c
                        countDis[forWordPos, i, len((engSent.split())) - 1, len((forSent.split())) - 1] += c
                        totalDisF[forWordPos, len((engSent.split())) - 1, len((forSent.split())) - 1] += c
                        if a[i] == 0:
                            null += 1.0
                    countP1 += null * c
                    countP0 += (len(engSent.split()) - 2 * null) * c
                    #if math.isnan(countP0):
                    #    pass
                    for i, forWord in enumerate(forSent.split()):
                        fWordIdx = self.forLexicon.index(forWord)
                        fertility = 0
                        for j, engWord in enumerate(engSent.split()):
                            if i == a[j]:
                                fertility += 1
                        countFer[fertility, fWordIdx] += c
                        totalFerF[fWordIdx] += c

            self.transProb3 = np.zeros([len(self.engLexicon), len(self.forLexicon)], dtype=float)
            self.distProb3 = np.zeros([self.maxLenForSent, self.maxLenEngSent, self.maxLenEngSent, self.maxLenForSent],
                                      dtype=float)
            self.fertProb3 = np.zeros([MAX_FERTILITY, len(self.forLexicon)], dtype=float)
            for j in range(0, len(self.engLexicon)):
                self.transProb3[j, :] = count[j, :] / totalF
                self.transProb3[np.isnan(self.transProb3)] = MIN_PROB
            for l in range(0, self.maxLenForSent):
                self.distProb3[l] = countDis[l] / totalDisF
                self.distProb3[np.isnan(self.distProb3)] = MIN_PROB
            for j in range(0, self.maxLenEngSent):
                self.fertProb3[j, :] = countFer[j, :] / totalFerF
                self.fertProb3[np.isnan(self.fertProb3)] = MIN_PROB

            self.p1 = countP1 / (countP1 + countP0)
            p0 = 1 - self.p1

            print('%s th iteration is completed...' % (con + 1))
        return self.transProb3, self.distProb3, self.fertProb3, self.p1,

    def sample(self, engSent, forSent):
        alignment = [0] * len(engSent.split())
        A = set()
        lenOfForSent = len(forSent.split())
        lenOfEngSent = len(engSent.split())
        for j1 in range(0, lenOfEngSent):
            for i in range(0, lenOfForSent):
                if isinstance(alignment, tuple):
                    alignment = list(alignment)
                alignment[j1] = i
                for j2 in range(0, lenOfEngSent):
                    if j1 != j2:
                        bestAlignIdx = self.findBestAlignmentModel2(j2, forSent, engSent)
                        alignment[j2] = bestAlignIdx
                        # cepts[bestAlignIdx].append(j2)
                alignment = self.hillclimb(engSent, forSent, alignment, j1)
                neighbors = self.neighboring(engSent, forSent, alignment, j1)
                A = A.union(neighbors)
        return A

    def hillclimb(self, engSent, forSent, alignment, jpegged):
        while True:
            oldAlignment = alignment
            oldAlignProb = self.findTranslationProbabilityModel3(engSent, forSent, alignment)
            neighboringAlignments = self.neighboring(engSent, forSent, alignment, jpegged)
            for neighbor in neighboringAlignments:
                neighborProb = self.findTranslationProbabilityModel3(engSent, forSent, alignment)
                if neighborProb > oldAlignProb:
                    alignment = neighbor
            if (alignment == oldAlignment):
                break
        return alignment

    def neighboring(self, engSent, forSent, alignment, jpegged):
        neighboringAlignments = set()
        for j in range(0, len(engSent.split())):
            if j != jpegged:
                for i in range(0, len(forSent.split())):
                    if isinstance(alignment, tuple):
                        alignment = list(alignment)
                    tempAlignment = alignment
                    tempAlignment[j] = i
                    neighboringAlignments.add(tuple(tempAlignment))

        for j1 in range(0, len(engSent.split())):
            if j1 != jpegged:
                for j2 in range(1, len(engSent.split())):
                    if j2 != jpegged and j2 != j1:
                        if isinstance(alignment, tuple):
                            alignment = list(alignment)
                        tempAlignment = alignment
                        tempAlignment[j2] = alignment[j1]
                        tempAlignment[j1] = alignment[j2]
                        neighboringAlignments.add(tuple(tempAlignment))

        return neighboringAlignments

    def findBestAlignmentModel2(self, engWordAlign, forSent, engSent):
        forWordsInSent = forSent.split()
        engWordsInSent = engSent.split()
        engWordIdx = self.engLexicon.index(engWordsInSent[engWordAlign])
        bestAlign = -1
        bestAlignIdx = 0
        for i, forWord in enumerate(forSent.split()):
            forWordIdx = self.forLexicon.index(forWord)
            tempAlign = self.transProb3[engWordIdx, forWordIdx] * self.distProb3[
                i, engWordAlign, len(engWordsInSent) - 1, len(forWordsInSent) - 1]
            if tempAlign > bestAlign:
                bestAlign = tempAlign
                bestAlignIdx = i
        return bestAlignIdx

    def findTranslationProbabilityModel1(self, engSent, forSent, alignmentFunction):
        prob = 1
        for i, engWord in enumerate(engSent.split()):
            eWordIdx = self.engLexicon.index(engWord)
            forWordPos = alignmentFunction[i]
            fWordIdx = self.forLexicon.index(forSent.split()[forWordPos])
            prob *= self.transProb1[eWordIdx, fWordIdx]
        return prob

    def findTranslationProbabilityModel2(self, engSent, forSent, alignmentFunction):
        prob = 1
        for i, engWord in enumerate(engSent.split()):
            eWordIdx = self.engLexicon.index(engWord)
            forWordPos = alignmentFunction[i]
            fWordIdx = self.forLexicon.index(forSent.split()[forWordPos])
            prob *= self.transProb2[eWordIdx, fWordIdx] * self.alignProb2[
                forWordPos, i, len((engSent.split())) - 1, len((forSent.split())) - 1]
        return prob

    def getCepts(self, alignmentFunction, forSent):
        cepts = [[] for i in range(len(forSent.split()))]
        for idx, align in enumerate(alignmentFunction):
            cepts[align].append(idx)
        return cepts

    def findTranslationProbabilityModel3(self, engSent, forSent, alignmentFunction):
        prob = 1
        cepts = self.getCepts(alignmentFunction, forSent)
        nullFertility = len(cepts[0])

        p0 = 1 - self.p1
        prob *= (pow(self.p1, nullFertility) * pow(p0, len(engSent.split()) - 2 * nullFertility))

        for i in range(1, nullFertility + 1):
            prob *= (len(engSent.split()) - nullFertility - i + 1) / i


        for i, forWord in enumerate(forSent.split()):
            fWordIdx = self.forLexicon.index(forWord)
            fertility = len(cepts[i])
            prob *= (factorial(fertility) *
                     self.fertProb3[fertility, fWordIdx])

        for i, engWord in enumerate(engSent.split()):
            eWordIdx = self.engLexicon.index(engWord)
            forWordPos = alignmentFunction[i]
            fWordIdx = self.forLexicon.index(forSent.split()[forWordPos])
            prob *= self.transProb3[eWordIdx, fWordIdx] * self.distProb3[
                forWordPos, i, len((engSent.split())) - 1, len((forSent.split())) - 1]

        if math.isnan(prob):
            prob = MIN_PROB
        return prob


def readEngCorpus(fPath):
    print('Reading %s corpus...' % (fPath))
    with open(fPath, encoding='utf-8') as f:
        corpus = f.readlines()
    corpus = [re.sub('[' + string.punctuation + ']', '', x.strip()) for x in corpus]
    return corpus


def readForeignCorpus(fPath):
    print('Reading %s corpus...' % (fPath))
    with open(fPath, encoding='utf-8') as f:
        corpus = f.readlines()
    corpus = ['null ' + re.sub('[' + string.punctuation + ']', '', x.strip()) for x in corpus]
    return corpus


def readReducedParallelCorpus(ePath, fPath):
    print('Reading %s corpus...' % (ePath))
    with open(ePath, encoding='utf-8') as f:
        engCorpus = f.readlines()
    engCorpus = [re.sub('[' + string.punctuation + ']', '', x.strip()) for x in engCorpus]

    print('Reading %s corpus...' % (fPath))
    with open(fPath, encoding='utf-8') as f:
        forCorpus = f.readlines()
    forCorpus = ['null ' + re.sub('[' + string.punctuation + ']', '', x.strip()) for x in forCorpus]

    reducedForCorpus = []
    reducedEngCorpus = []
    for forSent, engSent in zip(forCorpus, engCorpus):
        if len(forSent.split()) <= MAX_SENT_LEN and len(engSent.split()) <= MAX_SENT_LEN:
            reducedForCorpus.append(forSent)
            reducedEngCorpus.append(engSent)
    return reducedEngCorpus[:1000], reducedForCorpus[:1000]


def runModel1And2():
    engTrainCorpus = readEngCorpus('BU_en.txt')
    forTrainCorpus = readForeignCorpus('BU_tr.txt')
    print('Parallel corpus consists of 1000 sentence pairs')

    converge = 15
    ibm = IBMModel(forTrainCorpus[:1000], engTrainCorpus[:1000])
    transProb1 = ibm.IBMModel1(converge)
    ibm.IBMModel2(transProb1, converge)
    return ibm



def runModel3():
    engTrainCorpus, forTrainCorpus = readReducedParallelCorpus('BU_en.txt', 'BU_tr.txt')
    parallelCorpus = dict(zip(forTrainCorpus, engTrainCorpus))
    print('Parallel corpus consists of %s sentence pairs' % (len(parallelCorpus)))
    converge = 3
    ibm = IBMModel(forTrainCorpus, engTrainCorpus)
    transProb1 = ibm.IBMModel1(converge)
    transProb2, alignProb2 = ibm.IBMModel2(transProb1, converge)
    ibm.IBMModel3(transProb2, alignProb2, converge)
    return ibm



ibm1And2 = runModel1And2()
ibm3 = runModel3()

