import pandas as pd
import numpy as np
import random

np.seterr(divide='ignore', invalid='ignore')

SMOOTHING_FACTOR = 0.01000


class HMMPOSTagger:
    def __init__(self, data, posTags, train_test_split=0.9):
        self.data = data
        self.trainData, self.testData = self.splitTrainTestData(self.data, train_test_split)
        self.observations = list(set([word[0] for sent in self.trainData for word in sent]))
        self.states = posTags
        self.states.append('Start')
        self.states.append('End')

        self.transitionProbabilities = np.zeros([len(self.states), len(self.states)], dtype=int)
        self.emissionProbabilities = np.zeros([len(self.observations), len(self.states)], dtype=int)

        print('The training corpus consists of %s sentences.' % (len(self.trainData)))
        print('The test corpus consists of %s sentences.' % (len(self.testData)))

    def splitTrainTestData(self, data, train_test_split):
        random.shuffle(data)
        index = round(len(data) * train_test_split)
        return data[:index], data[index:]

    def train(self):
        for sentence in self.trainData:
            previousState = self.states.index('Start')
            state = previousState
            for token in sentence:
                obs = self.observations.index(token[0])
                state = self.states.index(token[1])
                self.transitionProbabilities[previousState, state] += 1
                self.emissionProbabilities[obs, state] += 1
                previousState = state
            self.transitionProbabilities[state][self.states.index('End')] += 1

        self.transitionProbabilities = self.transitionProbabilities + SMOOTHING_FACTOR
        self.emissionProbabilities = self.emissionProbabilities + SMOOTHING_FACTOR
        self.transitionProbabilities = self.transitionProbabilities / self.transitionProbabilities.sum()
        self.emissionProbabilities = self.emissionProbabilities / self.emissionProbabilities.sum()
        print('The corpus is trained.')

    def getObservationIdx(self, obs):
        if obs in self.observations:
            obsIdx = self.observations.index(obs)
        else:
            obsIdx = None  # self.observations.index('<UNK>')
        return obsIdx

    def getMostLikelyPreviousState(self, viterbi, prevTs, currObs, currState):
        probabilities = []
        for state in self.states:
            if state is not 'Start' and state is not 'End':
                prevState = self.states.index(state)
                if bool(currObs):
                    prob = viterbi[prevState, prevTs] * self.transitionProbabilities[prevState, currState] * \
                           self.emissionProbabilities[
                               currObs, currState]
                else:
                    prob = viterbi[prevState, prevTs] * self.transitionProbabilities[prevState, currState]
                probabilities.append(prob)
        maxProb = np.max(probabilities)
        bestState = np.argmax(probabilities)
        return maxProb, bestState

    def viterbi(self, sentence):
        startState = self.states.index('Start')
        endState = self.states.index('End')

        viterbi = np.zeros([len(self.states), len(sentence)], dtype=float)
        backpointer = np.zeros([len(self.states), len(sentence)], dtype=int)

        prevTs = 0
        for ts, obs in enumerate(sentence):
            currObs = self.getObservationIdx(obs)
            for state in self.states:
                if state is not 'Start' and state is not 'End':
                    currState = self.states.index(state)
                    if ts == 0:
                        if bool(currObs):
                            viterbi[currState, ts] = self.transitionProbabilities[startState, currState] * \
                                                     self.emissionProbabilities[currObs, currState]
                        else:
                            viterbi[currState, ts] = self.transitionProbabilities[startState, currState]
                        backpointer[currState, ts] = startState
                    else:
                        prob, prevState = self.getMostLikelyPreviousState(viterbi, prevTs, currObs, currState)
                        viterbi[currState, ts] = prob
                        backpointer[currState, ts] = prevState
            prevTs = ts

        probEnd, prevStateEnd = self.getMostLikelyPreviousState(viterbi, prevTs, None, endState)
        bestPath = self.getBestPath(backpointer, prevStateEnd)
        return bestPath

    def getBestPath(self, backpointer, prevStateIdx):
        path = []
        path.append(prevStateIdx)
        obsIdx = backpointer.shape[1] - 1
        while obsIdx > 0:
            prevStateIdx = backpointer[prevStateIdx, obsIdx]
            path.append(prevStateIdx)
            obsIdx -= 1
        path.reverse()
        bestPOSTags = []
        for p in path:
            bestPOSTags.append(self.states[p])
        return bestPOSTags

    def test(self):
        results = []
        for sentence in self.testData:
            obsList = []
            stateList = []
            for obs, state in sentence:
                obsList.append(obs)
                stateList.append(state)
            bestPOSTags = self.viterbi(obsList)
            results.append((stateList, bestPOSTags))
        self.results = results

    def evaluation(self):
        correctSentences = 0
        allSentences = len(self.results)
        correctWords = 0
        allWords = 0
        confusionMatrix = np.zeros([len(self.states), len(self.states)], dtype=int)
        for res in self.results:
            if res[0] == res[1]:
                correctSentences += 1
            for tag, pred in zip(res[0], res[1]):
                if tag == pred:
                    correctWords += 1
                confusionMatrix[self.states.index(pred), self.states.index(tag)] += 1
            allWords += len(res[0])
        print('Word-based success rate: %s%%' % (round((correctWords / allWords) * 100, 2)))
        print('Sentence-based success rate: %s%%' % (round((correctSentences / allSentences) * 100, 2)))
        print('Confusion matrix:')
        confusionMatrix = confusionMatrix / confusionMatrix.sum(axis=0)[None, :]
        confusionMatrix = confusionMatrix.round(2)
        confusionMatrix = np.nan_to_num(confusionMatrix, nan=0)
        self.states.remove('Start')
        self.states.remove('End')
        row_format = "{:>10}" * (len(self.states) + 1)
        print(row_format.format("", *self.states))
        for team, row in zip(self.states, confusionMatrix):
            print(row_format.format(team, *row))


def parseConllFile(fPath):
    with open(fPath, encoding='utf-8') as f:
        corpus = f.readlines()
    corpus = [x.split() for x in corpus]
    df = pd.DataFrame(corpus,
                      columns=['ID', 'FORM', 'LEMMA', 'UPOSTAG', 'XPOSTAG', 'FEATS', 'HEAD', 'DEPREL', 'DEPS',
                               'MISC'])
    df = df.loc[df['FORM'] != '_']
    df.loc[df['UPOSTAG'] == 'satın', 'UPOSTAG'] = 'Noun'
    df.loc[df['UPOSTAG'] == 'Zero', 'UPOSTAG'] = 'Verb'

    sentence = []
    sentenceList = []
    for idx, row in df.iterrows():
        if row['UPOSTAG'] == None:
            sentenceList.append(sentence)
            sentence = []
        else:
            sentence.append((row['FORM'], row['UPOSTAG']))
    posTags = list(set(df['UPOSTAG']))
    posTags.remove(None)
    print('The corpus is parsed.')
    return sentenceList, posTags


if __name__ == "__main__":
    data, posTags = parseConllFile('METUSABANCI_treebank.conll')
    hmmTagger = HMMPOSTagger(data, posTags)
    hmmTagger.train()
    hmmTagger.test()
    hmmTagger.evaluation()
