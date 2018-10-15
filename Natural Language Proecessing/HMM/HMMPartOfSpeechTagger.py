import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class HMMPOSTagger:
    def __init__(self, data, observations, states):
        self.data = data
        self.observations = observations
        self.states = states
        self.trainData, self.testData = self.splitTrainTestData(self.data)
        self.transitionProbabilities = np.zeros([len(self.states), len(self.states)], dtype=int)
        self.emissionProbabilities = np.zeros([len(self.observations), len(self.states)], dtype=int)
        print('The training corpus consists of %s sentences.' % (len(self.trainData)))
        print('The test corpus consists of %s sentences.' % (len(self.testData)))

    def splitTrainTestData(self, data):
        index = round(len(data) * 0.9)
        return data[:index], data[index:]

    def train(self):
        for sentence in self.trainData:
            previousState = self.states.index('Start')
            for token in sentence:
                obs = self.observations.index(token[0])
                state = self.states.index(token[1])
                self.transitionProbabilities[previousState, state] += 1
                self.emissionProbabilities[obs, state] += 1
                previousState = state
            self.transitionProbabilities[state][self.states.index('End')] += 1

        self.transitionProbabilities = self.transitionProbabilities / self.transitionProbabilities.sum()
        self.emissionProbabilities = self.emissionProbabilities / self.emissionProbabilities.sum()
        print('The corpus is trained.')

    def viterbi(self, obsList):
        startStateIdx = self.states.index('Start')
        endStateIdx = self.states.index('End')
        viterbi = np.zeros([len(self.states), len(obsList)], dtype=float)
        backpointer = np.zeros([len(self.states), len(obsList)], dtype=int)
        for state in self.states:
            if state is not 'Start' and state is not 'End':
                stateIdx = self.states.index(state)
                obsIdx = self.observations.index(obsList[0])
                viterbi[stateIdx, 0] = self.transitionProbabilities[startStateIdx, stateIdx] * \
                                       self.emissionProbabilities[obsIdx, stateIdx]
                backpointer[stateIdx, 0] = 0

        idx = 0
        for idx, obs in enumerate(obsList[1:]):
            for state in self.states:
                if state is not 'Start' and state is not 'End':
                    stateIdx = self.states.index(state)
                    prob, prevStateIdx = self.getMostLikelyPreviousState(viterbi, idx, stateIdx)
                    viterbi[stateIdx, idx + 1] = prob
                    backpointer[stateIdx, idx + 1] = prevStateIdx

        prob, prevStateIdx = self.getMostLikelyEndState(viterbi, idx)
        if idx > 0:
            viterbi[endStateIdx, idx + 1] = prob
            backpointer[endStateIdx, idx + 1] = prevStateIdx
        return self.getBestPath(backpointer, prevStateIdx)

    def getMostLikelyPreviousState(self, viterbi, prevObsIdx, stateIdx):
        bestIdx = 0
        maxProb = - 0.1
        for state in self.states:
            if state is not 'Start' and state is not 'End':
                prevStateIdx = self.states.index(state)
                prob = viterbi[prevStateIdx, prevObsIdx] + self.transitionProbabilities[prevStateIdx, stateIdx] * \
                                                           self.emissionProbabilities[
                                                               prevObsIdx + 1, stateIdx]
                if prob > maxProb:
                    maxProb = prob
                    bestIdx = prevStateIdx
        return maxProb, bestIdx

    def getMostLikelyEndState(self, viterbi, prevObsIdx):
        stateIdx = self.states.index('End')
        bestIdx = 0
        maxProb = - 0.1
        for state in self.states:
            if state is not 'Start' and state is not 'End':
                prevStateIdx = self.states.index(state)
                prob = viterbi[prevStateIdx, prevObsIdx] + self.transitionProbabilities[prevStateIdx, stateIdx]
                if prob > maxProb:
                    maxProb = prob
                    bestIdx = prevStateIdx
        return maxProb, bestIdx

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
    observations = list(set(df['FORM']))
    states = list(set(df['UPOSTAG']))
    states.append('Start')
    states.append('End')
    states.remove(None)
    print('The corpus is parsed.')
    return sentenceList, observations, states


if __name__ == "__main__":
    data, observations, states = parseConllFile('METUSABANCI_treebank.conll')
    hmmTagger = HMMPOSTagger(data, observations, states)
    hmmTagger.train()
    hmmTagger.test()
    hmmTagger.evaluation()
