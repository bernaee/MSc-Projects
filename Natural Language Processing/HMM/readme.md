### Hidden Markov Model based POS Tagger for Turkish
- The model is trained on the METU-Sabancı Turkish Dependency Treebank dataset. The corpus is randomly divided into two parts 
(training data and test data). 
- The Viterbi decoding is implemented to find maximum likely POS tag sequences.
- add-k smoothing is applied to tackle with sparse data problem.
- For unknown words, the most likely previous state is calculated without using the word likelihood probabilities P(wi|ti).