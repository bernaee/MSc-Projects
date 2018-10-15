import itertools
import numpy as np

MATCH = 1
INDEL = 0
MISMATCH = 0
SYMBOLS = ['A', 'C', 'G', 'T', '_']


def backtrace(v, w, score_matrix, path_matrix):
    alignment_v = ""
    alignment_w = ""
    i = len(v) - 1
    j = len(w) - 1
    alignment_score = score_matrix[i, j]
    while i > 0 and j > 0:
        point = path_matrix[i][j]
        if point == 0:
            alignment_v = v[i] + alignment_v
            alignment_w = w[j] + alignment_w
            i = i - 1
            j = j - 1
        elif point == 1:
            alignment_v = "_" + alignment_v
            alignment_w = w[j] + alignment_w
            j = j - 1
        elif point == 2:
            alignment_v = v[i] + alignment_v
            alignment_w = "_" + alignment_w
            i = i - 1

    while i > 0:
        alignment_v = v[i] + alignment_v
        alignment_w = "_" + alignment_w
        i = i - 1

    while j > 0:
        alignment_v = "_" + alignment_v
        alignment_w = w[j] + alignment_w
        j = j - 1

    return (alignment_v, alignment_w, alignment_score)


def needlemanWunschAlgorithm(v, w):
    v = '_' + v
    w = '_' + w
    score_matrix = np.zeros((len(v), len(w)), dtype=int)
    path_matrix = np.zeros((len(v), len(w)), dtype=int)
    # initialize first row with indel score
    for i in range(1, len(v)):
        score_matrix[i, 0] = score_matrix[i - 1, 0] + INDEL
        path_matrix[i, 0] = 1
    # initialize first column with indel score
    for j in range(1, len(w)):
        score_matrix[0, j] = score_matrix[0, j - 1] + INDEL
        path_matrix[0, j] = 2
    # construct score matrix
    for i in range(1, len(v)):
        for j in range(1, len(w)):
            if v[i] == w[j]:
                match_score = score_matrix[i - 1, j - 1] + MATCH
            else:
                match_score = score_matrix[i - 1, j - 1] + MISMATCH
            insertion_score = score_matrix[i, j - 1] + INDEL
            deletion_score = score_matrix[i - 1, j] + INDEL
            scores = [match_score, insertion_score, deletion_score]
            max_score = max(scores)
            path_matrix[i, j] = scores.index(max_score)
            score_matrix[i, j] = max_score

    return backtrace(v, w, score_matrix, path_matrix)


def profileAlignment(other_sequence, aligned_sequences):
    number_of_aligned_sequences = len(aligned_sequences)
    col_length = len(aligned_sequences[0])
    profile = np.zeros((len(SYMBOLS), col_length), dtype=float)
    for i in range(0, col_length):
        for j in range(0, number_of_aligned_sequences):
            profile[SYMBOLS.index(aligned_sequences[j][i])][i] += 1.0 / number_of_aligned_sequences

    w = '_' + other_sequence
    v = ['_' + aligned_s for aligned_s in aligned_sequences]
    score_matrix = np.zeros((len(v[0]), len(w)), dtype=int)
    path_matrix = np.zeros((len(v[0]), len(w)), dtype=int)
    # initialize first row with indel score
    for i in range(1, len(v[0])):
        score_matrix[i, 0] = score_matrix[i - 1, 0] + INDEL
        path_matrix[i, 0] = 1
    # initialize first column with indel score
    for j in range(1, len(w)):
        score_matrix[0, j] = score_matrix[0, j - 1] + INDEL
        path_matrix[0, j] = 2
    # construct score matrix
    for i in range(1, len(v[0])):
        for j in range(1, len(w)):
            for k in range(0, number_of_aligned_sequences):
                match_prob = 0
                if v[k][i] == w[j]:
                    match_prob += profile[SYMBOLS.index(v[i])][j - 1] * MATCH
                else:
                    match_prob += profile[SYMBOLS.index(v[i])][j - 1] * MISMATCH
            match_score = score_matrix[i - 1, j - 1] + match_prob
            insertion_score = score_matrix[i, j - 1] + INDEL
            deletion_score = score_matrix[i - 1, j] + INDEL
            scores = [match_score, insertion_score, deletion_score]
            max_score = max(scores)
            path_matrix[i, j] = scores.index(max_score)
            score_matrix[i, j] = max_score

    alignment_w_list = []
    for k in range(0, number_of_aligned_sequences):
        alignment_v, alignment_w, alignment_score = backtrace(v, w[k], score_matrix, path_matrix)
        alignment_w_list.append(alignment_w)

    return (alignment_v, alignment_w_list, alignment_score)


def main(fPath):
    with open(fPath, encoding='utf-8') as f:
        sequences = f.readlines()
    sequences = [s.strip() for s in sequences]
    pairwiseSequences = tuple(itertools.combinations(sequences, 2))
    pairwise_alignments = []
    print('The Pairwise Alignments of The Sequences and Their Scores')
    print()
    for pair in pairwiseSequences:
        v = pair[0]
        w = pair[1]
        alignment_v, alignment_w, alignment_score = needlemanWunschAlgorithm(v, w)
        print('%s is aligned to %s with the score %s' % (v, w, alignment_score))
        arrow = ''
        for i, j in zip(alignment_v, alignment_w):
            if i == j:
                arrow += '|'
            else:
                arrow += ' '
        print('     %s' % alignment_v)
        print('     %s' % arrow)
        print('     %s' % alignment_w)
        print()
        pairwise_alignments.append([v, w, alignment_v, alignment_w, alignment_score])

    print('The Alignment of The Profile with The Third Sequence and The Score of The Alignment')
    print()

    multiple_alignments = []
    for alignment in pairwise_alignments:
        v = (set(sequences) - set(alignment[0:2])).pop()
        w = alignment[2:4]
        alignment_v, alignment_w_list, alignment_score = profileAlignment(v, w)
        multiple_alignments.append([alignment_v, alignment_w_list, alignment_score])
        print('Multiple Sequence Alignment with score %s' % (alignment_score))
        print('Profile:')
        print('      %s' % "   ".join(alignment_v))
        profile = ''
        for i in range(0, len(alignment_w_list[0])):
            if alignment_w_list[0][i] == alignment_w_list[1][i]:
                profile +=  ' ' + alignment_w_list[0][i] + '  '
            else:
                profile += alignment_w_list[0][i].lower() + '/' + alignment_w_list[1][i].lower() + ' '
        print('     %s' % profile)
        print('Alignment:')
        print('     %s' % alignment_v)
        for alignment_w in alignment_w_list:
            print('     %s' % alignment_w)
        print()

    print('The Final Multiple Alignment Achieving The Maximum Alignment Score')
    scores = [alignment[4] for alignment in pairwise_alignments]
    max_score = max(scores)
    _, _, best_alignment_v, best_alignment_w, best_alignment_score = pairwise_alignments[scores.index(max_score)]
    print('The best pairwise alignment with score %s' % best_alignment_score)
    print('     %s' % best_alignment_v)
    print('     %s' % best_alignment_w)
    best_alignment_v, best_alignment_w_list, best_alignment_score = multiple_alignments[scores.index(max_score)]
    print('The best alignment with score %s' % best_alignment_score)
    print('     %s' % best_alignment_v)
    for best_alignment_w in best_alignment_w_list:
        print('     %s' % best_alignment_w)

    print()


filePaths = ['test-seq1.txt', 'test-seq2.txt']

print('------------------The Outputs of test-seq1.txt------------------')
main(filePaths[0])
print('------------------The Outputs of test-seq2.txt------------------')
main(filePaths[1])
