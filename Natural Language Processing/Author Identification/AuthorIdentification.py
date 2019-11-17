import os
import re
import random
import codecs
import pandas as pd

SMOOTHING_FACTOR = 3

random.seed(2)


def key_with_max_val(dic):
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]


def split_train_test_corpus(root_path, train_split=0.6):
    authors = os.listdir(root_path)
    authors = sorted(authors)
    train_articles_path = dict()
    test_articles_path = dict()
    for author in authors:
        articles = os.listdir(os.path.join(root_path, author))
        random.shuffle(articles)
        l_articles = len(articles)
        split_idx = int(l_articles * train_split)
        train_articles = articles[:split_idx]
        test_articles = articles[split_idx:]
        train_articles_path[author] = train_articles
        test_articles_path[author] = test_articles
    return train_articles_path, test_articles_path, authors


def read_corpus(root_path, articles_path):
    corpus = dict()
    for author, articles in articles_path.items():
        corpus[author] = []
        for article in articles:
            f_path = os.path.join(root_path, author, article)
            with codecs.open(os.path.join(f_path), 'r', encoding='iso-8859-9', errors='ignore') as filedata:
                text = filedata.read()
                corpus[author].append(text)
    return corpus


def tokenize(text):
    tokens = re.split('\r|\n|\s|\t|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|\[|\]|\^|_|`|{|\||}|~', text)
    return tokens


def get_train_summary(train_corpus):
    train_summary = dict()
    for author, articles in train_corpus.items():
        author_tokens = []

        author_articles = dict()
        for article_idx, article in enumerate(articles):
            tokens = tokenize(article)
            tokens = [t for t in tokens if t != '']
            author_articles[article_idx] = tokens
            author_tokens.extend(tokens)

        author_tokens = [t for t in author_tokens if t != '']
        author_vocab = set(author_tokens)
        n_author_tokens = len(author_tokens)
        l_author_vocab = len(author_vocab)

        author_word_freq = dict()
        author_word_prob = dict()
        for word in author_vocab:
            author_word_freq[word] = 0
        for word in author_tokens:
            author_word_freq[word] += 1
        for word in author_vocab:
            author_word_prob[word] = (author_word_freq[word] + SMOOTHING_FACTOR) / (
                    n_author_tokens + SMOOTHING_FACTOR * l_author_vocab)

        train_summary[author] = dict()
        train_summary[author]['articles'] = author_articles
        train_summary[author]['tokens'] = author_tokens
        train_summary[author]['n_tokens'] = n_author_tokens
        train_summary[author]['vocab'] = author_vocab
        train_summary[author]['l_vocab'] = l_author_vocab
        train_summary[author]['freq'] = author_word_freq
        train_summary[author]['prob'] = author_word_prob
    return train_summary


def naive_bayes_predict(train_corpus_summary, article):
    tokens = tokenize(article)
    tokens = [t for t in tokens if t != '']
    probabilities = dict()
    for author, summary in train_corpus_summary.items():
        prob = 0.0
        for token in tokens:
            if token in list(summary['vocab']):
                prob = prob + summary['prob'][token]
        probabilities[author] = prob
    pred_author = key_with_max_val(probabilities)
    return pred_author


def naive_bayes_test(train_corpus_summary, test_corpus):
    results = dict()
    for author, articles in test_corpus.items():
        results[author] = dict()
        for article in articles:
            pred_author = naive_bayes_predict(train_corpus_summary, article)
            if pred_author in list(results[author].keys()):
                results[author][pred_author] += 1
            else:
                results[author][pred_author] = 1
    results = pd.DataFrame(results).T
    results.fillna(0, inplace=True)
    return results


def write_svm_file(authors, document_vocab, corpus_summary, file_path):
    lines = ''
    for author_idx, author in enumerate(authors):
        summary = corpus_summary[author]
        for article, tokens in summary['articles'].items():
            line = str(author_idx + 1) + ' '
            vocab = sorted(list(set(tokens)))
            for word in vocab:
                if word in document_vocab:
                    word_idx = document_vocab.index(word)
                    line += str(word_idx + 1)
                    line += ':1 '
            lines += line
            lines += '\n'
    f = open(file_path, 'w')
    f.write(lines)
    f.close()


def create_svm_file(authors, train_corpus_summary, test_corpus_summary, train_file, test_file):
    authors = sorted(authors)
    document_vocab = []
    for author, summary in train_corpus_summary.items():
        document_vocab.extend(list(summary['vocab']))
    document_vocab = list(set(document_vocab))
    document_vocab = sorted(document_vocab)
    write_svm_file(authors, document_vocab, train_corpus_summary, train_file)
    write_svm_file(authors, document_vocab, test_corpus_summary, test_file)
    return authors, document_vocab


def read_svm_file(f_path):
    with codecs.open(f_path, 'r', ) as filedata:
        output_svm = filedata.read()
    lines = output_svm.split('\n')[:-1]
    results = [line.split(' ') for line in lines]
    df = pd.DataFrame(results)
    outputs = df[0].to_list()
    return outputs


def parse_svm_results(authors, test_file, output_file):
    labels = read_svm_file(test_file)
    preds = read_svm_file(output_file)
    results = dict()
    for label, pred in zip(labels, preds):
        author = authors[int(label) - 1]
        pred_author = authors[int(pred) - 1]
        if not author in list(results.keys()):
            results[author] = dict()
        if pred_author in list(results[author].keys()):
            results[author][pred_author] += 1
        else:
            results[author][pred_author] = 1
    results = pd.DataFrame(results).T
    results.fillna(0, inplace=True)
    return results


def calculate_results(tp, fp, fn):
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = (tp / (tp + fp)) * 100
    recall = (tp / (tp + fn)) * 100
    f1_measure = 2 * prec * recall / (prec + recall)
    return prec, recall, f1_measure


def evaluate(results):
    confusion_matrix = []
    for idx, row in results.iterrows():
        n_author_articles = row.sum()
        if idx in list(results.columns):
            tp = row[idx]
            fn = n_author_articles - tp
            n_author_preds = results[idx].sum()
            fp = n_author_preds - tp
        else:
            tp = 0
            fn = n_author_articles - tp
            fp = 0
        prec, recall, f1_measure = calculate_results(tp, fp, fn)
        confusion_matrix.append((idx, tp, fp, fn, prec, recall, f1_measure))
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=['Author', 'TP', 'FP', 'FN', 'Prec', 'Recall', 'F1'])
    confusion_matrix.fillna(0, inplace=True)

    author_articles = results.sum(axis=1)
    n_articles = author_articles.sum()
    micro_averages = confusion_matrix.sum()
    micro_tp = micro_averages['TP']
    micro_fp = micro_averages['FP']
    micro_fn = micro_averages['FN']
    micro_prec, micro_recall, micro_f1_measure = calculate_results(micro_tp, micro_fp, micro_fn)
    macro_averages = confusion_matrix.mean()
    macro_prec, macro_recall, macro_f1_measure = macro_averages['Prec'], macro_averages['Recall'], macro_averages['F1']
    accuracy = (micro_tp / n_articles) * 100
    print('Micro Averages')
    print('P: %s R: %s F1: %s' % (round(micro_prec, 2), round(micro_recall, 2), round(micro_f1_measure, 2)))
    print('Macro Averages')
    print('P: %s R: %s F1: %s' % (round(macro_prec, 2), round(macro_recall, 2), round(macro_f1_measure, 2)))
    print('Accuracy: %s' % round(accuracy, 2))


def main_naive_bayes():
    print('Preprocessing data for Naive Bayes...')
    root_path = 'authors'
    train_articles_path, test_articles_path, authors = split_train_test_corpus(root_path)
    train_corpus = read_corpus(root_path, train_articles_path)
    test_corpus = read_corpus(root_path, test_articles_path)
    train_corpus_summary = get_train_summary(train_corpus)
    naive_bayes_results = naive_bayes_test(train_corpus_summary, test_corpus)
    evaluate(naive_bayes_results)


def main_svm():
    print('Preprocessing data for SVM...')
    root_path = 'authors'
    train_file = 'train_svm'
    test_file = 'test_svm'
    output_file = 'output_file'
    train_articles_path, test_articles_path, authors = split_train_test_corpus(root_path)
    train_corpus = read_corpus(root_path, train_articles_path)
    test_corpus = read_corpus(root_path, test_articles_path)
    train_corpus_summary = get_train_summary(train_corpus)
    test_corpus_summary = get_train_summary(test_corpus)
    create_svm_file(authors, train_corpus_summary, test_corpus_summary, train_file, test_file)
    results = parse_svm_results(authors, test_file, output_file)
    evaluate(results)

main_naive_bayes()
main_svm()
