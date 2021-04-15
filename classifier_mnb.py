from typing import List, Any
from random import random
from math import log
from collections import defaultdict

cnt_pos_docs = 0
cnt_neg_docs = 0

def count_labels(labels: List):
    return {
        unique_label: sum(1 for label in labels if label == unique_label)
        for unique_label in set(labels)
    }


def preprocessing(texts: List[str]):
    for i in range(len(texts)):
        texts[i] = texts[i].lower()

    
    for j in range(len(texts)):
        for i in range(len(texts[j]) - 1):
            if (not (texts[j][i].isdigit() or texts[j][i].isalpha() or texts[j][i].isspace() or texts[j][i] == "'")):
                if (not texts[j][i - 1].isspace() and not texts[j][i + 1].isspace()):
                    texts[j] = texts[j][:i] + ' ' + texts[j][i] + ' ' + texts[j][i + 1:]
                elif (not texts[j][i - 1].isspace()):
                    texts[j] = texts[j][:i] + ' ' + texts[j][i:]
                elif (not texts[j][i + 1].isspace()):
                    texts[j] = texts[j][:i + 1] + ' ' + texts[j][i + 1:]

        if (not (texts[j][len(texts[j]) - 1].isdigit() or texts[j][len(texts[j]) - 1].isalpha() or texts[j][len(texts[j]) - 1].isspace())):
            if (not texts[j][len(texts[j]) - 2].isspace()):
                texts[j] = texts[j][:len(texts[j]) - 1] + ' ' + texts[j][len(texts[j]) - 1]
    return texts

def text_to_tokens(texts: List[str]):
    tokenized_texts: List[dict] = []
    for text in texts:
        tokens_freq = defaultdict(int)
        tokens = text.split()
        length = len(tokens)
        for i in range(length):
            #tokens_freq[tokens[i]] += 1
            if (i + 1 < length):
                bigram_token = tokens[i] + ' ' + tokens[i + 1]
                tokens_freq[bigram_token] += 1
            if (i + 2 < length):
                threegram_token = tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2]
                tokens_freq[threegram_token] += 1
            #if (i + 3 < length):
            #    fourgram_token = tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2] + ' ' + tokens[i + 3]
            #    tokens_freq[fourgram_token] += 1
        tokenized_texts.append(tokens_freq)
    
    return tokenized_texts

def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param pretrain_params: parameters that were learned at the pretrain step
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """
    train_texts = preprocessing(train_texts)
    train_tokenized_texts = text_to_tokens(train_texts)

    train_pos = [train_tokenized_texts[i] for i in range(len(train_labels)) if train_labels[i] == 'pos']
    train_neg = [train_tokenized_texts[i] for i in range(len(train_labels)) if train_labels[i] == 'neg']
    
    cnt_pos_docs = len(train_pos)
    cnt_neg_docs = len(train_neg)


    all_words_freq = defaultdict(int)
    all_words = set()

    pos_dict = defaultdict(int)
    neg_dict = defaultdict(int)
    sum_len_pos = 0
    sum_len_neg = 0

    for text in train_pos:
        for token in text:
            all_words.add(token)
            all_words_freq[token] += text[token]
            pos_dict[token] += text[token]
            sum_len_pos += text[token]
    
    for text in train_neg:
        for token in text:
            all_words.add(token)
            all_words_freq[token] += text[token]
            neg_dict[token] += text[token]
            sum_len_neg += text[token]
    
    alpha = 1 #For additive smoothing
    M = len(all_words)
    sum_len = 0
    print("____________")
    print("Sum of text lens", sum_len)
    print("____________")
    print("Words quantity", M)
    print("____________")

    token_probs_pos = defaultdict(int)
    token_probs_neg = defaultdict(int)
    print("Calculate probablity for", M, "tokens")

    i = 0
    for token in all_words:
        if (i % 5000 == 0):
            print("__________")
            print("Calculated", i, "tokens")
            print("__________")
        token_probs_pos[token] = (alpha + pos_dict[token]) / (alpha * M + sum_len_pos)
        token_probs_neg[token] = (alpha + neg_dict[token]) / (alpha * M + sum_len_neg)
        i += 1
    
    return {
        "token_probs_pos": token_probs_pos,
        "token_probs_neg": token_probs_neg,
        "all_words": all_words,
        "sum_len_pos": sum_len_pos,
        "sum_len_neg": sum_len_neg,
        "cnt_pos_docs": cnt_pos_docs,
        "cnt_neg_docs": cnt_pos_docs,
        "pos_dict": pos_dict,
        "neg_dict": neg_dict
    }



def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    
    return None


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    alpha = 1
    token_probs_pos = params["token_probs_pos"]
    token_probs_neg = params["token_probs_neg"]
    all_words = params["all_words"]
    M = len(all_words)
    cnt_pos_docs = params["cnt_pos_docs"]
    cnt_neg_docs = params["cnt_neg_docs"]

    sum_len_neg = params["sum_len_neg"]
    sum_len_pos = params["sum_len_pos"]
    pos_dict = params["pos_dict"]
    neg_dict = params["neg_dict"]


    test_texts = preprocessing(texts)
    test_tokenized_texts = text_to_tokens(test_texts)
    
    res = []
    log_pos_probablity = 0
    log_neg_probablity = 0
    i = 0
    for text in test_tokenized_texts:
        if (i % 5000 == 0):
            print("Classified", i, "texts")
        i += 1
        log_pos_probablity = log(cnt_pos_docs)
        log_neg_probablity = log(cnt_neg_docs)
        for token in text:
            if (token_probs_pos[token] == 0):
                token_probs_pos[token] = alpha / (alpha * M + sum_len_pos)
            else:
                log_pos_probablity += log(token_probs_pos[token])
            if (token_probs_neg[token] == 0):
                token_probs_neg[token] = alpha / (alpha * M + sum_len_neg)
            else:
                log_neg_probablity += log(token_probs_neg[token])
        if (log_neg_probablity > log_pos_probablity):
            res.append("neg")
            #for token in text:
            #    all_words.add(token)
            #    M = len(all_words)
            #    neg_dict[token] += text[token]
            #    sum_len_neg += text[token]
            #    token_probs_neg[token] = (alpha + neg_dict[token]) / (alpha * M + sum_len_neg)

        else:
            res.append("pos")
            #for token in text:
            #    all_words.add(token)
            #    M = len(all_words)
            #    pos_dict[token] += text[token]
            #    sum_len_pos += text[token]
            #    token_probs_pos[token] = (alpha + pos_dict[token]) / (alpha * M + sum_len_pos)


    
    print('Predicted labels counts:')
    print(count_labels(res))
    return res
    
