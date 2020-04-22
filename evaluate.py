import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import json
from scipy import stats
from scipy.stats import ttest_1samp
import os
from utils import makedir

def get_word_vectors(embed, words, labels=None):
    X_words = []
    X_emb = []
    Y_emb = []

    for i, x in enumerate(words):
        if x in embed:
            X_emb.append(embed[x])
            X_words.append(x)
            if labels is not None:
                Y_emb.append(labels[i])
    X_emb = np.array(X_emb)
    Y_emb = np.array(Y_emb).ravel()

    if labels is None:
        return X_words, X_emb
    else:
        return X_words, X_emb, Y_emb


def clustering(embed, save_dir, embed_name):
    data = pd.read_csv("data/stereotype_list.csv")
    X = data["male"].values.tolist() + data["female"].values.tolist()
    Y = [1] * len(data) + [0] * len(data)

    X_words, X_emb, Y_emb = get_word_vectors(embed, X, Y)

    X_embedded = TSNE(n_components=2, random_state=1).fit_transform(X_emb)
    X_male = X_embedded[Y_emb == 1]
    X_female = X_embedded[Y_emb == 0]
    plt.figure()
    plt.scatter(X_male[:, 0], X_male[:, 1], label="male-biased", color='b', s=10, alpha=.55)
    plt.scatter(X_female[:, 0], X_female[:, 1], label="female-biased", color='C1', s=10, alpha=.55)
    plt.legend()
    plt.savefig(makedir([save_dir, "clustering"], "{}_plot.png".format(embed_name)))

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_emb)
    score = (kmeans.labels_ == Y_emb).mean()
    score = max(score, 1- score)

    score = pd.DataFrame([[score]], columns=["score"])
    score.to_csv(makedir([save_dir, "clustering"], "{}_acc.csv".format(embed_name)), index=False)
    return score

def gender_classification(embed, ml_model, save_dir, embed_name):
    data = pd.read_csv("data/GenderWords.csv")
    X = data["word"].values
    Y = data["label"].values

    X_words, X_emb, Y_emb = get_word_vectors(embed, X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X_emb, Y_emb, test_size=0.2, random_state=1)

    param_dict = {
        "svc": {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]},
        "rf": {"max_depth": [1, 25, 50, 75, 100]},
        "boost": {"learning_rate": [1e-7, 1e-5, 1e-3, 1, 10]}
    }
    model_dict = {
        "svc": SVC(),
        "rf": RandomForestClassifier(n_estimators=100),
        "boost": AdaBoostClassifier(n_estimators=100)
    }

    model = model_dict[ml_model]
    parameters = param_dict[ml_model]
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    score = pd.DataFrame([[score]], columns=["score"])
    score.to_csv(makedir([save_dir, "gender_classification"], "{}_acc.csv".format(embed_name)), index=False)
    return score

def cosine(x, y):
    sim = x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return sim

def projection(embed, save_dir, embed_name):
    data = pd.read_csv("data/stereotype_list.csv")
    X = data["male"].values.tolist() + data["female"].values.tolist()
    X_words, X_emb = get_word_vectors(embed, X)

    gender_direction = embed["he"] - embed["she"]
    gender_direction = gender_direction

    project = []
    for x in X_emb:
        sim = cosine(x, gender_direction)
        project.append(sim)
    project = np.array(project)
    avg_project = np.abs(project).mean()

    orders = np.argsort(project)

    plt.figure()
    plt.scatter(project, range(len(project)), s=10)
    plt.yticks([])
    plt.xlim([-0.5, 0.5])
    plt.xlabel("Similarity")
   

    for i in range(5):
        plt.text(project[orders[i]], orders[i]+0.2, X_words[orders[i]])
        plt.text(project[orders[-(i+1)]], orders[-(i+1)]+0.2, X_words[orders[-(i+1)]])
    plt.savefig(makedir([save_dir, "projection"], "{}_plot.png".format(embed_name)))

    score = pd.DataFrame([[avg_project]], columns=["score"])
    score.to_csv(makedir([save_dir, "projection"], "{}_acc.csv".format(embed_name)), index=False)
    return avg_project

def weat(embed, save_dir, embed_name):
    def association(w, M, F):
        s = 0
        for m in M:
            s += cosine(w, m) / len(M)
        for f in F:
            s -= cosine(w, f) / len(F)
        return s

    def S(X, Y, M, F):
        s = 0
        for x in X:
            s+= association(x, M, F)
        for y in Y:
            s-= association(y, M, F)
        return s

    def test(X, Y, M, F):
        s0 = S(X, Y, M, F)
        np.random.seed(1)
        U = np.vstack([X, Y])
        s_hat = []
        for i in range(50):
            idx = np.random.permutation(len(U))
            X_hat = U[idx[:len(X)]]
            Y_hat = U[idx[len(X):]]
            si = S(X_hat, Y_hat, M, F)
            s_hat.append(si)
        t, pvalue = ttest_1samp(s_hat, s0)
        pvalue = pvalue / 2
        return pvalue

    with open("data/weat.json") as f:
        data = json.load(f)

    vectors = {}
    for name, words in data.items():
        _, vectors[name] = get_word_vectors(embed, words)

    M = vectors["M"]
    F = vectors["F"]

    pvalues = []
    for b in ["B1", "B2", "B3", "B4", "B5"]:
        X = vectors[b + "_X"]
        Y = vectors[b + "_Y"]
        pvalues.append(test(X, Y, M, F))

    score = pd.DataFrame([pvalues], columns=["B1", "B2", "B3", "B4", "B5"])
    score.to_csv(makedir([save_dir, "weat"], "{}_score.csv".format(embed_name)), index=False)
    return pvalues

def analogy(embed, save_dir, embed_name):
    bias_analogy_f = open("data/Sembias")

    definition_num = 0
    none_num = 0
    stereotype_num = 0
    total_num = 0
    sub_definition_num = 0
    sub_none_num = 0
    sub_stereotype_num = 0
    sub_size = 40

    sub_start = -(sub_size - sum(1 for line in open("data/Sembias")))

    gender_v = embed['he'] - embed['she']
    for sub_idx, l in enumerate(bias_analogy_f):
        l = l.strip().split()
        max_score = -100
        for i, word_pair in enumerate(l):
            word_pair = word_pair.split(':')
            pre_v = embed[word_pair[0]] - embed[word_pair[1]]
            score = cosine(gender_v, pre_v)
            if score > max_score:
                max_idx = i
                max_score = score
        if max_idx == 0:
            definition_num += 1
            if sub_idx >= sub_start:
                sub_definition_num += 1
        elif max_idx == 1 or max_idx == 2:
            none_num += 1
            if sub_idx >= sub_start:
                sub_none_num += 1
        elif max_idx == 3:
            stereotype_num += 1
            if sub_idx >= sub_start:
                sub_stereotype_num += 1
        total_num += 1

    definition_acc = definition_num / total_num
    stereotype_acc = stereotype_num / total_num
    none_acc = none_num / total_num

    score = pd.DataFrame([[definition_acc, stereotype_acc, none_acc]], columns=["definition_acc", "stereotype_acc", "none_acc"])
    score.to_csv(makedir([save_dir, "analogy"], "{}_score.csv".format(embed_name)), index=False)

    return definition_acc, stereotype_acc, none_acc











