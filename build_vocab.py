import pandas as pd
import numpy as np
import json
data = pd.read_csv("data/stereotype_list.csv")
X_1 = data["male"].values.tolist() + data["female"].values.tolist()

data = pd.read_csv("data/GenderWords.csv")
X_2 = data["word"].values.tolist()

with open("data/weat.json") as f:
    data = json.load(f)
X_3 = []
for name, words in data.items():
    X_3.extend(words)

X_4 = []
bias_analogy_f = open("data/Sembias")
for sub_idx, l in enumerate(bias_analogy_f):
    l = l.strip().split()
    for i, word_pair in enumerate(l):
        word_pair = word_pair.split(':')
        X_4.append(word_pair[0])
        X_4.append(word_pair[1])

X = ["he", "she"] + X_1 + X_2 + X_3 + X_4
X = np.array(list(set(X))).reshape(-1, 1)

X = pd.DataFrame(X, columns=["Vocabulary"])
X.to_csv("Vocabulary.csv", index=False)
