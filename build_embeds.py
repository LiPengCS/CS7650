# tfidf
# ppmi
# word2vec - google news 300
# fasttext - wikipedia 2017: https://fasttext.cc/docs/en/english-vectors.html
# glove - wikipedia gigaword 300
# Elmo - 
# BERT

import pickle
import gensim.downloader as api
import io
import pandas as pd

def save_embedding(embed, name):
    vocab = pd.read_csv("data/Vocabulary.csv")

    compressed_emb = {}
    for word in vocab["Vocabulary"]:
        if word in embed:
            compressed_emb[word] = embed[word]

    with open("models/{}.p".format(name), "wb") as f:
        pickle.dump(compressed_emb, f)

def build_common_embeddings():
    # commonly used glove, word2vec and fasttext
    glove = api.load("glove-wiki-gigaword-300")
    save_embedding(glove, "glove_common")
    word2vec = api.load("word2vec-google-news-300")
    save_embedding(word2vec, "word2vec_common")
    fasttext = api.load("fasttext-wiki-news-subwords-300")
    save_embedding(fasttext, "fasttext_common")

build_common_embeddings()




