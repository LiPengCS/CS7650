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

def save_embedding(embed, name):
    with open("models/{}.p".format(name), "wb") as f:
        pickle.dump(embed, f)

def build_common_embeddings():
    # commonly used glove, word2vec and fasttext
    # glove = api.load("glove-wiki-gigaword-300")
    # word2vec = api.load("word2vec-google-news-300")
    fasttext = api.load("fasttext-wiki-news-subwords-300")
    # save_embedding(glove, "glove_common")
    # save_embedding(word2vec, "word2vec_common")
    save_embedding(fasttext, "fasttext_common")

build_common_embeddings()




