from evaluate import *
from utils import *
import os
import pandas as pd

save_dir = "result"

def evaluate_bias(embed, embed_name):
    # projection
    projection(embed, save_dir, embed_name)

    # clustering
    clustering(embed, save_dir, embed_name)

    # weat
    weat(embed, save_dir, embed_name)

    # gender classification
    gender_classification(embed, "svc", save_dir, embed_name)

    # analogy
    analogy(embed, save_dir, embed_name)

# evaluate commonly used embeddings
glove_common = load_embedding("glove_common")
word2vec_common = load_embedding("word2vec_common")
fasttext_common = load_embedding("fasttext_common")

evaluate_bias(glove_common, "glove_common")
evaluate_bias(word2vec_common, "word2vec_common")
evaluate_bias(fasttext_common, "fasttext_common")
