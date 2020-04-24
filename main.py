from evaluate import *
from utils import *
import os
import pandas as pd

save_dir = "result"

def evaluate_bias(embed, embed_name):
    # projection
    projection(embed, save_dir, embed_name)

    # # clustering
    clustering(embed, save_dir, embed_name)

    # weat
    weat(embed, save_dir, embed_name)

    # gender classification
    gender_classification(embed, "svc", save_dir, embed_name)

    # # analogy
    analogy(embed, save_dir, embed_name)

def evaluate(embed_name):
    print("evaluate", embed_name)
    embed = load_embedding(embed_name)
    evaluate_bias(embed, embed_name)

evaluate("glove-wiki")
evaluate("word2vec-google")
evaluate("fasttext-crawl")
evaluate("fasttext-wiki-hard-debiased")
evaluate("fasttext-wiki")
evaluate("glove-crawl")
evaluate("glove-wiki-hard-debiased")
evaluate("glove-wiki")
evaluate("word2vec-google-hard-debiased")
evaluate("gn-glove")
evaluate("gp-glove")
evaluate("word2vec-wiki")