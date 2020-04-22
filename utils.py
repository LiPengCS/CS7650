import pickle
import os

def load_embedding(model):
    with open("models/{}.p".format(model), "rb") as f:
        embed = pickle.load(f)
    return embed

def makedir(dir_list, file=None):
    save_dir = os.path.join(*dir_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir