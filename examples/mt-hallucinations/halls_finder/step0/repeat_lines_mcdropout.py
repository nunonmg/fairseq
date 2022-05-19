import pandas as pd
import pickle 
import numpy as np
import argparse
import os
import torch
from collections import Counter
from tqdm import tqdm

def create_score_dropout_files():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, required=True)
    parser.add_argument("--savepath", type=str, required=True)
    parser.add_argument("--lp", type=str, required=True)
    parser.add_argument("--N_dropout", type=int, required=False, default=10)
    args, _ = parser.parse_known_args()

    pkl_path = args.df_path
    savepath = args.savepath
    n_dropout = args.N_dropout
    lp = args.lp

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    df = pd.read_pickle(pkl_path)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("Preparing files.")
    for idx in tqdm(range(len(df))):
        src = df.src.values[idx]
        mt = df.mt.values[idx]
        i = 0
        while i < n_dropout:
            with open(savepath + "/score-dropout." + lp[:2], "a") as f:
                f.write(src + " \n")
            with open(savepath + "/score-dropout." + lp[-2:], "a") as f:
                f.write(mt + " \n")
            i += 1

if __name__ == "__main__":
    create_score_dropout_files()