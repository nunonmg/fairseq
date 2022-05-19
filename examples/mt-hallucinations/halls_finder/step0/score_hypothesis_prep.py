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
    parser.add_argument("--lp", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bicleaner", type=bool, required=False)
    parser.add_argument("--dist", type=str, required=False, default="lowcomet")
    parser.add_argument("--N_dropout", type=int, required=False, default=16)
    args, _ = parser.parse_known_args()
    lp = args.lp
    ckpt = args.ckpt
    n_dropout = args.N_dropout
    dist = args.dist
    bicleaner = args.bicleaner

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if bicleaner:
        bicleaner_str = "_w_bicleaner"
    else:
        bicleaner_str = ""

    if lp == "de-en":
        datapath = "/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_de-en_heldout/"
        ckpt_path = "/home/nunomg/mt-hallucinations/HALO/fairseq/checkpoints/wmt18_de-en/"
    elif lp == "en-ru":
        datapath = "/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_en-ru_heldout/"
        ckpt_path = "/home/nunomg/mt-hallucinations/HALO/fairseq/checkpoints/wmt18_en-ru/"

    if dist == "lowcomet":
        df_path = datapath + ckpt + "/dataframes/heldout_lowcomet" + bicleaner_str + ".pkl"
        savepath = "/home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/score-dropout/lowcomet/" + lp + "/" + ckpt
    else:
        df_path = datapath + ckpt + "/dataframes/datasampleswstats" + bicleaner_str + ".pkl"
        savepath = "/home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/score-dropout/datasamples/" + lp + "/" + ckpt

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    df = pd.read_pickle(df_path)

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