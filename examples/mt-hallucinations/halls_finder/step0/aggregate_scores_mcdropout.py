import pickle 
import argparse
import itertools
import os

import pandas as pd
import numpy as np
from tqdm import tqdm


def add_score_dropout_uncertainty_measures():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, required=True)
    parser.add_argument("--savepath", type=str, required=True)
    parser.add_argument("--N_dropout", type=int, required=False, default=16)
    args, _ = parser.parse_known_args()

    pkl_path = args.df_path
    savepath = args.savepath
    n_dropout = args.N_dropout

    df_score_dropout = pd.read_pickle(pkl_path)
   

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("Aggregating Ensemble Probs")
    score_mts = df_score_dropout.score.values
    score_agg = [score_mts[i:i + n_dropout] for i in range(0, len(score_mts), n_dropout)]
    mean_agg = [round(np.mean(sublist),4) for sublist in score_agg]
    min_agg = [round(np.min(sublist),4) for sublist in score_agg]
    var_agg = [round(np.var(sublist),4) for sublist in score_agg]

    df_score_dropout["mean_agg"] = mean_agg
    df_score_dropout["min_agg"] = min_agg
    df_score_dropout["var_agg"] = var_agg

    df_to_save = df_score_dropout[["idx", "mean_agg", "min_agg", "var_agg"]]
    df_to_save.to_csv(savepath + "/agg_probs.pkl")

if __name__ == "__main__":
    add_score_dropout_uncertainty_measures()

    