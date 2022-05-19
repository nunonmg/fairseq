import pickle 
import argparse
import itertools
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from sacrebleu import BLEU, CHRF
from Levenshtein import distance as lev
CHRF = CHRF(lowercase=True)
CHRF.BETA = 2

def add_score_dropout_uncertainty_measures():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bicleaner", type=bool, required=False, default=False)
    parser.add_argument("--dist", type=str, required=False, default="lowcomet")
    parser.add_argument("--N_dropout", type=int, required=False, default=16)
    args, _ = parser.parse_known_args()

    lp = args.lp
    ckpt = args.ckpt
    dist = args.dist
    bicleaner = args.bicleaner
    n_dropout = args.N_dropout

    if bicleaner:
        bicleaner_str = "_w_bicleaner"
    else:
        bicleaner_str = ""

    if lp == "de-en":
        datapath = "/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_de-en_heldout/"
    elif lp == "en-ru":
        datapath = "/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_en-ru_heldout/"

    if dist == "lowcomet":
        pkl_path = "/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/score_dropout/lowcomet/" + lp + "/" + ckpt + "/" + ckpt + "/dataframes/dataframe_test_beam5.pkl"
        df_score_dropout = pd.read_pickle(pkl_path)
        savepath = datapath + ckpt + "/mc-dropout-scores-lowcomet" + bicleaner_str
    else:
        pkl_path = "/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/score_dropout/datasamples/" + lp + "/" + ckpt + "/" + ckpt + "/dataframes/dataframe_test_beam5.pkl"
        df_score_dropout = pd.read_pickle(pkl_path)
        savepath = datapath + ckpt + "/mc-dropout-scores-datasamples" + bicleaner_str

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("Aggregating Ensemble Probs")
    score_mts = df_score_dropout.score.values
    score_agg = [score_mts[i:i + n_dropout] for i in range(0, len(score_mts), n_dropout)]
    mean_agg = [round(np.mean(sublist),4) for sublist in score_agg]
    min_agg = [round(np.min(sublist),4) for sublist in score_agg]
    std_agg = [round(np.std(sublist),4) for sublist in score_agg]

    agg_scores = [mean_agg, min_agg, std_agg]
    with open(savepath + "/agg_ensembleprobs_scores" + bicleaner_str + ".pkl", "wb") as f:
        pickle.dump(agg_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    add_score_dropout_uncertainty_measures()