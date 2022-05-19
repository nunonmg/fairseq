import os
import pickle 
import argparse
import itertools

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from sacrebleu import BLEU
from Levenshtein import distance as lev


def compute_d_lex_sim():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bicleaner", type=bool, required=False, default=False)
    parser.add_argument("--dist", type=str, required=False, default="lowcomet")
    parser.add_argument("--metrics", type=str, required=False, default="levdist")
    args, _ = parser.parse_known_args()

    lp = args.lp
    ckpt = args.ckpt
    dist = args.dist
    bicleaner = args.bicleaner
    metrics = args.metrics

    bertscore = "bertscore" in metrics.split()
    meteor = "meteor" in metrics.split()
    levdist = "levdist" in metrics.split()

    if bicleaner:
        bicleaner_str = "_w_bicleaner"
    else:
        bicleaner_str = ""

    if lp == "de-en":
        datapath = "/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_de-en_heldout/"
    elif lp == "en-ru":
        datapath = "/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_en-ru_heldout/"

    if dist == "lowcomet":
        pkl_path = datapath + ckpt + "/mc-dropout-gens-lowcomet" + bicleaner_str + "/" + "mts-word-dropout" + bicleaner_str + ".pkl"
        savepath = datapath + ckpt + "/mc-dropout-scores-lowcomet" + bicleaner_str
        df = pd.read_pickle(datapath + ckpt + "/dataframes/heldout_lowcomet" + bicleaner_str + ".pkl")
    else:
        pkl_path = datapath + ckpt + "/mc-dropout-gens-datasamples/dicts" + bicleaner_str + ".pkl"
        savepath = datapath + ckpt + "/mc-dropout-scores-datasamples" + bicleaner_str
        df = pd.read_pickle(datapath + ckpt + "/dataframes/datasampleswstats" + bicleaner_str + ".pkl")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    with open(pkl_path, "rb") as f:
        mts_dropout = pickle.load(f)

    print("MC-dropout Stats Loaded.")

    #mts_dropout = [gen["sequences"] for gen in mc_d_gen_dicts]
    mc_d_gen_dicts = []

    metric_meteor = load_metric("meteor")

    if bertscore:
        metric_bertscore = load_metric("bertscore")
        d_lex_sim_bertscore_avg = []
        d_lex_sim_bertscore_min = []
        d_lex_sim_bertscore_std = []

    if meteor:
        metric_meteor = load_metric("meteor")
        d_lex_sim_meteor_avg = []
        d_lex_sim_meteor_min = []
        d_lex_sim_meteor_std = []

    if levdist:
        d_lex_sim_levdist_avg = []
        d_lex_sim_levdist_max = []
        d_lex_sim_levdist_std = []

    for idx in tqdm(range(len(mts_dropout))):
        mts = mts_dropout[idx] + [df.mt.values[idx]]
        h_set = list(itertools.combinations(mts, 2))
        len_mts = [len(mt) for mt in mts]
        #h_set = [(df.mt.values[idx], mc_dropout_translation) for mc_dropout_translation in mts_dropout[idx]]
        references = [pair[0] for pair in h_set]
        predictions = [pair[1] for pair in h_set]
        
        if bertscore:
            metric_bertscore.add_batch(predictions=predictions, references=references)
            metric_vals = metric_bertscore.compute(lang="en")["f1"]
            d_lex_sim_bertscore_avg.append(round(np.mean(metric_vals),4))
            d_lex_sim_bertscore_min.append(round(np.min(metric_vals), 4))
            d_lex_sim_bertscore_std.append(round(np.std(metric_vals),4))

        if meteor:
            sentence_meteor_vals = []
        if levdist:
            sentence_levdist_vals = []
        for j in range(len(h_set)):
            if meteor:
                metric_meteor.add_batch(predictions=[predictions[j]], references=[references[j]])
                sentence_meteor_vals.append(metric_meteor.compute()["meteor"])
            if levdist:
                sentence_levdist_vals.append(lev(predictions[j], references[j]))
        
        if levdist:
            d_lex_sim_levdist_avg.append(round(np.mean(sentence_levdist_vals), 4))
            d_lex_sim_levdist_max.append(round(np.max(sentence_levdist_vals), 4))
            d_lex_sim_levdist_std.append(round(np.std(sentence_levdist_vals), 4))

        if meteor:
            d_lex_sim_meteor_avg.append(round(np.mean(sentence_meteor_vals), 4))
            d_lex_sim_meteor_min.append(round(np.min(sentence_meteor_vals), 4))
            d_lex_sim_meteor_std.append(round(np.std(sentence_meteor_vals), 4))

    if meteor:
        agg_scores_meteor = [d_lex_sim_meteor_avg, d_lex_sim_meteor_min, d_lex_sim_meteor_std]
        with open(savepath + "/agg_scores.meteor" + bicleaner_str + ".pkl", "wb") as f:
            pickle.dump(agg_scores_meteor, f, protocol=pickle.HIGHEST_PROTOCOL)

    if bertscore:
        agg_scores_bertscore = [d_lex_sim_bertscore_avg, d_lex_sim_bertscore_min, d_lex_sim_bertscore_std]
        with open(savepath + "/agg_scores.bertscore" + bicleaner_str + ".pkl", "wb") as f:
            pickle.dump(agg_scores_bertscore, f, protocol=pickle.HIGHEST_PROTOCOL)

    if levdist:
        agg_scores_levdist = [d_lex_sim_levdist_avg, d_lex_sim_levdist_max, d_lex_sim_levdist_std]
        with open(savepath + "/agg_scores.levdist" + bicleaner_str + ".pkl", "wb") as f:
            pickle.dump(agg_scores_levdist, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    compute_d_lex_sim()