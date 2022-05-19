import argparse
import os

import pandas as pd 
import numpy as np 
import pickle 

from tqdm import tqdm 
from utils.repscore import create_ngram_sentence_list, count_repetitions, n_gram_score, consecutive_words_score, calculate_final_scores


def ids_to_str(list_of_ids: list):
    ids_str = []
    for i in tqdm(range(len(list_of_ids))):
        ids_list = list_of_ids[i]
        ids_str.append(" ".join([str(idx) for idx in ids_list]))
    return ids_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_bin", type=str, required=True)
    parser.add_argument("--format", type=str, required=True, default="bpe")
    args, _ = parser.parse_known_args()
    datapath = args.data_bin
    form = args.format

    print("Loading data")
    df = pd.read_pickle(datapath + "/dataframes/dataframe_test_beam5.pkl")

    if form == "bpe":
        print("Running RepScore w/ BPE Tokens")
        src_ids = ids_to_str(df["src_ids"].values)
        mt_ids = ids_to_str(df["mt_ids"].values)
        ref_ids = ids_to_str(df["ref_ids"].values)

        df["src_ids"] = src_ids
        df["mt_ids"] = mt_ids
        df["ref_ids"] = ref_ids

        n = 2
        ref_sentence_list = df["ref_ids"]
        pred_sentence_list = df["mt_ids"]
    else:
        print("Running RepScore w/ Str Tokens")
        n = 2
        ref_sentence_list = df["ref"]
        pred_sentence_list = df["mt"]

    # Create list of n-grams
    n_gram_ref_sentence_list = create_ngram_sentence_list(ref_sentence_list, n)
    n_gram_pred_sentence_list = create_ngram_sentence_list(pred_sentence_list, n)

    # Create lists with the counts of repetitions of the n-grams
    n_gram_count_ref_sentence_list = count_repetitions(n_gram_ref_sentence_list, 0)
    n_gram_count_pred_sentence_list = count_repetitions(n_gram_pred_sentence_list, 0)

    # Obtain the score for n-gram counts
    n_gram_scores = n_gram_score(n_gram_count_ref_sentence_list, n_gram_count_pred_sentence_list)

    # Obtain repetition score for consecutive words
    if n == 2:
        consec_scores = consecutive_words_score(n_gram_count_ref_sentence_list, n_gram_count_pred_sentence_list)
    else:
        aux_n_gram_ref_sentence_list = create_ngram_sentence_list(ref_sentence_list, 2)
        aux_n_gram_pred_sentence_list = create_ngram_sentence_list(pred_sentence_list, 2)
        aux_n_gram_count_ref_sentence_list = count_repetitions(aux_n_gram_ref_sentence_list, 0)
        aux_n_gram_count_pred_sentence_list = count_repetitions(aux_n_gram_pred_sentence_list, 0)
        consec_scores = consecutive_words_score(aux_n_gram_count_ref_sentence_list, aux_n_gram_count_pred_sentence_list)

    # Calculate list with the final scores
    repetition_scores = calculate_final_scores(n_gram_scores, consec_scores, w1=1, w2=2)
    with open(datapath + '/stats/repscores_' + form + '.pkl', 'wb') as f:
        pickle.dump(repetition_scores, f)


if __name__ == "__main__":
    main()