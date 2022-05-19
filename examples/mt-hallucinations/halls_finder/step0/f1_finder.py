import argparse
import os

import pandas as pd 
import numpy as np 
import pickle 

from collections import Counter
from nltk import ngrams
from tqdm import tqdm 

def ids_to_str(list_of_ids: list):
    ids_str = []
    for i in tqdm(range(len(list_of_ids))):
        ids_list = list_of_ids[i]
        ids_str.append(" ".join([str(idx) for idx in ids_list]))
    return ids_str

def finder_f1(
    src_list: list, mt_list: list, threshold: int = 2, ngram_order: int = 4
):
    matches = []
    n_samples = len(src_list)
    for i in tqdm(range(n_samples)):
        src_sample = src_list[i]
        mt_sample = mt_list[i]
        ngram_counts_src = Counter(ngrams(src_sample.split(), ngram_order))
        ngram_counts_mt = Counter(ngrams(mt_sample.split(), ngram_order))
        if (
            ngram_counts_src.most_common(1) != []
            and ngram_counts_mt.most_common(1) != []
        ):
            repeated_counts_src = ngram_counts_src.most_common(1)[0][1]
            repeated_counts_mt = ngram_counts_mt.most_common(1)[0][1]
            if repeated_counts_mt > repeated_counts_src + threshold:
                matches.append(1)
            else:
                matches.append(0)
        else:
            matches.append(0)
    return matches

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
        print("Running F1 w/ BPE Tokens")
        src = ids_to_str(df["src_ids"].values)
        mt = ids_to_str(df["mt_ids"].values)
    else:
        print("Running F1 w/ original Tokens")
        src = df["src"].values
        mt = df["mt"].values

    matches = finder_f1(src, mt)

    with open(datapath + '/stats/f1_scores_' + form + '.pkl', 'wb') as f:
        pickle.dump(matches, f)



if __name__ == "__main__":
    main()
