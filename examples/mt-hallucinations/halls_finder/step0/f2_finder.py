import argparse
import os

import pandas as pd 
import numpy as np 
import pickle 

from tqdm import tqdm 

def ids_to_str(list_of_ids: list):
    ids_str = []
    for i in tqdm(range(len(list_of_ids))):
        ids_list = list_of_ids[i]
        ids_str.append(" ".join([str(idx) for idx in ids_list]))
    return ids_str

def finder_f2(dataframe, repeated_mts_df, form):
    if form == "bpe":
        unique_mts = list(set(list(repeated_mts_df["mt_ids"])))
        score_mts = []
        for j in tqdm(range(len(unique_mts))):
            mt = unique_mts[j]
            score_mts.append({"mt": mt, "flag": 0})
            matches_mt = dataframe.loc[dataframe["mt_ids"] == mt]["src_ids"]
            if len(set(matches_mt)) > 1:
                score_mts[j]["flag"] = 1
            else:
                score_mts[j]["flag"] = 0
    else:
        unique_mts = list(set(list(repeated_mts_df["mt"])))
        score_mts = []
        for j in tqdm(range(len(unique_mts))):
            mt = unique_mts[j]
            score_mts.append({"mt": mt, "flag": 0})
            matches_mt = dataframe.loc[dataframe["mt"] == mt]["src"]
            if len(set(matches_mt)) > 1:
                score_mts[j]["flag"] = 1
            else:
                score_mts[j]["flag"] = 0

    return score_mts

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
        print("Running F2 w/ BPE Tokens")
        src_ids = ids_to_str(df["src_ids"].values)
        mt_ids = ids_to_str(df["mt_ids"].values)
        ref_ids = ids_to_str(df["ref_ids"].values)

        df["src_ids"] = src_ids
        df["mt_ids"] = mt_ids
        df["ref_ids"] = ref_ids

        sorted_df = df.iloc[
        df.groupby("mt_ids").mt.transform("size").mul(-1).argsort(kind="mergesort")
        ]
        sorted_df["counts"] = sorted_df["mt_ids"].map(df["mt_ids"].value_counts())
        repeated_mts = sorted_df.loc[(sorted_df["counts"] > 1)]

        print("Finding F2 samples")

        score_mts = finder_f2(df, repeated_mts, form)
        dataframe_score_mts = df.from_dict(score_mts)

        matches_f2 = []
        for j in tqdm((df.index)):
            mt = df["mt_ids"][j]
            flag = dataframe_score_mts.loc[dataframe_score_mts["mt"] == mt][
                "flag"
            ].values
            if len(flag) == 0:
                matches_f2.append(0)
            else:
                if flag == 1:
                    matches_f2.append(1)
                else:
                    matches_f2.append(0)
    
    else:
        print("Running F2 w/ Str Tokens")
        sorted_df = df.iloc[
        df.groupby("mt").mt.transform("size").mul(-1).argsort(kind="mergesort")
        ]
        sorted_df["counts"] = sorted_df["mt"].map(df["mt"].value_counts())
        repeated_mts = sorted_df.loc[(sorted_df["counts"] > 1)]

        print("Finding F2 samples")

        score_mts = finder_f2(df, repeated_mts, form)
        dataframe_score_mts = df.from_dict(score_mts)

        matches_f2 = []
        for j in tqdm((df.index)):
            mt = df["mt"][j]
            flag = dataframe_score_mts.loc[dataframe_score_mts["mt"] == mt][
                "flag"
            ].values
            if len(flag) == 0:
                matches_f2.append(0)
            else:
                if flag == 1:
                    matches_f2.append(1)
                else:
                    matches_f2.append(0)

    with open(datapath + '/stats/f2_scores_' + form + '.pkl', 'wb') as f:
        pickle.dump(matches_f2, f)


if __name__ == "__main__":
    main()