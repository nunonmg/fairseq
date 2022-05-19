import argparse
import os
import numpy as np
from sacrebleu import CHRF
import pandas as pd
from tqdm import tqdm
from collections import Counter
from nltk import ngrams
from comet import download_model, load_from_checkpoint
from laserembeddings import Laser
import pickle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def create_dataframe(translations_list: list):
    data = {
        "src": [],
        "mt": [],
        "ref": [],
    }
    for line in translations_list:
        src, mt, ref = line.split("\t")
        data["src"].append(src)
        data["mt"].append(mt)
        data["ref"].append(ref)

    df = pd.DataFrame(data)
    return df


def prepare_data_comet(dataframe):
    data_comet = []
    for i in tqdm(range(len(dataframe))):
        hyp = dataframe["mt"][i]
        ref = dataframe["ref"][i]
        src = dataframe["src"][i]
        assert type(hyp) == str
        assert type(ref) == str
        assert type(src) == str
        dict_sample = {"src": src, "mt": hyp, "ref": ref}
        data_comet.append(dict_sample)
    return data_comet


def compute_metric_similarity():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_bin", type=str, required=True)
    parser.add_argument("--metric_model", type=str, required=True)
    args, _ = parser.parse_known_args()
    metric_model = args.metric_model
    datapath = args.data_bin

    if metric_model == "all":
        compute_comet, compute_comet_qe, compute_laser = True, True, True
    elif metric_model == "comet":
        compute_comet, compute_comet_qe, compute_laser = True, False, False
    elif metric_model == "comet-qe-da":
        compute_comet, compute_comet_qe, compute_laser = False, True, False
    elif metric_model == "laser":
        compute_comet, compute_comet_qe, compute_laser = False, False, True

    print("Loading data")
    df = pd.read_pickle(datapath + "/dataframes/dataframe_test_beam5.pkl")

    if compute_comet:
        print("Computing COMET")
        model_path = download_model("wmt20-comet-da")
        model_comet = load_from_checkpoint(model_path)
        data_comet = prepare_data_comet(df)
        seg_scores, sys_score = model_comet.predict(data_comet, batch_size=16, gpus=1)
        with open(datapath + '/stats/comet_scores.pkl', 'wb') as f:
            pickle.dump(seg_scores, f)

    if compute_comet_qe:
        # print("Computing COMET-QE")
        # model_path = download_model("wmt21-comet-qe-mqm")
        # model_comet = load_from_checkpoint(model_path)
        # data_comet = prepare_data_comet(df)
        # seg_scores, sys_score = model_comet.predict(data_comet, batch_size=16, gpus=1)
        # with open(datapath + '/stats/comet_qe_scores.pkl', 'wb') as f:
        #     pickle.dump(seg_scores, f)

        print("Computing COMET-QE-DA")
        model_path = download_model("wmt20-comet-qe-da-v2")
        model_comet = load_from_checkpoint(model_path)
        data_comet = prepare_data_comet(df)
        seg_scores, sys_score = model_comet.predict(data_comet, batch_size=16, gpus=1)
        with open(datapath + '/stats/comet_qe_da_scores.pkl', 'wb') as f:
            pickle.dump(seg_scores, f)

    if compute_laser:
        print("Computing LASER")
        laser = Laser()
        laser_scores = []
        for i in tqdm(range(len(df))):
            src = df["src"].values[i]
            mt = df["mt"].values[i]
            emb_src, emb_mt = laser.embed_sentences([src, mt], lang=["de", "en"])
            cos_sim = np.dot(emb_src, emb_mt) / (
                np.linalg.norm(emb_src) * np.linalg.norm(emb_mt)
            )
            laser_scores.append(cos_sim)
        with open(datapath + '/stats/laser_scores.pkl', 'wb') as f:
            pickle.dump(laser_scores, f)


if __name__ == "__main__":
    compute_metric_similarity()
