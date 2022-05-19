from fairseq.models.transformer import TransformerModel
import pandas as pd
import pickle 
import numpy as np
import argparse
import os
import torch
from collections import Counter
from tqdm import tqdm
from nltk import ngrams
from comet import download_model, load_from_checkpoint
from HALO.fairseq.halls_finder.step0.utils.repscore import create_ngram_sentence_list, count_repetitions, n_gram_score, consecutive_words_score, calculate_final_scores
torch.multiprocessing.set_sharing_strategy('file_system')

def compute_comet(dict_samples: dict, model_comet):
    seg_scores, _ = model_comet.predict(dict_samples, batch_size=16, gpus=1)
    return seg_scores

def compute_repscore(pred_sentence_list: list, ref_sentence_list: list, n: int = 2):
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
    return repetition_scores

def finder_f1(src_list: list, mt_list: list, threshold: int = 2, ngram_order: int = 4):
        matches = []
        n_samples = len(src_list)
        for i in range(n_samples):
            src_sample = src_list[i]
            mt_sample = mt_list[i]
            ngram_counts_src = Counter(ngrams(src_sample.split(), ngram_order))
            ngram_counts_mt = Counter(ngrams(mt_sample.split(), ngram_order))
            if ngram_counts_src.most_common(1) != [] and ngram_counts_mt.most_common(1) != []:
                repeated_counts_src = ngram_counts_src.most_common(1)[0][1]
                repeated_counts_mt = ngram_counts_mt.most_common(1)[0][1]
                if repeated_counts_mt > repeated_counts_src + threshold:
                    matches.append(1)
                else:
                    matches.append(0)
            else:
                matches.append(0)
        return matches

def list_ids_to_str_ids(list_ids: list):
    return [" ".join([str(elem) for elem in sublist]) for sublist in list_ids]

def generate_mc_dropout():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bicleaner", type=bool, required=True)
    parser.add_argument("--dist", type=str, required=False, default="lowcomet")
    parser.add_argument("--N_dropout", type=int, required=False, default=16)
    args, _ = parser.parse_known_args()
    lp = args.lp
    ckpt = args.ckpt
    n_dropout = args.N_dropout
    dist = args.dist
    bicleaner = args.bicleaner

    model_path = download_model("wmt20-comet-qe-da-v2")
    model_comet = load_from_checkpoint(model_path)
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

    if dist != "data":
        df_path = datapath + ckpt + "/dataframes/heldout_lowcomet" + bicleaner_str + ".pkl"
        savepath = datapath + ckpt + "/mc-dropout-gens-lowcomet" + bicleaner_str
    else:
        df_path = datapath + ckpt + "/dataframes/datasampleswstats" + bicleaner_str + ".pkl"
        savepath = datapath + ckpt + "/mc-dropout-gens-datasamples" + bicleaner_str

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    trainedmodel = TransformerModel.from_pretrained(
        ckpt_path,
        checkpoint_file=ckpt + ".pt",
        data_name_or_path=datapath,
        retain_dropout=True,
    )
    tgt_dict = trainedmodel.task.target_dictionary

    trainedmodel.cuda()

    print("Generating MC Dropout for Hallucinations")
    df = pd.read_pickle(df_path)
    lowcomet_mc_dropout_gen_dicts = []

    for idx in tqdm(df["idx"].values):
        sample = df.loc[df["idx"]==idx]
        sentences = [torch.tensor(sample["src_ids"].values[0])]
        tokenized_sentences = sentences * n_dropout
        translation = trainedmodel.generate(tokenized_sentences, beam=5)

        output_dict = {}
        output_dict["sequences_tokens"] = [hypos[0]["tokens"].cpu().numpy() for hypos in translation]
        output_dict["sequences"] = [tgt_dict.string(example, return_list=True, extra_symbols_to_ignore=[tgt_dict.pad()], bpe_symbol="sentencepiece") for example in output_dict["sequences_tokens"]]
        output_dict["sequence_scores"] = [hypos[0]["score"].cpu().numpy() for hypos in translation]

        src_txt = [sample["src"].values[0]] * n_dropout
        mt_txt = output_dict["sequences"]
        ref_txt = [sample["ref"].values[0]] * n_dropout
        dict_text = {"src": src_txt, "mt": mt_txt, "ref": ref_txt}

        dict_comet = [dict(zip(dict_text,t)) for t in zip(*dict_text.values())]
        output_dict["dict_for_comet"] = dict_comet
        lowcomet_mc_dropout_gen_dicts.append(output_dict)

    print("Computation of MTs has ended.")

    mts_dropout = [gen["sequences"] for gen in lowcomet_mc_dropout_gen_dicts]
    with open(savepath + "/mts-word-dropout" + bicleaner_str + ".pkl", "wb") as f:
        pickle.dump(mts_dropout, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("MTs in text have been saved.")

    mts_ids_dropout = [gen["sequences_tokens"] for gen in lowcomet_mc_dropout_gen_dicts]
    with open(savepath + "/mts-ids-dropout" + bicleaner_str + ".pkl", "wb") as f:
        pickle.dump(mts_ids_dropout, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("MTs in BPE IDs have been saved.")

    seq_scores = [gen["sequence_scores"] for gen in lowcomet_mc_dropout_gen_dicts]
    with open(savepath + "/seq-scores-mts-dropout" + bicleaner_str + ".pkl", "wb") as f:
        pickle.dump(seq_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Sequence Scores for MTs have been saved.")

    for j in range(len(lowcomet_mc_dropout_gen_dicts)):
        lowcomet_mc_dropout_gen_dicts[j].pop("sequences")
        lowcomet_mc_dropout_gen_dicts[j].pop("sequences_tokens")
        lowcomet_mc_dropout_gen_dicts[j].pop("sequence_scores")

    print("Computing COMET-QE")
    dicts_for_comet = [output_dict["dict_for_comet"] for output_dict in lowcomet_mc_dropout_gen_dicts]
    dicts_for_comet = [item for sublist in dicts_for_comet for item in sublist]
    comet_vals = compute_comet(dicts_for_comet, model_comet)
    for j in range(len(lowcomet_mc_dropout_gen_dicts)):
        lowcomet_mc_dropout_gen_dicts[j]["comet"] = comet_vals[j*n_dropout:j*n_dropout+n_dropout]
        lowcomet_mc_dropout_gen_dicts[j].pop("dict_for_comet")

    comet_mts_dropout = [gen["comet"] for gen in lowcomet_mc_dropout_gen_dicts]
    with open(savepath + "/comet-mts-dropout" + bicleaner_str + ".pkl", "wb") as f:
        pickle.dump(comet_mts_dropout, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("COMET scores have been saved.")

    for j in range(len(lowcomet_mc_dropout_gen_dicts)):
        lowcomet_mc_dropout_gen_dicts[j].pop("comet")

    with open(savepath + "/dicts" + bicleaner_str + ".pkl", "wb") as f:
        pickle.dump(lowcomet_mc_dropout_gen_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("All other stats saved.")


if __name__ == "__main__":
    generate_mc_dropout()