from typing import List
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

def attn_shape_curation(attn_maps: List, df: pd.Dataframe):
    attn_samples = []
    token_logprobs_samples = []
    for idx in tqdm(df.index.values):
        src_len = len(df.loc[idx]["src_ids"])
        mt_len = len(df.loc[idx]["mt_ids"])
        attn = np.array(attn_maps[idx])
        if attn.shape != (src_len, mt_len):
            attn = attn[~np.all(attn == 0, axis=1)]
        assert attn.shape == (src_len, mt_len)
        attn_samples.append(attn)

