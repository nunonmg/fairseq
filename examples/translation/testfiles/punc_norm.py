import argparse
from mosestokenizer import *
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--src_lang", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)
    args, _ = parser.parse_known_args()
    data = args.data
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    normalize_src = MosesPunctuationNormalizer(src_lang)
    normalize_tgt = MosesPunctuationNormalizer(tgt_lang)

    with open(data + "." + src_lang, "r") as f:
        data_src = f.readlines()

    with open(data + "." + tgt_lang, "r") as f:
        data_tgt = f.readlines()

    data_norm_src = []
    data_norm_tgt = []
    for i in tqdm(range(len(data_src))):
        data_norm_src.append(normalize_src(data_src[i]))
        data_norm_tgt.append(normalize_tgt(data_tgt[i]))
    
    src_file = open(data + ".norm." + src_lang, "w")
    for element in data_norm_src:
        src_file.write(element + "\n")
    src_file.close()

    tgt_file = open(data + ".norm." + tgt_lang, "w")
    for element in data_norm_tgt:
        tgt_file.write(element + "\n")
    tgt_file.close()

if __name__ == "__main__":
    main()