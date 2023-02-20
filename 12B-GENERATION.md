# Using fairseq to translate with m2m100-12B

In this document, we provide instructions on how to perform generation with m2m100-12B using 2 GPUs. For more information, check the [fairseq documentation](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100).

## Instalation

First, start by installing the following dependencies:

```shell
pip install "omegaconf==2.1.2" "hydra-core==1.0.0" "antlr4-python3-runtime==4.8" "sentencepiece==0.1.97"
```

Then install fairseq:

```shell
git clone https://github.com/deep-spin/fairseq-multi-mt.git
cd fairseq-multi-mt
git checkout -b llm-hallucination-generation
pip install -e .
```

After this, download the following files with the model parameters:

```shell
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
wget https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs.txt
wget https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_2_gpus.pt
```

## Usage

Throughout the following instructions the following variables are assumed to be defined:

```bash
fairseq="path to fairseq home (directory where the repository was cloned)"

model_root="path to m2m100 files (directory where the above files were donwloaded)"
spm_model=${model_root}/spm.128k.model
pt_model=${model_root}/12b_last_chk_2_gpus.pt
data_dict=${model_root}/data_dict.128k.txt
model_dict=${model_root}/model_dict.128k.txt
model_lang_pairs=${model_root}/language_pairs.txt

src_lang="language code for the source language"
tgt_lang="language code for the target language"

data_prefix="prefix for original data"
encode_prefix="prefix for encoded data"
data_bin="directory for binarized data"
```

### Encoding

```bash
for lang in ${src_lang} ${tgt_lang} do;
    python ${fairseq}/scripts/spm_encode.py \
        --model ${spm_model} \
        --output_format=piece \
        --inputs=${data_prefix}.${lang} \
        --outputs=${encode_prefix}.${lang}
done
```

### Binarization

```bash
fairseq-preprocess \
    --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --testpref ${encode_prefix} \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir ${data_bin} \
    --srcdict ${data_dict} --tgtdict ${data_dict}
```

### Generation

```bash
fairseq-generate $original_data_bin \
    --beam 4 \
    --path ${pt_model} \
    --fixed-dictionary ${model_dict} \
    -s ${src_lang} -t ${tgt_lang} \
    --remove-bpe 'sentencepiece' \
    --task translation_multi_simple_epoch \
    --lang-pairs ${model_lang_pairs} \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test \
    --fp16 \
    --dataset-impl mmap \
    --distributed-world-size 1 \
    --distributed-no-spawn \
    --pipeline-model-parallel \
    --pipeline-chunks 1 \
    --pipeline-encoder-balance '[26]' \
    --pipeline-encoder-devices '[0]' \
    --pipeline-decoder-balance '[3,22,1]' \
    --pipeline-decoder-devices '[0,1,0]' \
    --skip-invalid-size-inputs-valid-test
```