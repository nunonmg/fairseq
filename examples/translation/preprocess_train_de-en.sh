DATA=/home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_de-en
prep=$DATA/prep
src_lang=en
tgt_lang=de

TGT=en
BPESIZE=32000

# #VOCAB W/ JOINT BPE
# TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $DATA/train.${SRC}; echo $DATA/train.${TGT}; done | tr "\n" ",")
# echo "learning joint BPE over ${TRAIN_FILES}..."
# python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_train.py \
#     --input=$TRAIN_FILES \
#     --model_prefix=$DATA/sentencepiece.joint.bpe \
#     --vocab_size=$BPESIZE \
#     --character_coverage=1.0 \
#     --model_type=bpe

# #JOINT ENCODING
# for split in train valid; do
#     python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py \
#             --model $DATA/sentencepiece.joint.bpe.model \
#             --output_format piece \
#             --inputs $DATA/$split.${src_lang} $DATA/$split.${tgt_lang} \
#             --outputs $DATA/$split.jsp.${src_lang} $DATA/$split.jsp.${tgt_lang}
# done

# echo "binarizing..."
# fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
#     --trainpref $DATA/train.jsp --validpref $DATA/valid.jsp \
#     --destdir /home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_de-en \
#     --workers 20 \
#     --joined-dictionary \
#     --bpe sentencepiece

#JOINT ENCODING
for split in heldout; do
    python /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/testfiles/punc_norm.py --data $DATA/$split --src_lang $src_lang --tgt_lang $tgt_lang
    python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py \
            --model $DATA/sentencepiece.joint.bpe.model \
            --output_format piece \
            --inputs $DATA/$split.norm.${src_lang} $DATA/$split.norm.${tgt_lang} \
            --outputs $DATA/$split.jsp.${src_lang} $DATA/$split.jsp.${tgt_lang}
done

echo "binarizing..."
fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref $DATA/train.jsp --validpref $DATA/heldout.jsp \
    --destdir /home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_${src_lang}-${tgt_lang}_heldout \
    --workers 20 \
    --joined-dictionary \
    --bpe sentencepiece