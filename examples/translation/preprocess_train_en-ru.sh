DATA=/home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_en-ru
prep=$DATA/prep
joint_vocab=false
src_lang=en
tgt_lang=ru

SRCS=(
    "en"
)
TGT=ru
BPESIZE=32000

# echo "learning BPE over ${src_lang}..."
# python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_train.py \
#     --input=$DATA/train.$src_lang \
#     --model_prefix=$DATA/sentencepiece.bpe.$src_lang \
#     --vocab_size=$BPESIZE \
#     --character_coverage=1.0 \
#     --model_type=bpe

# echo "learning BPE over ${tgt_lang}..."
# python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_train.py \
#     --input=$DATA/train.$tgt_lang \
#     --model_prefix=$DATA/sentencepiece.bpe.$tgt_lang \
#     --vocab_size=$BPESIZE \
#     --character_coverage=1.0 \
#     --model_type=bpe

# echo "applying sentecepiece model..."
# python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py --model $DATA/sentencepiece.bpe.${src_lang}.model < $DATA/train.${src_lang} > $DATA/train.sp.${src_lang} &
# python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py --model $DATA/sentencepiece.bpe.${tgt_lang}.model < $DATA/train.${tgt_lang} > $DATA/train.sp.${tgt_lang} &
# wait

for split in heldout; do
    #python /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/testfiles/punc_norm.py --data $DATA/$split --src_lang $src_lang --tgt_lang $tgt_lang 
    python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py  \
            --model $DATA/sentencepiece.bpe.en.model \
            --output_format piece \
            --inputs $DATA/$split.norm.${src_lang} \
            --outputs $DATA/$split.sp.${src_lang} 
    python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py \
            --model $DATA/sentencepiece.bpe.ru.model \
            --output_format piece \
            --inputs $DATA/$split.norm.${tgt_lang} \
            --outputs $DATA/$split.sp.${tgt_lang}
done

# echo "binarizing..."
# fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
#     --trainpref $DATA/train.sp --validpref $DATA/valid.sp \
#     --destdir /home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_en-ru \
#     --workers 20 \
#     --bpe sentencepiece

echo "binarizing..."
fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref $DATA/train.sp --validpref $DATA/heldout.sp \
    --destdir /home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/wmt18_en-ru_heldout \
    --workers 20 \
    --bpe sentencepiece

