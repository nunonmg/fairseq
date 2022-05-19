src_lang=de
tgt_lang=en
CKPT=checkpoint_best
DATA_DIST=datasamples
DATA=/home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/score-dropout/${DATA_DIST}/${src_lang}-${tgt_lang}/${CKPT}
DATA_GEN=/home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/score_dropout/${DATA_DIST}/${src_lang}-${tgt_lang}/${CKPT}

if [ $src_lang == 'de' ]
then
    rm -r $DATA
    CUDA_VISIBLE_DEVICES=0 python3 /home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/score_hypothesis_prep.py --lp ${src_lang}-${tgt_lang} --ckpt $CKPT --dist ${DATA_DIST}
    echo "Preparing $TESTFILE on ${src_lang}-${tgt_lang}"
    #JOINT ENCODING
    for split in score-dropout; do
        python /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/testfiles/punc_norm.py --data $DATA/$split --src_lang $src_lang --tgt_lang $tgt_lang 
        python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py \
                --model /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_de-en/sentencepiece.joint.bpe.model \
                --output_format piece \
                --inputs $DATA/$split.norm.${src_lang} $DATA/$split.norm.${tgt_lang} \
                --outputs $DATA/$split.jsp.${src_lang} $DATA/$split.jsp.${tgt_lang}
    done

    echo "binarizing..."
    fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_${src_lang}-${tgt_lang}/train.jsp --testpref $DATA/score-dropout.jsp \
        --destdir $DATA_GEN \
        --workers 20 \
        --bpe sentencepiece \
        --joined-dictionary
fi

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_GEN \
    --path /home/nunomg/mt-hallucinations/HALO/fairseq/checkpoints/wmt18_${src_lang}-${tgt_lang}/${CKPT}.pt --source-lang $src_lang --target-lang $tgt_lang\
    --gen-subset test --beam 5 --batch-size 16 --score-reference --retain-dropout --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --skip-invalid-size-inputs-valid-test --quiet | tee $DATA_GEN/gen.out

# if [ $src_lang == 'en' ]
# then
#     for split in score-dropout; do
#         python /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/testfiles/punc_norm.py --data $DATA/$split --src_lang $src_lang --tgt_lang $tgt_lang 
#         python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py  \
#                 --model /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_en-ru/sentencepiece.bpe.en.model \
#                 --output_format piece \
#                 --inputs $DATA/$split.norm.${src_lang} \
#                 --outputs $DATA/$split.sp.${src_lang} 
#         python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py \
#                 --model /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_en-ru/sentencepiece.bpe.ru.model \
#                 --output_format piece \
#                 --inputs $DATA/$split.norm.${tgt_lang} \
#                 --outputs $DATA/$split.sp.${tgt_lang} &
#         wait
#     done

#     echo "binarizing..."
#     fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
#         --trainpref /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_${src_lang}-${tgt_lang}/train.sp --testpref $DATA/test.sp \
#         --destdir /home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/${TESTFILE}_${src_lang}-${tgt_lang} \
#         --workers 20 \
#         --bpe sentencepiece
# fi
