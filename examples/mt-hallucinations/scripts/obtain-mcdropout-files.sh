dist=all
src_lang=de
tgt_lang=en
# testfile=wmt18_de-en_heldout
testfile=hallucinations_data
# DF_PATH=/home/nunomg/hallucinations-in-mt/fairseq/data-bin/${testfile}/dataframes/heldout_${dist}_w_bicleaner.pkl
DF_PATH=/home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/annotation_setup/de-en/first_setup/data/hallucinations_data_stats_final.pkl
SAVEPATH=/home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/mc_dropout_files/${testfile}/${dist}
SP_SAVEPATH=/home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/mc_dropout_files/${testfile}/sp_${dist}
CKPT=checkpoint_best

if [ $src_lang == "de" ]
then
    if [ ! -d "$SP_SAVEPATH" ]; then
        rm -r $SAVEPATH
        CUDA_VISIBLE_DEVICES=0 python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/repeat_lines_mcdropout.py --lp ${src_lang}-${tgt_lang} --df_path ${DF_PATH} --savepath ${SAVEPATH}
        echo "Preparing files on ${src_lang}-${tgt_lang}"
        #JOINT ENCODING
        for split in 10_score-dropout; do
            #python /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/testfiles/punc_norm.py --data $SAVEPATH/$split --src_lang $src_lang --tgt_lang $tgt_lang 
            python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py \
                    --model /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_de-en/sentencepiece.joint.bpe.model \
                    --output_format piece \
                    --inputs $SAVEPATH/$split.${src_lang} $SAVEPATH/$split.${tgt_lang} \
                    --outputs $SAVEPATH/$split.jsp.${src_lang} $SAVEPATH/$split.jsp.${tgt_lang}
        done

        echo "binarizing..."
        fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
            --trainpref /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/data_${src_lang}-${tgt_lang}/train.jsp --testpref $SAVEPATH/10_score-dropout.jsp \
            --destdir $SP_SAVEPATH \
            --workers 20 \
            --bpe sentencepiece \
            --joined-dictionary        
    fi
fi

# CUDA_VISIBLE_DEVICES=2 fairseq-generate $SP_SAVEPATH \
#     --path /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/wmt18_${src_lang}-${tgt_lang}/${CKPT}.pt --source-lang $src_lang --target-lang $tgt_lang\
#     --gen-subset test --beam 5 --unkpen 5 --no-length-ordering --seed 42 --retain-dropout --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --skip-invalid-size-inputs-valid-test --quiet | tee $SAVEPATH/gen.out
CUDA_VISIBLE_DEVICES=2 fairseq-generate $SP_SAVEPATH \
    --path /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/wmt18_${src_lang}-${tgt_lang}/${CKPT}.pt --source-lang $src_lang --target-lang $tgt_lang\
    --gen-subset test --beam 5 --unkpen 5 --score-reference --no-length-ordering --seed 42 --save-attn-maps --retain-dropout --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --skip-invalid-size-inputs-valid-test --quiet | tee $SAVEPATH/gen.out