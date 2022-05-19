dist=datasamples
src_lang=de
tgt_lang=en
DF_PATH=/home/nunomg/hallucinations-in-mt/fairseq/data-bin/wmt18_de-en_heldout/dataframes/heldout_${dist}_w_bicleaner.pkl
SAVEPATH=/home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/halls_finder/mc_dropout_files/${dist}
SP_SAVEPATH=/home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/halls_finder/mc_dropout_files/sp_${dist}
CKPT=checkpoint_best

if [ $src_lang == "de" ]
then
    if [ ! -d "$SP_SAVEPATH" ]; then
        rm -r $SAVEPATH
        CUDA_VISIBLE_DEVICES=0 python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/halls_finder/step0/repeat_lines_mcdropout.py --lp ${src_lang}-${tgt_lang} --df_path ${DF_PATH} --savepath ${SAVEPATH}
        echo "Preparing files on ${src_lang}-${tgt_lang}"
        #JOINT ENCODING
        for split in score-dropout; do
            python /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/testfiles/punc_norm.py --data $SAVEPATH/$split --src_lang $src_lang --tgt_lang $tgt_lang 
            python /home/nunomg/mt-hallucinations/HALO/fairseq/scripts/spm_encode.py \
                    --model /home/nunomg/mt-hallucinations/HALO/fairseq/examples/translation/data_de-en/sentencepiece.joint.bpe.model \
                    --output_format piece \
                    --inputs $SAVEPATH/$split.norm.${src_lang} $SAVEPATH/$split.norm.${tgt_lang} \
                    --outputs $SAVEPATH/$split.jsp.${src_lang} $SAVEPATH/$split.jsp.${tgt_lang}
        done

        echo "binarizing..."
        fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
            --trainpref /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/data_${src_lang}-${tgt_lang}/train.jsp --testpref $SAVEPATH/score-dropout.jsp \
            --destdir $SP_SAVEPATH \
            --workers 20 \
            --bpe sentencepiece \
            --joined-dictionary        
    fi
fi

CUDA_VISIBLE_DEVICES=2 fairseq-generate $SP_SAVEPATH \
    --path /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/wmt18_${src_lang}-${tgt_lang}/${CKPT}.pt --source-lang $src_lang --target-lang $tgt_lang\
    --gen-subset test --beam 5 --unkpen 5 --no-length-ordering --seed 42 --retain-dropout --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --skip-invalid-size-inputs-valid-test --quiet | tee $SAVEPATH/gen.out

rm -r $SAVEPATH