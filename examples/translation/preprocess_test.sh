TESTFILE=hallucinations
src_lang=de
tgt_lang=en

echo "Preparing $TESTFILE on ${src_lang}-${tgt_lang}"

DATA=/home/nunomg/hallucinations-in-mt/fairseq/examples/translation/testfiles/$TESTFILE/${src_lang}-${tgt_lang}

if [ $src_lang == 'de' ]
then
    #JOINT ENCODING
    for split in test; do
        python /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/testfiles/punc_norm.py --data $DATA/$split --src_lang $src_lang --tgt_lang $tgt_lang 
        python /home/nunomg/hallucinations-in-mt/fairseq/scripts/spm_encode.py \
                --model /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/data_de-en/sentencepiece.joint.bpe.model \
                --output_format piece \
                --inputs $DATA/$split.norm.${src_lang} $DATA/$split.norm.${tgt_lang} \
                --outputs $DATA/$split.jsp.${src_lang} $DATA/$split.jsp.${tgt_lang}
    done

    echo "binarizing..."
    fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/data_${src_lang}-${tgt_lang}/train.jsp --testpref $DATA/test.jsp \
        --destdir /home/nunomg/hallucinations-in-mt/fairseq/data-bin/${TESTFILE}_${src_lang}-${tgt_lang} \
        --workers 20 \
        --bpe sentencepiece \
        --joined-dictionary
fi

if [ $src_lang == 'en' ]
then
    for split in test; do
        python /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/testfiles/punc_norm.py --data $DATA/$split --src_lang $src_lang --tgt_lang $tgt_lang 
        python /home/nunomg/hallucinations-in-mt/fairseq/scripts/spm_encode.py  \
                --model /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/data_en-ru/sentencepiece.bpe.en.model \
                --output_format piece \
                --inputs $DATA/$split.norm.${src_lang} \
                --outputs $DATA/$split.sp.${src_lang} 
        python /home/nunomg/hallucinations-in-mt/fairseq/scripts/spm_encode.py \
                --model /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/data_en-ru/sentencepiece.bpe.ru.model \
                --output_format piece \
                --inputs $DATA/$split.norm.${tgt_lang} \
                --outputs $DATA/$split.sp.${tgt_lang} &
        wait
    done

    echo "binarizing..."
    fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref /home/nunomg/hallucinations-in-mt/fairseq/examples/translation/data_${src_lang}-${tgt_lang}/train.sp --testpref $DATA/test.sp \
        --destdir /home/nunomg/hallucinations-in-mt/fairseq/data-bin/${TESTFILE}_${src_lang}-${tgt_lang} \
        --workers 20 \
        --bpe sentencepiece
fi