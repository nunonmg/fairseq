# src_lang=de
# tgt_lang=en
# DATA=/home/nunomg/hallucinations-in-mt/fairseq/data-bin/newstest2014_${src_lang}-${tgt_lang}
# CKPT=checkpoint_best
# mkdir $DATA/${CKPT}
# CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA \
#     --path /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/wmt18_de-en/checkpoint_best.pt --source-lang $src_lang --target-lang $tgt_lang\
#     --gen-subset test --beam 5 --batch-size 256 --sacrebleu --scoring sacrebleu --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece | tee $DATA/${CKPT}/avg_gen.out
# grep ^H $DATA/${CKPT}/avg_gen.out | LC_ALL=C sort -V | cut -f3- | sacrebleu --test-set wmt14/full --language-pair ${src_lang}-${tgt_lang}

# DATA=/home/nunomg/hallucinations-in-mt/fairseq/data-bin/wmt18_${src_lang}-${tgt_lang}
# CUDA_VISIBLE_DEVICES=2 fairseq-generate $DATA \
#     --path /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/wmt18_de-en/checkpoint_best.pt --source-lang $src_lang --target-lang $tgt_lang\
#     --gen-subset test --beam 5 --batch-size 256 --seed 42 --sacrebleu --scoring sacrebleu --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --skip-invalid-size-inputs-valid-test | tee $DATA/avg_gen.out
# grep ^H /home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/newstest2017_en-ru/avg_gen.out | LC_ALL=C sort -V | cut -f3- | sacrebleu --test-set wmt17 --language-pair ${src_lang}-${tgt_lang}

# CUDA_VISIBLE_DEVICES=3 python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/qualitymetrics_compute.py --data_bin $DATA --metric_model all
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f1_finder.py --data_bin $DATA --format bpe
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f2_finder.py --data_bin $DATA --format bpe
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/repscore_compute.py --data_bin $DATA/$CKPT


# src_lang=de
# tgt_lang=en
# DATA=data-bin/newstest2017_${src_lang}-${tgt_lang}
# CUDA_VISIBLE_DEVICES=2 fairseq-generate $DATA \
#     --path /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/wmt18_de-en/checkpoint_best.pt --source-lang $src_lang --target-lang $tgt_lang\
#     --gen-subset test --beam 5 --batch-size 256 --sacrebleu --scoring sacrebleu --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece | tee $DATA/${CKPT}/avg_gen.out
# grep ^H /home/nunomg/mt-hallucinations/HALO/fairseq/data-bin/newstest2014_de-en/avg_gen.out | LC_ALL=C sort -V | cut -f3- | sacrebleu --test-set wmt14/full --language-pair de-en

DATA=/home/nunomg/hallucinations-in-mt/fairseq/data-bin/m2m_halls
CUDA_VISIBLE_DEVICES=3 fairseq-generate $DATA \
    --path /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/M2M/418M_last_checkpoint.pt --source-lang de --target-lang en --fixed-dictionary /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/M2M/model_dict.128k.txt\
    --lang-pairs /home/nunomg/hallucinations-in-mt/fairseq/checkpoints/M2M/language_pairs_small_models.txt --decoder-langtok --encoder-langtok src \
    --gen-subset test --batch-size 1 --score-reference --beam 5 --save-attn-maps --remove-bpe=sentencepiece --quiet | tee $DATA/avg_gen.out


# echo "Running Heldout"
# src_lang=de
# tgt_lang=en
# DATA=data-bin/wmt18_${src_lang}-${tgt_lang}_heldout_attn_force
# # DATA=data-bin/hallucinations_de-en
# #DATA=data-bin/newstest2018_de-en
# CKPT=checkpoint_best
# echo "Running generation for $CKPT"
# mkdir $DATA/${CKPT}
# CUDA_VISIBLE_DEVICES=1 fairseq-generate $DATA \
#     --path /mnt/data-poseidon/nunomg/checkpoints/wmt18_${src_lang}-${tgt_lang}/${CKPT}.pt --source-lang $src_lang --target-lang $tgt_lang\
#     --gen-subset valid --beam 5 --score-reference --batch-size 128 --sacrebleu --scoring sacrebleu --save-attn-maps --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --quiet --skip-invalid-size-inputs-valid-test | tee $DATA/${CKPT}/gen.out



# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f1_finder.py --data_bin $DATA/$CKPT
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f2_finder.py --data_bin $DATA/$CKPT
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/repscore_compute.py --data_bin $DATA/$CKPT
# CUDA_VISIBLE_DEVICES=3 python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/qualitymetrics_compute.py --data_bin $DATA/$CKPT --metric_model comet-qe-da

# echo "Running Heldout"
# src_lang=de
# tgt_lang=en
# DATA=data-bin/wmt18_${src_lang}-${tgt_lang}_heldout
# for epoch_no in 55 56 57 58 59 60 61 62 63
# do
#     CKPT=checkpoint${epoch_no}
#     echo "Running generation for $CKPT"
#     mkdir $DATA/${CKPT}
#     CUDA_VISIBLE_DEVICES=3 fairseq-generate $DATA \
#         --path /home/nunomg/mt-hallucinations/HALO/fairseq/checkpoints/wmt18_${src_lang}-${tgt_lang}/${CKPT}.pt --source-lang $src_lang --target-lang $tgt_lang\
#         --gen-subset valid --beam 5 --batch-size 128 --sacrebleu --scoring sacrebleu --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --quiet --skip-invalid-size-inputs-valid-test | tee $DATA/${CKPT}/gen.out

#     python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f1_finder.py --data_bin $DATA/$CKPT
#     python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f2_finder.py --data_bin $DATA/$CKPT
#     python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/repscore_compute.py --data_bin $DATA/$CKPT
#     CUDA_VISIBLE_DEVICES=3 python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/qualitymetrics_compute.py --data_bin $DATA/$CKPT --metric_model comet-qe-da
# done

# echo "Running Heldout"
# src_lang=en
# tgt_lang=ru
# DATA=data-bin/wmt18_${src_lang}-${tgt_lang}_heldout
# for epoch_no in 151 152 153 154 155 156 157 158 159 160
# do
#     CKPT=checkpoint${epoch_no}
#     echo "Running generation for $CKPT"
#     mkdir $DATA/${CKPT}
#     CUDA_VISIBLE_DEVICES=1 fairseq-generate $DATA \
#         --path /home/nunomg/mt-hallucinations/HALO/fairseq/checkpoints/wmt18_${src_lang}-${tgt_lang}/${CKPT}.pt --source-lang $src_lang --target-lang $tgt_lang\
#         --gen-subset valid --beam 5 --batch-size 128 --sacrebleu --scoring sacrebleu --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --quiet --skip-invalid-size-inputs-valid-test | tee $DATA/${CKPT}/gen.out

#     python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f1_finder.py --data_bin $DATA/$CKPT
#     python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f2_finder.py --data_bin $DATA/$CKPT
#     python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/repscore_compute.py --data_bin $DATA/$CKPT
#     CUDA_VISIBLE_DEVICES=1 python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/qualitymetrics_compute.py --data_bin $DATA/$CKPT --metric_model comet-qe-da
# done

# src_lang=en
# tgt_lang=de
# DATA=data-bin/newstest2014_${src_lang}-${tgt_lang}
# CKPT=checkpoint_best
# echo "Running generation for $CKPT"
# mkdir $DATA/${CKPT}
# CUDA_VISIBLE_DEVICES=2 fairseq-generate $DATA \
#     --path /home/nunomg/mt-hallucinations/HALO/fairseq/checkpoints/wmt18_${src_lang}-${tgt_lang}/${CKPT}.pt --source-lang $src_lang --target-lang $tgt_lang\
#     --gen-subset test --beam 5 --batch-size 128 --sacrebleu --scoring sacrebleu --remove-bpe=sentencepiece --eval-bleu-remove-bpe sentencepiece --quiet --skip-invalid-size-inputs-valid-test | tee $DATA/${CKPT}/gen.out

# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f1_finder.py --data_bin $DATA/$CKPT
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f2_finder.py --data_bin $DATA/$CKPT
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/repscore_compute.py --data_bin $DATA/$CKPT

# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f1_finder.py --data_bin $DATA/$CKPT --format str
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/f2_finder.py --data_bin $DATA/$CKPT --format str
# python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/repscore_compute.py --data_bin $DATA/$CKPT --format str

# CUDA_VISIBLE_DEVICES=2 python3 /home/nunomg/hallucinations-in-mt/fairseq/examples/mt-hallucinations/utils/qualitymetrics_compute.py --data_bin $DATA/$CKPT --metric_model comet-qe-da