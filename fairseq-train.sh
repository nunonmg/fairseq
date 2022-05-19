SRC_LANG=en
TGT_LANG=de
LP=de-en
CUDA_VISIBLE_DEVICES=1 fairseq-train \
    data-bin/wmt18_${LP}\
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192\
    --update-freq 4 \
    --max-update 250000 \
    --log-interval 10 \
    --seed 42 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --reset-optimizer \
    --eval-bleu-print-samples \
    --keep-last-epochs 10 \
    --save-interval	1 \
    --save-interval-updates 50000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project mt-hallucinations \
    --fp16 \
    --save-dir /home/nunomg/mt-hallucinations/HALO/fairseq/checkpoints/wmt18_$SRC_LANG-${TGT_LANG} \
    --source-lang $SRC_LANG --target-lang $TGT_LANG

