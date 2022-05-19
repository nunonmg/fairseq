LP=de-en
CKPT=checkpoint_best
DIST=lowcomet
CUDA_VISIBLE_DEVICES=0 python3 /home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/dropout_lowcometqe.py --lp $LP --ckpt $CKPT --bicleaner True
CUDA_VISIBLE_DEVICES=0 python3 /home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/divergence_similarity_compute.py --lp $LP --ckpt $CKPT  --metrics levdist --dist $DIST --bicleaner True
CUDA_VISIBLE_DEVICES=0 python3 /home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/score_hypothesis_prep.py --lp $LP --ckpt $CKPT --dist $DIST --bicleaner True
CUDA_VISIBLE_DEVICES=0 python3 /home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/score_hypothesis_add.py --lp $LP --ckpt $CKPT --dist $DIST --bicleaner True

# LP=en-ru
# CKPT=checkpoint_best
# CUDA_VISIBLE_DEVICES=2 python3 /home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/dropout_lowcometqe.py --lp $LP --ckpt $CKPT
#CUDA_VISIBLE_DEVICES=1 python3 /home/nunomg/mt-hallucinations/HALO/fairseq/halls_finder/step0/divergence_similarity_compute.py --lp $LP --ckpt $CKPT
